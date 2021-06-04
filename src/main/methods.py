import numpy as np
import time
import torch

from src.main.attacks import pgd_linf, fgsm_linf
from src.utils.plot import plot_images
from src.utils.train_utils import adjust_lr, compute_batch_error, get_optimizer


def epoch_ARKS(loss_func, loader, model=None, model_swa=None, sigma=0.5, opt=None, opt_name_inner='amsgrad',
               lr_inner=0.01, num_epoch_inner=15, decay_lr_inner=False, device='cpu'):
    """
    One training step of Adversarially Robust Kernel Smoothing (ARKS)

    Parameters
    ----------
    loss_func : loss function
    loader : data loader object with mini-batches (X, y)
    model : model to be trained
    model_swa : stochastic weight averaged model
    sigma : kernel bandwidth
    opt : optimizer for model weights
    opt_name_inner : optimizer for finding worst-case perturbations U
    lr_inner : learning rate for inner optimization
    num_epoch_inner : number of iterations in the inner optimization
    decay_lr_inner : whether to decay the learning rate in the inner optimization
    device : gpu if available, otherwise cpu

    Returns
    -------
    total_err / len(loader.sampler) : average train error
    np.average(losses) : average train loss
    np.average(adv_losses) : average adversarial loss corresponding to predictions on worst-case perturbations u^*
    np.average(surr_losses) : average value of the surrogate loss at the end of the inner optimization
    np.average(Ks) : average value of Kernel at the end of the inner optimization
    epoch_time : average time per epoch
    """
    epoch_start_time = time.perf_counter()

    losses, adv_losses, surr_losses, Ks = [], [], [], []
    total_err = 0.

    for X, y in loader:
        model.train()

        X, y = X.to(device), y.to(device)

        #  ============  INNER OPTIMIZATION: FIND WORST-CASE PERTURBATIONS U ============== #
        U = X.data.clone().requires_grad_()
        opt_inner = get_optimizer(var=[U], lr=lr_inner, opt_name=opt_name_inner)

        for epoch in range(num_epoch_inner):
            opt_inner.zero_grad()

            yp = model(U)
            loss_u = loss_func(yp, y)  # loss on perturbed input U
            K = rbf_kernel_torch(U.flatten(1), X.flatten(1), sigma=sigma)

            surr_loss = -torch.dot(torch.diag(K), loss_u) / len(loss_u)  # surrogate loss; - because we want to maximize
            surr_loss.backward()
            opt_inner.step()  # gradient ascent on phi w.r.t. U

            if decay_lr_inner:
                adjust_lr(opt_inner, lr0=lr_inner, epoch=epoch, method='sqrt')
        U.detach_()

        #  ============  OUTER OPTIMIZATION: OPTIMIZE MODEL WEIGHTS ============== #
        opt.zero_grad()
        K = rbf_kernel_torch(U.flatten(1), X.flatten(1), sigma=sigma)
        yp_adv = model(U)
        loss_u_adv = loss_func(yp_adv, y)
        adv_loss = torch.dot(torch.diag(K), loss_u_adv) / len(loss_u_adv)  # surrogate loss
        adv_loss.backward()
        opt.step()

        #  ============  EVALUATE ON UNPERTURBED TRAINING DATA X ============== #
        loss, err = eval_train(model, X, y, loss_func, model_swa)

        #  ============  LOG STATS ============== #
        total_err += err
        losses.append(loss.item())
        adv_losses.append(adv_loss.item())
        surr_losses.append(surr_loss.item())
        Ks.append(torch.mean(K.data.detach()).item())

    epoch_end_time = time.perf_counter()
    epoch_time = epoch_end_time - epoch_start_time
    return total_err / len(loader.sampler), np.average(losses), np.average(adv_losses), \
           np.average(surr_losses), np.average(Ks), epoch_time


def rbf_kernel_torch(U, X, sigma=1.):
    """
    PyTorch implementation of Radial Basis Function (RBF) kernel.

    The RBF kernel is a stationary kernel. It is also known as the
    "Squared Exponential" kernel. It is parameterized by the bandwidth
    parameter sigma, which controls the width of the kernel.

    Parameters
    ----------
    U : tensor of shape (num_samples, num_features)
        Left argument of the returned kernel k(U, X)
    X : tensor of shape (num_samples, num_features)
        Right argument of the returned kernel k(U, X)
    sigma : kernel bandwidth

    Returns
    -------
    K : kernel K(U, X)
    """
    K = torch.cdist(U, X) ** 2
    K *= - (0.5 / sigma**2)
    K = torch.exp(K)
    return K


def epoch_ERM(loss_func, loader, model=None, model_swa=None, opt=None, device='cpu'):
    """
    One training step of Empirical Risk Minimization (ERM)

    Parameters
    ----------
    loss_func : loss function
    loader : data loader object with mini-batches (X, y)
    model : model to be trained
    model_swa : stochastic weight averaged model
    opt : optimizer for model weights
    device : gpu if available, otherwise cpu

    Returns
    -------
    total_err / len(loader.sampler) : average train error
    np.average(losses) : average train loss
    epoch_time : average time per epoch
    """
    epoch_start_time = time.perf_counter()
    total_err = 0
    losses = []

    model.train()

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        opt.zero_grad()
        yp = model(X)
        loss = torch.mean(loss_func(yp, y))
        loss.backward()
        opt.step()

        if model_swa:  # model with averaged weights
            model_swa.update_parameters(model)
            with torch.no_grad():
                yp = model_swa(X)
                loss = torch.mean(loss_func(yp, y))

        total_err += compute_batch_error(yp, y)  # log stats
        losses.append(loss.item())

    epoch_end_time = time.perf_counter()
    epoch_time = epoch_end_time - epoch_start_time
    return total_err / len(loader.sampler), np.average(losses), epoch_time


def epoch_WRM(loss_func, loader, model=None, model_swa=None, gamma=1.0, opt=None, opt_name_inner='amsgrad',
              lr_inner=0.01, num_epoch_inner=15, decay_lr_inner=False, device='cpu'):
    """
    One training step of the Wasserstein Risk Method (WRM) introduced in the 'Certifying Some Distributional Robustness
    with Principled Adversarial Training' paper by Sinha et al. This is our own implementation in PyTorch.

    Parameters
    ----------
    loss_func : loss function
    loader : data loader object with mini-batches (X, y)
    model : model to be trained
    model_swa : stochastic weight averaged model
    gamma : Lagrangian penalty coefficient
    opt : optimizer for model weights
    opt_name_inner : optimizer for finding worst-case perturbations U
    lr_inner : learning rate for inner optimization
    num_epoch_inner : number of iterations in the inner optimization
    decay_lr_inner : whether to decay the learning rate in the inner optimization
    device : gpu if available, otherwise cpu

    Returns
    -------
    total_err / len(loader.sampler) : average train error
    np.average(losses) : average train loss
    np.average(adv_losses) : average adversarial loss corresponding to predictions on worst-case perturbations u^*
    np.average(surr_losses) : average value of the surrogate loss at the end of the inner optimization
    np.average(costs) : average value of the Wasserstein cost at the end of the inner optimization
    epoch_time : average time per epoch
    """
    epoch_start_time = time.perf_counter()
    losses, adv_losses, surr_losses, costs = [], [], [], []
    total_err = 0.

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        model.train()

        #  ============  INNER OPTIMIZATION: FIND WORST-CASE PERTURBATIONS Z ============== #
        Z = X.data.clone().requires_grad_()  # sample initial z_hat from P_0
        opt_inner = get_optimizer(var=[Z], lr=lr_inner, opt_name=opt_name_inner)

        for epoch in range(num_epoch_inner):
            opt_inner.zero_grad()

            yp = model(Z)
            loss_inner = loss_func(yp, y)  # loss on perturbed input Z

            delta = Z - X
            cost = torch.norm(delta.view(len(X), -1), p=2, dim=1) ** 2  # cost c(Z,X) = ||Z-X||^2_2

            surr_loss = - (loss_inner - gamma * cost)  # surrogate loss phi = l(Z) - g * cost; - as we want to maximize
            surr_loss = torch.mean(surr_loss)
            surr_loss.backward()
            opt_inner.step()  # gradient ascent  on phi w.r.t. Z

            if decay_lr_inner:
                adjust_lr(opt=opt_inner, lr0=lr_inner, epoch=epoch, method='sqrt')
        Z.detach_()

        #  ============  OUTER OPTIMIZATION: OPTIMIZE MODEL WEIGHTS ============== #
        opt.zero_grad()
        yp_adv = model(Z)
        adv_loss = torch.mean(loss_func(yp_adv, y))
        adv_loss.backward()
        opt.step()

        #  ============  EVALUATE ON UNPERTURBED TRAINING DATA X ============== #
        loss, err = eval_train(model, X, y, loss_func, model_swa)

        #  ============  LOG STATS ============== #
        total_err += err  # error on unperturbed training data
        losses.append(loss.item())
        adv_losses.append(adv_loss.item())
        surr_losses.append(surr_loss.item())  # surrogate loss at the end of the inner optimization for each batch
        costs.append(torch.mean(cost.data.detach()).item())

    epoch_end_time = time.perf_counter()
    epoch_time = epoch_end_time - epoch_start_time
    return total_err / len(loader.sampler), np.average(losses), np.average(adv_losses), \
           np.average(surr_losses), np.average(costs), epoch_time


def eval_train(model, X, y, loss_func, model_swa=None):
    """
    Evaluates the model on a batch of unperturbed input samples X

    Parameters
    ----------
    model : model being trained
    X : batch of input samples
    y : batch of labels
    loss_func : loss function
    model_swa : stochastic weight averaged model

    Returns
    -------
    loss : loss averaged across the batch samples
    err : classification error averaged across the batch samples
    """
    if model_swa:  # if stochastic weight averaging, get model predictions by averaged model
        model_swa.update_parameters(model)
        with torch.no_grad():
            yp = model_swa(X)
            loss = torch.mean(loss_func(yp, y))
    else:
        model.eval()
        with torch.no_grad():
            yp = model(X)
            loss = torch.mean(loss_func(yp, y))
    err = compute_batch_error(yp, y)
    return loss, err


def test(loss_func, loader, model=None, device='cpu'):
    """
    Tests the trained model on a batch of test samples.
    Use torch.no_grad() to disable gradient computations for fast inference

    Parameters
    ----------
    loss_func : loss function
    loader : data loader object with mini-batches (X, y)
    model : evaluation model
    device : gpu if available, otherwise cpu

    Returns
    -------
    total_err / len(loader.sampler) : average test error
    np.average(losses) : average test loss
    """
    total_err = 0
    losses = []

    model.eval()

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # get predictions and loss
        with torch.no_grad():
            yp = model(X)
            loss = torch.mean(loss_func(yp, y))

        # log error and loss
        total_err += compute_batch_error(yp, y)
        losses.append(loss.item())
    return total_err / len(loader.sampler), np.average(losses)


def epochAttack(data, loader, loss_func, model, model_attack=None, delta_attack=0.1, alpha=0.03, attack_epoch=15,
                attack='pgd_linf', record_test_images=False, device='cpu'):
    """
    One iteration of adversarial attack. Generates disturbances delta of maximum magnitude delta_attack. Currently
    supports attacks generated using Projected Gradient Descent (PGD) and the Fast Gradient Sign Method (FGSM) with
    respect to the L-infinity norm.

    Parameters
    ----------
    data : dataset name
    loss_func : loss function
    loader : data loader object with mini-batches (X, y)
    model : evaluation model
    model_attack : model used to generate adversarial perturbations
    delta_attack : maximum magnitude of the disturbances delta
    attack_epoch : number of update steps for finding worst-case disturbances delta
    alpha : step-size for delta updates
    attack : type of attack (pgd_linf or fgsm_linf)
    record_test_images: whether to display a grid of sample perturbed images with predictions
    device : gpu if available, otherwise cpu

    Returns
    -------
    total_err / len(loader.sampler) : average test error on perturbed images
    np.average(losses) : average test loss on perturbed images
    fig : figure with grid of perturbed images
    """
    total_err, losses = 0., []
    fig = None

    if model_attack:
        model_attack.eval()
        model.eval()
    else:
        model.eval()

    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        if 'pgd_linf' in attack:
            delta = pgd_linf(loss_func=loss_func,
                             model_attack=model_attack if model_attack else model,
                             X=X, y=y,
                             delta_attack=delta_attack,
                             alpha=alpha,
                             attack_epoch=attack_epoch,
                             randomize=False)
        elif 'fgsm_linf' in attack:
            delta = fgsm_linf(loss_func=loss_func,
                              model_attack=model_attack if model_attack else model,
                              X=X, y=y,
                              delta_attack=delta_attack,
                              randomize=False)
        else:
            raise NotImplementedError(f"invalid attack name: {attack}")

        # perturb input images
        Z = X.data.clone() + delta

        # restrict space to real images
        Z.clamp_(0, 1)

        # evaluate on target model after generating the attack
        with torch.no_grad():
            yp = model(Z)
            loss = torch.mean(loss_func(yp, y))

        # plot sample images with labels
        if i == 0 and record_test_images:
            fig = plot_images(X=Z, y=y, yp=yp, M=4, N=8, data=data)

        # record stats
        losses.append(loss.item())
        total_err += compute_batch_error(yp, y)

        # to speed up computation, evaluate on 20 first-batches only for CIFAR
        if data == 'cifar_10' and i == 19:
            break

    return total_err / len(loader.sampler), np.average(losses), fig
