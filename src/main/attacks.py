import torch


def pgd_linf(loss_func, model_attack, X, y, delta_attack=1.0, alpha=0.01, attack_epoch=15, randomize=False):
    """
    Generate adversarial perturbations using the Projected Gradient Descent (PGD) w.r.t the infinity norm

    Parameters
    ----------
    loss_func : loss function
    model_attack : model to be attacked
    X : batch of unperturbed input samples
    y : batch of labels
    delta_attack : constraint for the amount of disturbances delta
    alpha : step size for gradient ascent on the disturbances delta
    attack_epoch : number of update steps for finding worst-case disturbances delta
    randomize : whether to randomly initialize delta. If False, delta is initialized to a batch of zeros (default)

    Returns
    -------
    delta : batch of adversarial perturbations
    """
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = 2 * (delta.data - 0.5) * delta_attack
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for _ in range(attack_epoch):
        yp = model_attack(X + delta)
        loss = torch.mean(loss_func(yp, y))
        loss.backward()

        # gradient ascent on perturbation delta
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-delta_attack, delta_attack)
        delta.grad.zero_()

    return delta.detach()


def fgsm_linf(loss_func, model_attack, X, y, delta_attack=1.0, randomize=False):
    """
    Generate adversarial perturbations using the Fast Gradient Sign Method (FGSM) w.r.t the infinity norm

    Parameters
    ----------
    loss_func : loss function
    model_attack : model to be attacked
    X : batch of unperturbed input samples
    y : batch of labels
    delta_attack : constraint for the amount of disturbances delta
    randomize : whether to randomly initialize delta. If False, delta is initialized to a batch of zeros (default)

    Returns
    -------
    delta : batch of adversarial perturbations
    """

    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = 2 * (delta.data - 0.5) * delta_attack
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    yp = model_attack(X + delta)
    loss = torch.mean(loss_func(yp, y))
    loss.backward()

    return delta_attack * delta.grad.detach().sign()

