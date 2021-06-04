import numpy as np
import torch
import torch.nn as nn


def compute_batch_error(yp, y):
    """
    Computes the count of wrong predictions for a batch of samples

    Parameters
    ----------
    yp : predictions (before activation)
    y : labels

    Returns
    -------
    total_err
    """
    errors = yp.max(dim=1)[1] != y
    total_error = errors.sum().item()
    return total_error


def get_loss(reduction=False):
    """
    Configures and returns the loss function. At the moment
    only the Cross Entropy Loss is supported.

    Parameters
    ----------
    reduction : whether to average the loss across the batch

    Returns
    -------
    loss_func : loss function
    """
    reduction = 'mean' if reduction else 'none'
    loss_func = nn.CrossEntropyLoss(reduction=reduction)
    return loss_func


def get_optimizer(var, opt_name, lr, weight_decay=0):
    """
    Configures and returns the loss function. At the moment
    only the Cross Entropy Loss is supported.

    Parameters
    ----------
    var : optimization parameter(s)
    opt_name : optimization method
    lr : learning rate
    weight_decay : weight decay in L2 penalty

    Returns
    -------
    opt : optimizer
    """
    if opt_name == 'sgd':
        opt = torch.optim.SGD(var, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif opt_name == 'adam':
        opt = torch.optim.Adam(var, lr=lr)
    elif opt_name == 'amsgrad':
        opt = torch.optim.Adam(var, lr=lr, amsgrad=True)
    else:
        raise NotImplementedError(f'invalid optimisation method: {opt_name}')
    return opt


def adjust_lr(opt, lr0, epoch, total_epochs=None, method=None):
    """
    Decays the learning rate at every epoch

    Parameters
    ----------
    opt : optimizer for parameters to be optimised
    lr0 : initial learning rate
    epoch : training iteration number (note, it starts from 1)
    total_epochs: total number of training epoch
    method : 'sqrt' for decay 1/sqrt(epoch) otherwise decay 0.1**(epoch / total_epochs)
    """

    if method == 'sqrt':
        lr = lr0 * (1.0 / np.sqrt(epoch + 1))
    else:
        lr = lr0 * (0.1 ** (epoch / float(total_epochs)))

    for param_group in opt.param_groups:
        param_group['lr'] = lr