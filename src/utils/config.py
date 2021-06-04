import os
import sys
import logging

import numpy as np
import torch
import random


def init_seed(seed: int = 0):
    """
    Sets the seed of random number generators

    Parameters
    ----------
    seed : int, 0 by default
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_logging(log_name: str = None, save: str = False, save_dir: str = None):
    """
    Configures logging.

    Parameters
    ----------
    log_name : name of .csv file with logs
    save : if True, the logs are saved to a .csv file
    save_dir : the logs are saved in `.{save_dir}/logs'
    """
    name = (log_name if log_name else 'log') + '.log'

    if save:
        save_path = os.path.join(save_dir, 'logs')
        os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, name)

        handlers = [logging.FileHandler(filename=file_path),
                    logging.StreamHandler(sys.stdout)]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=handlers
    )


def model_save(model, args):
    """
    Saves the trained model in the directory ./{save_dir}/models/{data}/{alg_name}

    Parameters
    ----------
    model : trained model
    args : arguments

    Returns
    -------
    save_path : path for the save directory of the model
    """
    sub_path = f'models/{args.data}/{args.alg_name}'

    # create save directory if it does not exist
    save_path = os.path.join(args.save_dir, sub_path)
    os.makedirs(save_path, exist_ok=True)

    model_name = f'model_{args.alg_name}'
    if args.alg_name == 'arks':
        model_name += f'_{args.sigma}'
    if args.alg_name == 'wrm':
        model_name += f'_{args.gamma}'
    model_name += f'_{args.seed}.pt'

    save_path = os.path.join(save_path, model_name)
    torch.save(model, save_path)
    return save_path


def model_load(args, alg, device='cpu'):
    """
    Loads model for further generating attacks or evaluating on attacks

    Parameters
    ----------
    args : arguments
    device : on which device to load the model; equals GPU if cuda enabled

    Returns
    -------
    model_attack : model used to generate adversarial perturbations
    """
    sub_path = f'models/{args.data}/{alg}'
    load_path = os.path.join(args.save_dir, sub_path)

    model_name = f'model_{alg}'
    if alg == 'arks':
        model_name += f'_{args.sigma}'
    if alg == 'wrm':
        model_name += f'_{args.gamma}'
    model_name += f'_{args.seed}.pt'

    model_path = os.path.join(load_path, model_name)
    model_attack = torch.load(model_path, map_location=device).to(device)
    return model_attack, model_path


def wandb_config(args):
    """
    Wandb configuration of group name. This acts as a reference for the
    grouping of model instances trained with the same method and hyper-parameters,
    but different random seeds. This allows to aggregate results across seeds.

    Parameters
    ----------
    args : arguments

    Returns
    -------
    group_name : reference name for grouping instances of a model under different random seeds
    """

    group_name = f'{args.alg_name}_{args.model_class}_{args.activation}_{args.lr}'

    if args.alg_name == 'arks':
        group_name += f'_{args.lr_inner}_{args.sigma}_{args.num_epoch_inner}'

    elif args.alg_name == 'wrm':
        group_name += f'_{args.lr_inner}_{args.gamma}_{args.num_epoch_inner}'

    if args.decay_lr:
        group_name += '_dlr'
    if args.decay_lr_inner:
        group_name += '_dlri'
    if args.model_swa:
        group_name += '_swa'
    return group_name
