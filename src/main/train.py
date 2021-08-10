import datetime
import torch.optim as optim
import wandb
import warnings
from torch.optim.swa_utils import AveragedModel

from src.arguments import ArgumentParser
from src.main.methods import *
from src.utils.model import build_model
from src.utils.config import *
from src.utils.data import *
from src.utils.train_utils import *

warnings.filterwarnings('ignore')

INPUT_DIMS = {'fashion_mnist': 784, 'cifar_10': 1024, 'celeba': 3078}
OUTPUT_DIMS = {'fashion_mnist': 10, 'cifar_10': 10, 'celeba': 2}

if __name__ == '__main__':

    # ============== SET-UP ARGS, DEVICE, SEED, LOGGING ============== #
    args = ArgumentParser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_seed(args.seed)

    # set up model name for logging and tracking
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    model_name = suffix + '_' + args.model_class + '_' + args.alg_name + '_' + str(args.seed)
    config_logging(log_name=model_name, save=args.save_local, save_dir=args.save_dir)

    # log args and cuda availability
    logging.info(f'Current directory: {os.getcwd()}')
    logging.info(f'Command line args: \n {args}')
    logging.info(f'Torch version {torch.__version__}. CUDA available: {torch.cuda.is_available()}. '
                 f'CUDA version {torch.version.cuda}')

    if args.data in ['fashion_mnist', 'cifar_10', 'celeba']:
        train_loader = deep_dataloader(data=args.data,
                                       train=True,
                                       shuffle=True,  # shuffling at each iteration through the data
                                       augment=args.augment,
                                       batch_size=args.batch_size,
                                       seed=args.seed,
                                       save_dir=args.save_dir)

        test_loader = deep_dataloader(data=args.data,
                                      train=False,
                                      shuffle=True,  # shuffling only once at initialisation (i.e. for each seed)
                                      batch_size=args.batch_size,
                                      seed=args.seed,
                                      save_dir=args.save_dir)
    else:
        raise NotImplementedError(f"invalid data {args.data}")

    logging.info('Data loaded!')

    model = build_model(output_dim=OUTPUT_DIMS[args.data],
                        model_class=args.model_class,
                        activation=args.activation,
                        device=device)

    logging.info(f'Model: \n {model}')  # log prediction model
    model_swa = AveragedModel(model) if args.model_swa else None  # model with averaged weights
    loss_func = get_loss(reduction=args.reduction)

    #  ==============  SET-UP OPTIMIZER ================ #
    opt_params = model.parameters()
    opt = get_optimizer(var=opt_params,
                        opt_name=args.opt_name,
                        lr=args.lr,
                        weight_decay=args.weight_decay)

    if args.decay_lr and args.data == 'cifar_10':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, [25, 50], 0.1)

    #  ==============  SET-UP WANDB ================ #
    if args.wandb:
        group_name = wandb_config(args)  # unique identifier to group model runs across seeds
        wandb.init(project=args.wandb_project_name, name=model_name, reinit=True,
                   config={'group': group_name}, dir=args.save_dir)
        wandb.config.update(args)

    # ============== TRAIN & TEST ============== #
    logging.info('Training with: %s', args.alg_name)
    for i in range(1, args.num_epochs + 1):

        if args.alg_name == 'erm':
            train_err, train_loss, time = epoch_ERM(loss_func=loss_func,
                                                    loader=train_loader,
                                                    opt=opt,
                                                    model=model,
                                                    model_swa=model_swa,
                                                    device=device)

            test_err, test_loss = test(loss_func=loss_func,
                                       loader=test_loader,
                                       model=model_swa if args.model_swa else model,
                                       device=device)

            # logging and wandb tracking
            if i % args.log_interval == 0:
                format_str = 'Epoch: {} | Train error: {: .3%} | Train loss: {: .6f} | Test error: {: .3%} | ' \
                             'Test loss: {: .6f} | Time: {: .2f} '
                logging.info(format_str.format(*[i, train_err, train_loss, test_err, test_loss, time]))
                if args.wandb:
                    wandb.log({'train error': train_err, 'train loss': train_loss,
                               'test error': test_err, 'test loss': test_loss}, step=i)

        elif args.alg_name == 'arks':
            train_err, train_loss, train_adv_loss, surr_loss, K, time = \
                epoch_ARKS(sigma=args.sigma,
                            lr_inner=args.lr_inner,
                            loss_func=loss_func,
                            loader=train_loader,
                            model=model,
                            model_swa=model_swa,
                            opt=opt,
                            num_epoch_inner=args.num_epoch_inner,
                            device=device,
                            decay_lr_inner=args.decay_lr_inner)

            test_err, test_loss = test(loss_func=loss_func,
                                       loader=test_loader,
                                       model=model_swa if args.model_swa else model,
                                       device=device)

            # logging and wandb tracking
            if i % args.log_interval == 0:
                format_str = 'Epoch {} | Train error {: .3%} | Train loss {: .6f} | Adv loss {: .6f} ' \
                             '| Surr Loss {: .6f} | K {: .6f} | Test error {: .3%} | Test loss {: .6f} | Time {: .3f}'
                logging.info(format_str.format(*[i, train_err, train_loss, train_adv_loss, surr_loss, K,
                                                 test_err, test_loss, time]))
                if args.wandb:
                    wandb.log({'train error': train_err, 'train loss': train_loss, 'adversarial loss': train_adv_loss,
                               'surrogate loss': surr_loss, 'test error': test_err, 'test loss': test_loss}, step=i)

        elif args.alg_name == 'wrm':
            train_err, train_loss, train_adv_loss, surr_loss, rho, time = \
                    epoch_WRM(gamma=args.gamma,
                              lr_inner=args.lr_inner,
                              loss_func=loss_func,
                              loader=train_loader,
                              model=model,
                              model_swa=model_swa,
                              opt=opt,
                              num_epoch_inner=args.num_epoch_inner,
                              device=device,
                              decay_lr_inner=args.decay_lr_inner)
            test_err, test_loss = test(loss_func=loss_func,
                                       loader=test_loader,
                                       model=model_swa if args.model_swa else model,
                                       device=device)

            # logging and wandb tracking
            if i % args.log_interval == 0:
                format_str = 'Epoch {} | Train error {: .3%} | Train loss {: .6f} | Adv loss {: .6f} | ' \
                             'Surr Loss {: .6f} | Rho {: .6f} | Test error {: .3%} | Test loss {: .6f} | | Time {: .3f}'
                logging.info(format_str.format(*[i, train_err, train_loss, train_adv_loss, surr_loss, rho,
                                                 test_err, test_loss, time]))
                if args.wandb:
                    wandb.log({'train error': train_err, 'train loss': train_loss, 'adversarial loss': train_adv_loss,
                               'surrogate loss': surr_loss, 'test error': test_err, 'test loss': test_loss}, step=i)

        elif args.alg_name == 'pgd':
            train_err, train_loss, train_adv_loss, time = \
                epoch_PGD(lr_inner=args.lr_inner,
                          delta_attack=args.delta,
                          num_epoch_inner=args.num_epoch_inner,
                          loss_func=loss_func,
                          loader=train_loader,
                          model=model,
                          model_swa=model_swa,
                          opt=opt,
                          device=device)
            test_err, test_loss = test(loss_func=loss_func,
                                       loader=test_loader,
                                       model=model_swa if args.model_swa else model,
                                       device=device)

            # logging and wandb tracking
            if i % args.log_interval == 0:
                format_str = 'Epoch {} | Train error {: .3%} | Train loss {: .6f} | Adv loss {: .6f} | ' \
                             'Test error {: .3%} | Test loss {: .6f} | | Time {: .3f}'
                logging.info(format_str.format(*[i, train_err, train_loss, train_adv_loss, test_err, test_loss, time]))
                if args.wandb:
                    wandb.log({'train error': train_err, 'train loss': train_loss, 'test error': test_err,
                               'test loss': test_loss}, step=i)
        else:
            raise NotImplementedError(f"invalid algorithm name: {args.alg_name}")

        if args.decay_lr:
            if 'cifar' in args.data:
                lr_scheduler.step()
            else:
                adjust_lr(opt, args.lr, i, args.num_epochs)

    # ============== DONE TRAINING ============== #
    if args.save_model:
        save_path = model_save(model=model, args=args)
        logging.info(f'Saved model path: {save_path}')

    # ============== EVALUATE ============== #
    if args.evaluate:

        for k, attack_name in enumerate(args.attack_list):

            logging.info(f"\nEvaluating on: {attack_name}")
            delta_list = np.arange(0, 0.31, 0.01)
            display_list = [0, (len(delta_list) - 1) // 4, (len(delta_list) - 1) // 2]

            if args.alg_attack:
                model_attack, model_path = model_load(args=args, alg=args.alg_attack, device=device)
                logging.info(f'Attack model path: {model_path}')
            else:
                model_attack = None

            for j, delta_attack in enumerate(delta_list):
                #  log sample test images
                record_test_images_this_time = args.record_test_images and j in display_list

                # create a simple loss to create an attack
                adv_err, adv_loss, fig = epochAttack(data=args.data,
                                                     loader=test_loader,
                                                     model=model,
                                                     model_attack=model_attack,
                                                     loss_func=loss_func,
                                                     attack=attack_name,
                                                     delta_attack=delta_attack,
                                                     alpha=args.alpha,
                                                     attack_epoch=args.attack_epoch,
                                                     record_test_images=record_test_images_this_time,
                                                     device=device)
                if j % 5 == 0:
                    format_str = 'Attack: {} | Delta: {} | Adv error: {:.3%} | Adv loss: {:.6f}'
                    logging.info(format_str.format(*[attack_name, delta_attack, adv_err, adv_loss]))

                if args.wandb:
                    wandb.log({f'test error on {attack_name}_{k} {args.model_class} attack': adv_err,
                               f'test loss on {attack_name}_{k} {args.model_class} attack': adv_loss,
                               'delta': delta_attack * 100})  # multiplying by 100 due to wandb requiring delta > 1
                    if record_test_images_this_time:
                        wandb.log({f'{group_name}_{delta_attack}_{attack_name}_{args.model_class}': fig})