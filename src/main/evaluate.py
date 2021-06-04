import datetime
import torch.optim as optim
import wandb
import warnings
from torch.optim.swa_utils import AveragedModel

from src.arguments import ArgumentParser
from src.main.methods import *
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
        test_loader = deep_dataloader(data=args.data,
                                      train=False,
                                      shuffle=True,  # shuffling only once at initialisation (i.e. for each seed)
                                      batch_size=args.batch_size,
                                      seed=args.seed,
                                      save_dir=args.save_dir)
    else:
        raise NotImplementedError(f"invalid data {args.data}")

    logging.info('Data loaded!')

    loss_func = get_loss(reduction=args.reduction)

    #  ==============  SET-UP WANDB ================ #
    if args.wandb:
        group_name = wandb_config(args)  # unique identifier to group model runs across seeds
        wandb.init(project=args.wandb_project_name, name=model_name, reinit=True,
                   config={'group': group_name}, dir=args.save_dir)
        wandb.config.update(args)

    # ============== LOAD MODEL ============== #
    model, model_path = model_load(args=args, alg=args.alg_name, device=device)
    logging.info(f'Evaluation model path: {model_path}')
    logging.info(f'Evaluation model: {model}')

    # ============== EVALUATE ============== #
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
