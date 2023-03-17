from SystematicBackdoor import SystematicBackdoor
import argparse

parser = argparse.ArgumentParser(description='Systematic Backdoor Attack')


# Training arguments
parser.add_argument('--dataname', type=str, default='cifar10',
                    choices=['mnist', 'cifar10', 'tinyimagenet'],
                    help='The dataset to use')
parser.add_argument('--model', type=str, default='googlenet', choices=[
                    'resnet', 'googlenet', 'vgg', 'alexnet'], help='The model to use')
parser.add_argument('--pretrained', action='store_true',
                    help='Use pretrained model')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss', type=str, default='cross',
                    help='The loss function to use', choices=['mse', 'cross'])
parser.add_argument('--optimizer', type=str, default='adam',
                    help='The optimizer to use', choices=['adam', 'sgd'])
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum for SGD optimizer')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay for SGD optimizer')
parser.add_argument('--batch_size', type=int,
                    default=128, help='Train batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--datadir', type=str, default='./data',
                    help='path to save downloaded data')
parser.add_argument('--amp', action='store_true',
                    help='Use automatic mixed precision')
parser.add_argument('--pretrained_path', type=str, default='/home/jxu8/Code/SystematicBackdoors/pretrained_models',
                    help='path to save downloaded pretrained model')
parser.add_argument('--save_path', type=str, default="./experiments",
                    help='path to save training results')
parser.add_argument('--load_model', type=str, default=None,
                    help='path to load model')
parser.add_argument('--load_attack', type=str, default=None,
                    help='path to load attack')

# Attack arguments
subparsers = parser.add_subparsers(dest='mode')
attack_parser = subparsers.add_parser('attack', help='Attack help')
attack_parser.add_argument('--type', type=str, default=None,
                           help='Type of the attack', choices=['badnets', 'ssba', 'wanet'])

# Common arguments for all attacks
attack_parser.add_argument('--target_label', type=int, default=0,
                           help='The label of the target/objective class. The class to be changed to.')
attack_parser.add_argument('--epsilon', type=float, default=0.1,
                           help='The rate of poisoned data')


# Badnets arguments
badnets_parser = attack_parser.add_argument_group('Badnets')
badnets_parser.add_argument('--pos', type=str, default='top-left',
                            choices=['top-left', 'top-right', 'bottom-left',
                                     'bottom-right', 'middle', 'random'],
                            help='The position of the trigger')
badnets_parser.add_argument('--color', type=str, default='white',
                            choices=['white', 'black', 'green'], help='The color of the trigger')
badnets_parser.add_argument('--trigger_size', type=float, default=0.1,
                            help='The size of the trigger in percentage of the image size')

# SSBA arguments
ssba_parser = attack_parser.add_argument_group('SSBA')

# WANet arguments
wanet_parser = attack_parser.add_argument_group('WANet')


# Defense arguments
defense_parser = subparsers.add_parser('defense', help='Defense help')
defense_parser.add_argument('--type', type=str, default=None,
                            help='Type of the defense', choices=['neuralcleanse', 'fine-pruning'])

# Common arguments for all defenses

# NeuralCleanse arguments
neuralcleanse_parser = defense_parser.add_argument_group('NeuralCleanse')

neuralcleanse_parser.add_argument("--nc_lr", type=float, default=1e-1)
neuralcleanse_parser.add_argument("--nc_init_cost", type=float, default=1e-3)
neuralcleanse_parser.add_argument(
    "--nc_atk_succ_threshold", type=float, default=99.0)
neuralcleanse_parser.add_argument("--nc_early_stop",  action='store_true')
neuralcleanse_parser.add_argument(
    "--nc_early_stop_threshold", type=float, default=99.0)
neuralcleanse_parser.add_argument(
    "--nc_early_stop_patience", type=int, default=5)
neuralcleanse_parser.add_argument("--nc_patience", type=int, default=5)
neuralcleanse_parser.add_argument(
    "--nc_cost_multiplier", type=float, default=2)
neuralcleanse_parser.add_argument("--nc_epochs", type=int, default=3)
neuralcleanse_parser.add_argument("--nc_epsilon", type=float, default=1e-7)
neuralcleanse_parser.add_argument("--nc_n_times_test", type=int, default=1)

# neuralcleanse_parser.add_argument('--anomaly_threshold', type=float, default=2,
#                                   help='The threshold of anomaly score')
# neuralcleanse_parser.add_argument('--patience', type=int, default=10,
#                                   help='The patience of the early stopping')
# neuralcleanse_parser.add_argument('--asr_threshold', type=int, default=99.0,
#                                   help='The threshold of the ASR')

# neuralcleanse_parser.add_argument('--nc_epochs', type=int, default=50,
#                                   help='The number of epochs for NeuralCleanse')
# neuralcleanse_parser.add_argument('--nc_epsilon', type=float, default=1,
#                                   help='The epsilon for NeuralCleanse')
# neuralcleanse_parser.add_argument('--nc_lr', type=float, default=1e-1,
#                                   help='The learning rate for NeuralCleanse')
# neuralcleanse_parser.add_argument('--nc_n_test', type=int, default=5,
#                                   help='The number of test for NeuralCleanse')
# neuralcleanse_parser.add_argument('--nc_init_cost', type=float, default=1e-3,
#                                   help='The initial cost for NeuralCleanse')
# neuralcleanse_parser.add_argument('--nc_cost_multiplier', type=float, default=2,
#                                   help='The cost multiplier for NeuralCleanse')
# neuralcleanse_parser.add_argument('--nc_early_stop_threshold', type=float, default=99.0,
#                                   help='The early stop threshold for NeuralCleanse')

# Fine-pruning arguments
fine_pruning_parser = defense_parser.add_argument_group('Fine-pruning')
fine_pruning_parser.add_argument('--pruning_rate', type=float, default=0.1,
                                 help='The rate of neurons to be pruned')


args = parser.parse_args()


def main():
    """
    Main function
    """

    sb = SystematicBackdoor(
        args
    )

    # sb.trainer.train()
    # sb.attack.execute_attack()
    # sb.attack.save_results()
    # sb.attack.save_attack()
    sb.defense.execute_defense()
    # sb.defense.save_results()
    # sb.trainer.save_trainer()


if __name__ == '__main__':
    main()
