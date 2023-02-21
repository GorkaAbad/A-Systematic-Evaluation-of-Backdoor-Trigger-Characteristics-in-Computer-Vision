from SystematicBackdoor import SystematicBackdoor
import argparse

parser = argparse.ArgumentParser(description='Backdoor attack')
parser.add_argument('--dataname', type=str, default='cifar10',
                    choices=['mnist', 'cifar10', 'tinyimagenet'],
                    help='The dataset to use')
parser.add_argument('--attack', type=str, default='badnets',
                    choices=['badnets'], help='The attack to use')
parser.add_argument('--defense', type=str, default=None,
                    choices=['neuralcleanse'], help='The defense to use')
parser.add_argument('--model', type=str, default='alexnet', choices=[
                    'resnet', 'googlenet', 'vgg', 'alexnet'], help='The model to use')
parser.add_argument('--pretrained', action='store_true',
                    help='Use pretrained weights')
parser.add_argument('--load_model', type=str, default=None,
                    help='Load a saved model from the corresponding folder')
parser.add_argument('--epsilon', type=float, default=0.1,
                    help='The rate of poisoned data')
parser.add_argument('--pos', type=str, default='top-left',
                    choices=['top-left', 'top-right', 'bottom-left',
                             'bottom-right', 'middle', 'random'],
                    help='The position of the trigger')
parser.add_argument('--shape', type=str, default='square', choices=['square', 'random'],
                    help='The shape of the trigger')
parser.add_argument('--color', type=str, default='white',
                    choices=['white', 'black', 'green'], help='The color of the trigger')
parser.add_argument('--trigger_size', type=float, default=0.1,
                    help='The size of the trigger in percentage of the image size')
parser.add_argument('--trigger_label', type=int, default=0,
                    help='The label of the target/objective class. The class to be changed to.')
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
parser.add_argument('--batch_size_test', type=int,
                    default=128, help='Test batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--device', type=str, default='cpu', help='Device to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--datadir', type=str, default='./data',
                    help='path to save downloaded data')
parser.add_argument('--amp', action='store_true',
                    help='Use automatic mixed precision')
parser.add_argument('--pretrained_path', type=str, default='/home/jxu8/Code/SystematicBackdoors/pretrained_models',
                    help='path to save downloaded pretrained model')
parser.add_argument('--save_path', type=str, default="./experiments",
                    help='path to save training results')
args = parser.parse_args()


def main():
    """
    Main function
    """

    sb = SystematicBackdoor(
        args
    )

    # sb.trainer.train()
    sb.attack.execute_attack()
    sb.attack.save_results()
    sb.trainer.save_trainer()


if __name__ == '__main__':
    main()
