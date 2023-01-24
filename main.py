import numpy as np
import argparse
import torch
from poisoned_dataset import create_backdoor_data_loader
from models import get_model, loss_picker, optimizer_picker
from utils import backdoor_model_trainer, save_experiments, path_name

parser = argparse.ArgumentParser(description='Backdoor attack')
parser.add_argument('--dataname', type=str, default='cifar10',
                    choices=['mnist', 'cifar10', 'tinyimagenet'],
                    help='The dataset to use')
parser.add_argument('--model', type=str, default='alexnet', choices=[
                    'resnet', 'googlenet', 'vgg', 'alexnet'], help='The model to use')
parser.add_argument('--pretrained', action='store_true',
                    help='Use pretrained weights')
parser.add_argument('--load-model', action='store_true',
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
parser.add_argument('--batch_size', type=int,
                    default=128, help='Train batch size')
parser.add_argument('--batch_size_test', type=int,
                    default=128, help='Test batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--device', type=str, default='cpu', help='Device to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--datadir', type=str, default='./data',
                    help='path to save downloaded data')
parser.add_argument('--pretrained_path', type=str,
                    help='path to save downloaded pretrained model')
parser.add_argument('--save_path', type=str, default="./experiments",
                    help='path to save training results')
args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    path = path_name(args)

    # Define path to save downloaded pretrained model
    pretrained_path = args.pretrained_path

    # Choose the device to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get the model, loss and optimizer
    model = get_model(args.model, pretrained=args.pretrained,
                      dataname=args.dataname, load_model=args.load_model,
                      path=path, pretrained_path=pretrained_path)

    model = model.to(device)

    optimizer = optimizer_picker(args.optimizer, model.parameters(), args.lr)
    criterion = loss_picker(args.loss)
    poison_trainloader, clean_testloader, poison_testloader = create_backdoor_data_loader(
        args.dataname, args.trigger_label, args.epsilon, args.pos,
        args.shape, args.color, args.trigger_size, args.batch_size, args.batch_size_test, device, args)

    list_train_loss, list_train_acc, list_test_loss, list_test_acc, list_test_loss_backdoor, list_test_acc_backdoor = backdoor_model_trainer(
        model, criterion, optimizer, args.epochs, poison_trainloader, clean_testloader, poison_testloader,
        device, args)

    # Save the results
    save_experiments(args, list_train_acc, list_train_loss, list_test_acc, list_test_loss, list_test_acc_backdoor,
                     list_test_loss_backdoor, model)


if __name__ == '__main__':
    main()
