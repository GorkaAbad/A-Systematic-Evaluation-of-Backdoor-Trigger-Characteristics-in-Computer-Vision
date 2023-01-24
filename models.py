import os

import torch

from torchvision.models import resnet152, googlenet, vgg19_bn, alexnet, VGG, ResNet, GoogLeNet, AlexNet
from torchvision.models import ResNet152_Weights, VGG19_BN_Weights, GoogLeNet_Weights, AlexNet_Weights
from torch import nn, optim, load, flatten
from utils import path_name

import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Callable, Any


class BasicConv2d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Net(nn.Module):
    """
    Define a dummy simple model to test fast the pipeline.
    The model was taken from: https://nextjournal.com/gkoehler/pytorch-mnist

    TODO: For now this has only been tested with mnist (1-channel).
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def get_model(model_name='resnet', pretrained=True, dataname='mnist',
              load_model=False, path="", pretrained_path=""):
    '''
    For a given model name, return the model

    Parameters:
        model_name (str): name of the model
        pretrained (bool): whether to use pretrained weights
        dataname (str): name of the dataset
        load_model (bool): If we load the model from the harddrive
        path (str): The path of the saved experiments.
        pretrained_path (str): The path of the pretrained model.
    Returns:
        model (torch.nn.Module): model
    '''

    if dataname == 'mnist':
        num_classes = 10
        n_channels = 3
    elif dataname == 'cifar10':
        num_classes = 10
        n_channels = 3
    elif dataname == 'tinyimagenet':
        num_classes = 200
        n_channels = 3
    else:
        raise NotImplementedError(
            f'{dataname} dataset not implemented')

    if load_model:

        if not os.path.exists(f"{path}/data.pt"):
            print(f"Invalid path is given ({path})")
            exit(1)

        data = load(f"{path}/data.pt")
        model = data["model"]
        print('Load model successfully!')

    else:

        if model_name == 'resnet':
            if pretrained:
                model = resnet152(model_dir=pretrained_path,
                                  weights=ResNet152_Weights.DEFAULT, num_classes=1000)
                # Freeze the parameters of the model
                model = freeze_layers(model)
                # Except the last layer which is trainable
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
                model.fc.requires_grad = True
            else:
                model = resnet152(num_classes=num_classes)

            if dataname == 'mnist':
                model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(
                    7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        elif model_name == 'vgg':
            if pretrained:
                model = vgg19_bn(
                    weights=VGG19_BN_Weights.DEFAULT, num_classes=1000, model_dir=pretrained_path)
                # Freeze the parameters of the model
                model = freeze_layers(model)
                # Except the last layer which is trainable
                num_ftrs = model._modules['classifier'][-1].in_features
                model._modules['classifier'][-1] = nn.Linear(
                    num_ftrs, num_classes)

                model.classifier.requires_grad = True
            else:
                model = vgg19_bn(num_classes=num_classes)

            if dataname == 'mnist':
                model.features[0] = nn.Conv2d(
                    n_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        elif model_name == 'alexnet':

            if pretrained:
                model = alexnet(pretrained)
                # Updating the third and the last classifier that is the output layer of the network.
                # Make sure to have 10 output nodes if we are going to get 10 class labels through our model.
                model = freeze_layers(model)
                model.classifier[6] = nn.Linear(4096, num_classes)

                for num, (name, param) in enumerate(model.named_parameters()):
                    if num > 7:
                        param.requires_grad = True
            else:
                model = alexnet(num_classes=num_classes)

        elif model_name == 'googlenet':
            if pretrained:
                # model = googlenet(
                #    init_weights=GoogLeNet_Weights.DEFAULT, num_classes=1000)
                # The argument transform_input=False is needed for mnist
                # dataset that has only 1 channel because otherwise a function
                # is called that assumes three channels.
                model = torch.hub.load("pytorch/vision:v0.6.0", "googlenet",
                                       weights=GoogLeNet_Weights.DEFAULT, transform_input=False, model_dir=pretrained_path)
                # Freeze the parameters of the model
                model = freeze_layers(model)
                # Except the last layer which is trainable
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
                model.fc.requires_grad = True
            else:
                model = googlenet(num_classes=num_classes)

            if dataname == 'mnist':
                model.conv1.conv = nn.Conv2d(
                    n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            raise NotImplementedError(
                f'{model_name} model not implemented')

    return model


def loss_picker(loss):
    '''
    Select the loss function

    Parameters:
        loss (str): name of the loss function

    Returns:
        loss_function (torch.nn.Module): loss function
    '''
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("Automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion


def optimizer_picker(optimization, param, lr):
    '''
    Select the optimizer

    Parameters:
        optimization (str): name of the optimization method
        param (list): model's parameters to optimize
        lr (float): learning rate

    Returns:
        optimizer (torch.optim.Optimizer): optimizer

    '''
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=0.9)
    else:
        print("Automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer


def freeze_layers(model):
    '''
    Helper function to freeze all layers of the model

    Parameters:
        model (torch.nn.Module): model to freeze

    Returns:
        model (torch.nn.Module): model with all layers frozen
    '''
    # Set parames in BatchNorm2d to be trainable
    if VGG is type(model):
        for param, layer_class in zip(model.features.parameters(), model.features):
            if type(layer_class) is nn.BatchNorm2d:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif type(model) is ResNet or type(model) is GoogLeNet:
        for name, param in model.named_parameters():
            if 'bn' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif type(model) is AlexNet:
        for param in model.parameters():
            param.requires_grad = False

    return model
