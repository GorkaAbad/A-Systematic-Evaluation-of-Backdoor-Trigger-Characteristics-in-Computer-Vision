from abc import ABC
from torchvision.models import resnet152, alexnet, googlenet, vgg19_bn,  VGG, ResNet, GoogLeNet, AlexNet
import torch.nn as nn
import torch


class Model(ABC):
    """
    Model
    """
    model = None
    name = None
    pretrained = None

    def __init__(self, args, n_classes) -> None:
        """
        Constructor
        """
        self.pretrained = args.pretrained
        self.pretrained_path = args.pretrained_path
        self.name = args.model
        if args.load_model:
            self.load_model(args.load_model)
            if self.pretrained:
                self.model = self.freeze(self.model)
        else:
            self.model = self.get_model(args.model, n_classes=n_classes)

    def load_model(self, path) -> torch.nn.Module:
        """
        Load model
        """
        self.model = torch.load(path)
        return self.model

    def save_model(self, path) -> None:
        """
        Save model
        """
        torch.save(self.model, path)

    def get_model(self, model, n_classes) -> torch.nn.Module:
        """
        Get model
        """

        md = None
        if model == 'resnet':
            if self.pretrained:
                md = resnet152(pretrained=self.pretrained, num_classes=1000)
                # Freeze the parameters of the model
                md = self.freeze(md)
                # Except the last layer which is trainable
                num_ftrs = md.fc.in_features
                md.fc = nn.Linear(num_ftrs, n_classes)
                md.fc.requires_grad = True
            else:
                md = resnet152(num_classes=n_classes)

        elif model == 'vgg':
            if self.pretrained:
                md = vgg19_bn(pretrained=self.pretrained, num_classes=1000)
                # Freeze the parameters of the model
                md = self.freeze(md)
                # Except the last layer which is trainable
                num_ftrs = md.classifier[-1].in_features
                md.classifier[-1] = nn.Linear(num_ftrs, n_classes)
                md.classifier.requires_grad = True
            else:
                md = vgg19_bn(num_classes=n_classes)
        elif model == 'googlenet':
            if self.pretrained:
                md = googlenet(pretrained=self.pretrained, num_classes=1000)
                # Freeze the parameters of the model
                md = self.freeze(md)
                # Except the last layer which is trainable
                num_ftrs = md.fc.in_features
                md.fc = nn.Linear(num_ftrs, n_classes)
                md.fc.requires_grad = True
            else:
                md = googlenet(num_classes=n_classes)

        elif model == 'alexnet':
            if self.pretrained:
                md = alexnet(pretrained=self.pretrained)
                # Freeze the parameters of the model
                md = self.freeze(md)
                md.classifier[6] = nn.Linear(4096, n_classes)
                for num, (name, param) in enumerate(md.named_parameters()):
                    if num > 7:
                        param.requires_grad = True
            else:
                md = alexnet(pretrained=self.pretrained,
                             num_classes=n_classes)
        else:
            raise ValueError('Invalid model')

        return md

    def freeze(self, model) -> torch.nn.Module:
        """
        Freeze model's convolutional layers
        """
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
