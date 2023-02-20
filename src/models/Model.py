from abc import ABC, abstractmethod
from torchvision.models import resnet152, alexnet, googlenet, vgg19_bn


class Model(ABC):
    """
    Model
    """
    model = None
    pretrained = None

    def __init__(self, model, pretrained=False):
        """
        Constructor
        """
        self.pretrained = pretrained
        self.model = self.get_model(model)

    def get_model(self, model) -> object:
        """
        Get model
        """

        md = None
        if model == 'resnet':
            md = resnet152(pretrained=self.pretrained)
        elif model == 'alexnet':
            md = alexnet(pretrained=self.pretrained)
        elif model == 'googlenet':
            md = googlenet(pretrained=self.pretrained)
        elif model == 'vgg':
            md = vgg19_bn(pretrained=self.pretrained)
        else:
            raise ValueError('Invalid model')
        return md

    def freeze(self):
        """
        Freeze model's convolutional layers
        """
        for param in self.model.parameters():
            param.requires_grad = False
