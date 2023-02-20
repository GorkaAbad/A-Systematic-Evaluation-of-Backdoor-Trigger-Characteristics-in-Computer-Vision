from abc import ABC, abstractmethod
from torchvision.datasets import MNIST, CIFAR10
from datasets.TinyImageNet import TinyImagenet
from torchvision import transforms


class Dataset(ABC):
    """
    Dataset
    """
    dataset = None
    trainset = None
    testset = None

    def __init__(self, args):
        """
        Constructor
        """
        self.dataset = self.get_dataset(args)

    @abstractmethod
    def get_dataset(self, args):
        """
        Get dataset
        """
        if args.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(64),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
            trainset = MNIST(
                root=args.datadir, train=True, download=True, transform=transform)
            testset = MNIST(
                root=args.datadir, train=False, download=True, transform=transform)

        elif args.dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                    0.247, 0.243, 0.261]),
            ])

            trainset = CIFAR10(
                root=args.datadir, train=True, download=True, transform=transform)
            testset = CIFAR10(
                root=args.datadir, train=False, download=True, transform=transform)

        elif args.dataset == 'imagenet':
            trainset, testset = TinyImagenet(args)

        self.trainset = trainset
        self.testset = testset
        return trainset, testset
