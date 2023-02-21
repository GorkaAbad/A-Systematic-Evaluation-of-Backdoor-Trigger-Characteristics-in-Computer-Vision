from datasets.Dataset import Dataset
from typing import Tuple
import torchvision.datasets
from torchvision import transforms


class CIFAR10(Dataset):
    def __init__(self, args) -> None:
        super().__init__(args)

    def get_dataset(self, args) -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.247, 0.243, 0.261]),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=args.datadir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root=args.datadir, train=False, download=True, transform=transform)

        return trainset, testset

    def get_n_classes(self) -> int:
        """
        Get number of classes
        """
        return 10
