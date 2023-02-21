from datasets.Dataset import Dataset
from typing import Tuple
import torchvision.datasets
from torchvision import transforms


class MNIST(Dataset):
    def __init__(self, args) -> None:
        super().__init__(args)

    def get_dataset(self, args) -> Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        trainset = torchvision.datasets.MNIST(
            root=args.datadir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root=args.datadir, train=False, download=True, transform=transform)
        self.trainset = trainset
        self.testset = testset
        return trainset, testset

    def get_n_classes(self) -> int:
        """
        Get number of classes
        """
        return 10
