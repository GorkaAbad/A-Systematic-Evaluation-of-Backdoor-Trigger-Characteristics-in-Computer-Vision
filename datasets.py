import torchvision
from torchvision import transforms
from data_prep import *


def get_dataset(dataname, args):
    '''
    For a given dataname, return the train and testset

    Parameters:
        dataname (str): name of the dataset
        args (argparse.Namespace): arguments

    Returns:
        trainset (torch.utils.data.Dataset): train dataset
        testset (torch.utils.data.Dataset): test dataset
    '''

    if dataname == 'mnist':
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

    elif dataname == 'cifar10':
        # InterpolationMode.BILINEAR
        # https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html#torchvision.transforms.Resize
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

    elif dataname == 'tinyimagenet':
        '''
        This is a special case for loading the dataset.
        The dataset has to be downloaded manually from http://cs231n.stanford.edu/tiny-imagenet-200.zip
        $ wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
        $ unzip tiny-imagenet-200.zip

        Then, the function create_val_img_folder() splits the images into train and val folders.
        The prepare_imagenet() function loads the images paths into the ImageDataset class.
        However, this is not suitable for our use case. We need the actual (loaded) images.
        For that we create an intermediary class TinyImageNet. Which loads the images and convert grayscale ones to color.
        It should not take too long, however the RAM usage is (a bit) high.
        The TinyImagenet class also prepares some attributes to fit the trigger injection pipeline.
        We are now fine to go!

        Some code was taken from: https://github.com/DennisHanyuanXu/Tiny-ImageNet
        '''
        create_val_img_folder(args)
        trainset, testset = prepare_imagenet(args)

    else:
        raise NotImplementedError(
            f'{dataname} dataset not implemented')

    return trainset, testset
