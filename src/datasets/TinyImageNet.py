#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image
from datasets.Dataset import Dataset
from datasets.TinyImagenetBase import TinyImagenetBase


class TinyImageNet(Dataset):
    def __init__(self, args):
        super().__init__(args)

    def create(self, dataset, mode='train', transform=None):

        print('Loading images into memory...')
        targets = [i[1] for i in dataset.imgs]
        dataset.class_to_idx = {i: c for c, i in dataset.class_to_idx.items()}

        data = np.array([np.array(read_image(image[0]).permute(1, 2, 0))
                         for image in dataset.imgs])

        base = TinyImagenetBase(train=mode, data=data,
                                targets=targets, transform=transform)

        return base

    def get_dataset(self, args):
        data_dir = '{}/tiny-224'.format(args.datadir)
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')

        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_trans = [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            norm]

        test_trans = [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            norm]

        image_folder_train = datasets.ImageFolder(
            train_dir, transform=transforms.Compose(train_trans)
        )
        image_folder_test = datasets.ImageFolder(
            test_dir, transform=transforms.Compose(test_trans)
        )

        train_base = self.create(
            image_folder_train, 'train', transforms.Compose(train_trans))
        test_base = self.create(image_folder_test, 'test',
                                transforms.Compose(test_trans))

        self.trainset = train_base
        self.testset = test_base

        return train_base, test_base

    def get_n_classes(self):
        return 200
