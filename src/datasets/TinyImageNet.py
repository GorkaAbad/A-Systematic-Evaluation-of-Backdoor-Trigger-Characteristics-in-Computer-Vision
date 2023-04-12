#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image
import torch
from datasets.Dataset import Dataset


class TinyImageNet(Dataset):
    def __init__(self, args):
        super().__init__(args)

    def create(self, dataset):

        dataset.targets = [i[1] for i in dataset.imgs]
        dataset.class_to_idx = {i: c for c, i in dataset.class_to_idx.items()}

        # Load every image in the dataset into memory
        print('Loading images into memmory...')
        dataset.data = np.array([np.array(self.read_img(image[0]))
                                 for image in dataset.imgs])
        print('Done')

        return dataset

    def read_img(self, path):
        img = read_image(path)
        img = self.convert_to_rgb(img)

        # Reshape correctly the image
        img = img.detach().cpu().numpy()
        img = img.transpose(1, 2, 0)
        img = torch.from_numpy(img)
        #img = img.reshape(img.shape[1], img.shape[2], img.shape[0])
        return img

    def convert_to_rgb(self, img):
        # Check if the image is grayscale
        if (img.shape[0] == 1):
            img = torch.squeeze(img)
            img = img.repeat(3, 1, 1)
        return img

    def get_dataset(self, args):
        self.create_val_img_folder(args)
        train_data, val_data = self.prepare_imagenet(args)
        print("doing tiny imagenet")
        self.trainset = train_data
        self.testset = val_data
        return train_data, val_data

    def get_n_classes(self):
        return 200

    def prepare_imagenet(self, args):
        dataset_dir = '{}/tiny-imagenet-200'.format(args.datadir)
        train_dir = os.path.join(dataset_dir, 'train')
        val_dir = os.path.join(dataset_dir, 'val', 'images')

        # Pre-calculated mean & std on imagenet:
        # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # For other datasets, we could just simply use 0.5:
        # norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        print('Preparing dataset ...')
        # Normalization
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        t = [transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.RandomResizedCrop(224)]

        # Normal transformation
        train_trans = [transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.RandomResizedCrop(224)]
        #train_trans = t

        val_trans = [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.CenterCrop(224),
            norm]
        #val_trans = t

        train_data = datasets.ImageFolder(train_dir,
                                          transform=transforms.Compose(train_trans + [norm]))

        val_data = datasets.ImageFolder(val_dir,
                                        transform=transforms.Compose(val_trans))

        # Here images are not load into memory. Just the paths, however, we need to load every imgage in order to inject the trigger.
        train_data = self.create(train_data)
        val_data = self.create(val_data)

        return train_data, val_data

    def create_val_img_folder(self, args):
        '''
        This method is responsible for separating validation images into separate sub folders
        '''
        dataset_dir = '{}/tiny-imagenet-200'.format(args.datadir)
        val_dir = os.path.join(dataset_dir, 'val')
        img_dir = os.path.join(val_dir, 'images')

        fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
        data = fp.readlines()
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        fp.close()

        # Create folder if not present and move images into proper folders
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(img_dir, folder))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(img_dir, img)):
                os.rename(os.path.join(img_dir, img),
                          os.path.join(newpath, img))

    def get_class_name(self):
        class_to_name = dict()
        fp = open(os.path.join('tiny-imagenet-200', 'words.txt'), 'r')
        data = fp.readlines()
        for line in data:
            words = line.strip('\n').split('\t')
            class_to_name[words[0]] = words[1].split(',')[0]
        fp.close()
        return class_to_name
