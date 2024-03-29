from attacks.Attack import Attack
import random
import torch
import torchvision
import kornia.augmentation as A
from copy import deepcopy
import os
import torch.nn.functional as F
import numpy as np
import csv
from torchvision import transforms


class WaNet(Attack):
    """
    WaNet Attack
    """

    def __init__(self, args, trainer):
        super().__init__(trainer, args.target_label)
        # self.identity_grid =  args.identity_grid #
        self.args = args
        self.s = args.s
        # self.noise_grid = args.noise_grid #
        # self.input_height = args.input_height #
        self.grid_rescale = args.grid_rescale
        self.cross_ratio = args.cross_ratio
        self.device = args.device
        # self.input_width = args.input_width #
        self.random_crop = args.random_crop
        self.random_rotation = args.random_rotation
        self.ckpt_path = args.ckpt_path
        self.dataname = args.dataname
        self.k = args.k
        self.epsilon = args.epsilon

    def save_results(self, path=None) -> None:
        if path is None:
            path = self.trainer.save_path

        path_csv = self.get_path(path)

        # Write the results to the csv file
        if self.args.type == 'wanet':
            header = ['id', 'dataset', 'model', 'epsilon', 'target_label',
                      'seed', 'train_acc', 'train_loss', 'clean_acc',
                      'bk_acc', 'clean_loss', 'bk_loss']
        else:
            # TODO: This block may not needed
            header = ['id', 'dataset', 'model', 'epsilon', 'trigger_size', 'target_label',
                      'pos', 'color', 'seed', 'train_acc', 'train_loss', 'clean_acc',
                      'bk_acc', 'clean_loss', 'bk_loss']
        if not os.path.exists(path_csv):
            with open(path_csv, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        with open(path_csv, 'a') as f:
            writer = csv.writer(f)
            if self.args.type == 'wanet':
                writer.writerow([self.trainer.id, self.trainer.dataset.name,
                                 self.trainer.model.name, self.epsilon,
                                 self.target_label,
                                 self.trainer.seed, self.trainer.train_acc[-1],
                                 self.trainer.train_loss[-1], self.trainer.test_acc[-1],
                                 self.trainer.bk_acc[-1],
                                 self.trainer.test_loss[-1],
                                 self.trainer.bk_loss[-1]])
            else:
                # TODO: This block may not needed
                writer.writerow([self.id, self.trainer.dataset.name, self.trainer.model.name, self.epsilon,
                                self.trigger_size, self.target_label, self.pos, self.color,
                                self.trainer.seed, self.trainer.train_acc[-1],
                                self.trainer.train_loss[-1], self.trainer.test_acc[-1],
                                self.trainer.bk_acc[-1], self.trainer.test_loss[-1], self.trainer.bk_loss[-1]])

    def insert_channel(self, trainset, testset):
        tr = torch.unsqueeze(trainset.data, 3)
        te = torch.unsqueeze(testset.data, 3)
        pre_transform = transforms.Lambda(lambda x: x.repeat(1, 1, 1, 3))
        trainset.data = pre_transform(tr)
        testset.data = pre_transform(te)

    def execute_attack(self):
        """
        Get attack
        """
        # Create a copy of the orginal training set and test set
        if self.dataname == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(64),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            poisoned_trainset = torchvision.datasets.MNIST(
                root=self.args.datadir, train=True, download=True, transform=transform)
            poisoned_testset = torchvision.datasets.MNIST(
                root=self.args.datadir, train=False, download=True, transform=transform)
            self.insert_channel(poisoned_trainset, poisoned_testset)
        else:
            poisoned_trainset = deepcopy(self.trainer.dataset.trainset)
            poisoned_testset = deepcopy(self.trainer.dataset.testset)
        poisoned_trainset, poisoned_testset, num_bd = self.add_trigger(
            poisoned_trainset, poisoned_testset)
        # transform tensor to array (for cifar10)
        if self.dataname in ['cifar10', 'tinyimagenet']:
            poisoned_trainset.data = poisoned_trainset.data.cpu().detach().numpy()
            poisoned_trainset.data = poisoned_trainset.data.astype(np.uint8)

        # Create a new trainer with the poisoned training set
        self.trainer.poisoned_dataset = deepcopy(self.trainer.dataset)
        self.trainer.poisoned_dataset.trainset = poisoned_trainset
        self.trainer.poisoned_dataset.testset = poisoned_testset
        self.trainer.poisoned_trainloader, self.trainer.poisoned_testloader = self.trainer.get_dataloader(
            clean=False)

        print(
            f'Successfully created a poisoned dataset with {self.epsilon * 100}% of the training set')
        print(f'Original training set: {len(self.trainer.dataset.trainset)}')
        print(
            f'Poisoned training set clean: {len(self.trainer.poisoned_dataset.trainset) - num_bd} Backdoor: {num_bd}')

        self.backdoor_train()
        print('WaNet Attack')

    def backdoor_train(self) -> None:
        """
        Train the model with the poisoned training set
        """
        self.trainer.train(clean=False)

    def add_trigger(self, poisoned_trainset, poisoned_testset):
        # get value for input_height, input_width
        if self.dataname == 'cifar10':
            self.input_height = 32
            self.input_width = 32
        elif self.dataname == 'mnist':
            self.input_height = 28
            self.input_width = 28
        elif self.dataname == 'tinyimagenet':
            self.input_height = 224
            self.input_width = 224
        else:
            self.input_height = 64
            self.input_width = 64
        transforms = PostTensorTransform(self).to(self.device)
        num_test = len(poisoned_testset)

        # get value for identity_grid and noise_grid
        if os.path.exists(self.ckpt_path):
            state_dict = torch.load(self.ckpt_path)
            identity_grid = state_dict["identity_grid"]
            noise_grid = state_dict["noise_grid"]
        else:
            print("Pretrained model does not exist")
            ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (
                F.upsample(ins, size=self.input_height,
                           mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
                .to(self.device)
            )
            array1d = torch.linspace(-1, 1, steps=self.input_height)
            x, y = torch.meshgrid(array1d, array1d)
            identity_grid = torch.stack((y, x), 2)[None, ...].to(self.device)
        # Get a random subset of the training set
        perm = torch.randperm(len(poisoned_trainset))
        idx = perm[:int(len(poisoned_trainset) * self.epsilon)]
        num_bd = len(idx)
        num_cross = int(num_bd * self.cross_ratio)
        idx_cross = perm[num_bd:num_bd+num_cross]

        grid_temps = (identity_grid + self.s * noise_grid /
                      self.input_height) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(num_cross, self.input_height,
                         self.input_height, 2).to(self.device) * 2 - 1
        grid_temps2 = grid_temps.repeat(
            num_cross, 1, 1, 1) + ins / self.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        if self.dataname in ['cifar10', 'tinyimagenet']:
            t = torch.FloatTensor(poisoned_trainset.data[:num_bd])
        elif self.dataname == 'mnist':
            t = poisoned_trainset.data[:num_bd].to(torch.float32)
        t = torch.permute(t, (0, 3, 1, 2))
        inputs_bd = F.grid_sample(t, grid_temps.repeat(
            num_bd, 1, 1, 1), align_corners=True)
        if self.dataname in ['cifar10', 'tinyimagenet']:
            t = torch.FloatTensor(
                poisoned_trainset.data[num_bd: (num_bd + num_cross)])
        elif self.dataname == 'mnist':
            t = poisoned_trainset.data[num_bd: (
                num_bd + num_cross)].to(torch.float32)
        t = torch.permute(t, (0, 3, 1, 2))
        inputs_cross = F.grid_sample(t, grid_temps2, align_corners=True)

        if self.dataname == 'mnist':
            inputs_bd = inputs_bd.to(torch.uint8)
            inputs_cross = inputs_cross.to(torch.uint8)
        inputs_bd = torch.permute(inputs_bd, (0, 2, 3, 1))
        inputs_cross = torch.permute(inputs_cross, (0, 2, 3, 1))
        poisoned_trainset.data[idx] = inputs_bd
        poisoned_trainset.data[idx_cross] = inputs_cross
        if self.dataname in ['cifar10', 'tinyimagenet']:
            # poisoned_trainset.data = transforms(torch.FloatTensor(poisoned_trainset.data))
            poisoned_trainset.data = torch.FloatTensor(poisoned_trainset.data)
            poisoned_trainset.data = torch.permute(
                poisoned_trainset.data, (0, 3, 1, 2))
            poisoned_trainset.data = transforms(poisoned_trainset.data)
            poisoned_trainset.data = torch.permute(
                poisoned_trainset.data, (0, 2, 3, 1))
        elif self.dataname == 'mnist':
            poisoned_trainset.data = transforms(poisoned_trainset.data.to(
                torch.float32))  # cause out-of-memory error
            poisoned_trainset.data = poisoned_trainset.data.to(torch.uint8)

        # Change the label to the target label
        poisoned_trainset.targets = torch.as_tensor(poisoned_trainset.targets)
        poisoned_trainset.targets[idx] = self.target_label

        # Poison the test set
        if self.dataname in ['cifar10', 'tinyimagenet']:
            t = torch.FloatTensor(poisoned_testset.data[:])
        elif self.dataname == 'mnist':
            t = poisoned_testset.data.to(torch.float32)
        t = torch.permute(t, (0, 3, 1, 2))
        inputs_bd = F.grid_sample(t, grid_temps.repeat(
            num_test, 1, 1, 1), align_corners=True)
        inputs_bd = torch.permute(inputs_bd, (0, 2, 3, 1))
        if self.dataname == 'mnist':
            inputs_bd = inputs_bd.to(torch.uint8)
        poisoned_testset.data[:] = inputs_bd
        poisoned_testset.targets = torch.as_tensor(poisoned_testset.targets)
        poisoned_testset.targets[:] = self.target_label
        return poisoned_trainset, poisoned_testset, num_bd


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(
            A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataname in ["cifar10", 'tinyimagenet']:
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
