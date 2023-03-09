from attacks.Attack import Attack
from copy import deepcopy
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv


class BadNets(Attack):
    """
    BadNets Attack
    """

    epsilon = None
    trigger_size = None
    pos = 'middle'
    color = 'white'

    def __init__(self, args, trainer) -> None:
        """
        Constructor

        Parameters
        ----------
        args : argparse.Namespace
            Arguments
        trainer : Trainer
            Trainer

        Returns
        -------
        None

        """

        super().__init__(trainer, args.target_label)
        self.epsilon = args.epsilon
        self.pos = args.pos
        self.color = args.color

        width = trainer.dataset.trainset.data.shape[2]
        size = int(width * args.trigger_size)
        self.trigger_size = size if size > 0 else 1

    def save_results(self, path=None) -> None:

        if path is None:
            path = self.trainer.save_path

        path_csv = self.get_path(path)

        # Write the results to the csv file
        header = ['id', 'dataset', 'model', 'epsilon', 'trigger_size', 'target_label',
                  'pos', 'color', 'seed', 'train_acc', 'train_loss', 'clean_acc',
                  'bk_acc', 'clean_loss', 'bk_loss']

        with open(path_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        with open(path_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.id, self.trainer.dataset.name, self.trainer.model.name, self.epsilon,
                             self.trigger_size, self.target_label, self.pos, self.color,
                             self.trainer.seed, self.trainer.train_acc[-1],
                             self.trainer.train_loss[-1], self.trainer.test_acc[-1],
                             self.trainer.bk_acc[-1], self.trainer.test_loss[-1], self.trainer.bk_loss[-1]])

    def execute_attack(self) -> None:
        """
        Get attack
        """
        # Create a copy of the orginal training set and test set
        poisoned_trainset = deepcopy(self.trainer.dataset.trainset)
        poisoned_testset = deepcopy(self.trainer.dataset.testset)

        # Get a random subset of the training set
        perm = torch.randperm(len(poisoned_trainset))
        idx = perm[:int(len(poisoned_trainset) * self.epsilon)]
        poisoned_subset_trainset = poisoned_trainset.data[idx]

        poisoned_trainset.data[idx] = self.create_trigger(
            poisoned_trainset.data[idx])

        poisoned_testset.data = self.create_trigger(poisoned_testset.data)

        # Change the label to the target label
        poisoned_trainset.targets = torch.as_tensor(poisoned_trainset.targets)
        poisoned_trainset.targets[idx] = self.target_label
        # Also for the test set
        poisoned_testset.targets = torch.as_tensor(poisoned_testset.targets)
        poisoned_testset.targets[:] = self.target_label

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
            f'Poisoned training set clean: {len(self.trainer.poisoned_dataset.trainset) - len(poisoned_subset_trainset)} Backdoor: {len(poisoned_subset_trainset)}')

        # Save a image as example
        plt.imsave('trigger.png', poisoned_testset.data[0])

        self.backdoor_train()

    def backdoor_train(self) -> None:
        """
        Train the model with the poisoned training set
        """
        self.trainer.train(clean=False)

    def create_trigger(self, data) -> np.ndarray:
        """
        Create trigger
        """

        if self.color == 'white':
            # Case with 1 channel
            if len(data.shape) == 3:
                value = 255
            # Case with 3 channels
            else:
                value = [[[255]], [[255]], [[255]]]

        elif self.color == 'black':
            # Case with 1 channel
            if len(data.shape) == 3:
                value = 0
            # Case with 3 channels
            else:
                value = [[[0]], [[0]], [[0]]]

        elif self.color == 'green':
            if len(data.shape) == 3:
                value = 0
            else:
                value = [[[102]], [[179]], [[92]]]

        else:
            raise ValueError('Color not supported')

        width = data.shape[1]
        height = data.shape[2]
        size_width = self.trigger_size
        size_height = self.trigger_size

        if self.pos == 'top-left':
            x_begin = 0
            x_end = size_width
            y_begin = 0
            y_end = size_height

        elif self.pos == 'top-right':
            x_begin = int(width - size_width)
            x_end = width
            y_begin = 0
            y_end = size_height

        elif self.pos == 'bottom-left':
            x_begin = 0
            x_end = size_width
            y_begin = int(height - size_height)
            y_end = height
        elif self.pos == 'bottom-right':
            x_begin = int(width - size_width)
            x_end = width
            y_begin = int(height - size_height)
            y_end = height

        elif self.pos == 'middle':
            x_begin = int((width - size_width) / 2)
            x_end = int((width + size_width) / 2)
            y_begin = int((height - size_height) / 2)
            y_end = int((height + size_height) / 2)

        else:
            raise ValueError('Position not supported')

        if len(data.shape) == 3:
            # MNIST case
            data[:, x_begin:x_end, y_begin:y_end] = value
        else:
            data[:, x_begin:x_end, y_begin:y_end, :] = value

        return data
