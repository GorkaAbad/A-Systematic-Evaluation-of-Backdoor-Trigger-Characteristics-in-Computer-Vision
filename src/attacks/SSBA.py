from attacks.Attack import Attack
import numpy as np
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
import csv
import os


class SSBA(Attack):
    """
    SSBA Attack
    """
    epsilon = None
    ssba_trainset_path = None
    ssba_testset_path = None

    def __init__(self, args, trainer):
        super().__init__(trainer, args.target_label)

        self.epsilon = args.epsilon

        self.ssba_trainset_path = os.path.join(
            'src/models/pretrained_models/ssba', f'{args.dataset}_ssba_train_b1.npy')
        self.ssba_testset_path = os.path.join(
            'src/models/pretrained_models/ssba', f'{args.dataset}_ssba_test_b1.npy')

    def execute_attack(self):
        """
        Get attack
        """

        # Create a copy of the orginal training set and test set
        poisoned_trainset = deepcopy(self.trainer.dataset.trainset)
        poisoned_testset = deepcopy(self.trainer.dataset.testset)

        # Load the already generated SSBA triggers
        # BE AWARE WE HAVE TO HANDLE RANDOMNESS HERE
        ssba_trainset = np.load(
            self.ssba_trainset_path, allow_pickle=True)
        ssba_testset = np.load(
            self.ssba_testset_path, allow_pickle=True)

        # Get a subset of the ssba_trainset which then would replace the original training set
        perm = np.random.permutation(len(ssba_trainset))
        idx = perm[:int(len(ssba_trainset) * self.epsilon)]

        # Replace the original training set with the ssba_trainset
        poisoned_trainset.data[idx] = ssba_trainset[idx]

        # The test set is replaced with the ssba_testset
        poisoned_testset.data = ssba_testset

        # Change the label to the target label
        poisoned_trainset.targets = torch.as_tensor(poisoned_trainset.targets)
        poisoned_trainset.targets[idx] = self.target_label

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
            f'Poisoned training set clean: {len(self.trainer.poisoned_dataset.trainset) - len(idx)} Backdoor: {len(idx)}')

        # Save a image as example
        plt.imsave('ssba.png', poisoned_testset.data[0])

        self.backdoor_train()

    def backdoor_train(self) -> None:
        """
        Train the model with the poisoned training set
        """
        self.trainer.train(clean=False)

    def save_results(self, path=None) -> None:

        if path is None:
            path = self.trainer.save_path

        path_csv = self.get_path(path)

        # Write the results to the csv file
        header = ['dataset', 'model', 'epsilon', 'target_label',
                  'seed', 'train_acc', 'train_loss', 'clean_acc',
                  'bk_acc', 'clean_loss', 'bk_loss']

        with open(path_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        with open(path_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.trainer.dataset.name, self.trainer.model.name,
                             self.epsilon, self.target_label,
                             self.trainer.seed, self.trainer.train_acc[-1],
                             self.trainer.train_loss[-1], self.trainer.test_acc[-1],
                             self.trainer.bk_acc[-1], self.trainer.test_loss[-1], self.trainer.bk_loss[-1]])
