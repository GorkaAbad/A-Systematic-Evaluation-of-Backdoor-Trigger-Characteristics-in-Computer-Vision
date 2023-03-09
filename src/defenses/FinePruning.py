from defenses.Defense import Defense
import torch
from copy import deepcopy
import csv
import datetime


class FinePruning(Defense):
    """
    Fine Pruning
    """

    # Attributes
    # ----------
    pruning_rate = None
    pruned_trainer = None
    pruned_acc = None
    pruned_bk_acc = None

    # Methods
    # -------

    def __init__(self, args, trainer):
        """
        Constructor

        Parameters
        ----------
        trainer : Trainer

        pruning_rate : float

        Returns
        -------
        None
        """
        super().__init__(trainer)
        self.pruning_rate = args.pruning_rate

    def execute_defense(self):
        """
        The fine-pruning defense deactivates a fraction of the neurons in the
        network (from the last convolutional layer). The fraction is determined by the pruning rate. The neurons
        are selected based on their activity during training. The neurons with
        the highest activity are deactivated. For evaluating that, we do a forward pass
        through the network and save the activations of the neurons. Then we
        deactivate the neurons with the highest activations.

        """

        self.pruned_trainer = deepcopy(self.trainer)

        model = self.pruned_trainer.model.model

        # Get the number of neurons in the last convolutional layer
        if self.trainer.model.name == 'resnet':
            layer_to_prune = model.layer4[2].conv3
        elif self.trainer.model.name == 'vgg':
            layer_to_prune = model.features[49]
        elif self.trainer.model.name == 'googlenet':
            layer_to_prune = model.inception5b.branch4[1].conv
        elif self.trainer.model.name == 'alexnet':
            layer_to_prune = model.features[10]
        else:
            raise ValueError('Model not supported')

        num_neurons = layer_to_prune.weight.shape[0]
        neurons_to_prune = int(num_neurons * self.pruning_rate)
        print(f'Number of neurons {num_neurons} in last convolutional layer')
        print(
            f'Number of neurons to prune {neurons_to_prune} ({self.pruning_rate * 100}%)')

        # Do a forward pass through the network and save the activations of the neurons
        # Define a forward hook to save the activations
        activations = []

        def hook(module, input, output):
            activations.append(output)

        # Register the hook
        handle = layer_to_prune.register_forward_hook(hook)

        # Do a forward pass through the network
        self.forward_pass(model)

        # Remove the hook
        handle.remove()

        activations = torch.cat(activations, dim=0)

        activations = torch.mean(activations, dim=(0, 2, 3))

        # Sort the activations of the neurons
        indices = torch.argsort(activations, descending=True)

        # Get the indices of the neurons with the highest activations
        indices = indices[:neurons_to_prune]

        # Deactivate the neurons with the highest activations
        for i in range(neurons_to_prune):
            layer_to_prune.weight[indices[i]].data = torch.zeros(
                layer_to_prune.weight[indices[i]].shape)
            try:
                layer_to_prune.bias[indices[i]].data = torch.zeros(
                    layer_to_prune.bias[indices[i]].shape)
            except:
                # Some layers do not have a bias
                pass

        # Retrain the model for 10% of the training epochs
        self.pruned_trainer.model.model = model

        # Evaluate the model without retraining
        print('Evaluating the model after pruning without retraining')
        test_acc, test_loss = self.pruned_trainer.evaluate()
        self.pruned_acc = test_acc

        if self.pruned_trainer.poisoned_dataset is not None:
            test_bk_acc, test_bk_loss = self.pruned_trainer.evaluate(
                clean=False)
            self.pruned_bk_acc = test_bk_acc
        else:
            print('No poisoned dataset available for evaluation. Omitting...')

        # At least 1 epoch
        self.pruned_trainer.epochs = max(
            1, int(self.pruned_trainer.epochs * 0.1))

        print(f'Retraining the model for {self.pruned_trainer.epochs} epochs')
        self.pruned_trainer.train()

        print('Evaluating the model after fine-pruning')
        self.pruned_trainer.evaluate()

        if self.pruned_trainer.poisoned_dataset is not None:
            test_bk_acc, test_bk_loss = self.pruned_trainer.evaluate(
                clean=False)
            self.pruned_bk_acc = test_bk_acc
        else:
            print('No poisoned dataset available for evaluation. Omitting...')

    def forward_pass(self, model):
        """
        Do a forward pass through the network
        """
        model = model.to(self.trainer.device)
        model.eval()
        with torch.no_grad():
            for data, target in self.pruned_trainer.trainloader:
                data, target = data.to(self.trainer.device), target.to(
                    self.trainer.device)
                output = model(data)
                break

    def save_results(self, path=None) -> None:

        if path is None:
            path = self.trainer.save_path

        path_csv = self.get_path(path)

        # Write the results to the csv file
        header = ['id', 'attack_id', 'dataset', 'model', 'pruning_rate', 'seed', 'train_acc', 'train_loss', 'clean_acc',
                  'bk_acc', 'clean_loss', 'bk_loss',
                  'pruned_clean_acc', 'pruned_bk_acc', 'fine-pruned_clean_acc', 'fine-pruned_bk_acc']

        with open(path_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        bk_acc = self.trainer.bk_acc[-1] if self.trainer.poisoned_dataset is not None else None
        bk_loss = self.trainer.bk_loss[-1] if self.trainer.poisoned_dataset is not None else None

        fine_pruned_bk_acc = self.pruned_trainer.bk_acc[-1] if self.trainer.poisoned_dataset is not None else None

        with open(path_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.id, self.attack_id, self.trainer.dataset.name, self.trainer.model.name, self.pruning_rate,
                             self.trainer.seed, self.trainer.train_acc[-1],
                             self.trainer.train_loss[-1], self.trainer.test_acc[-1],
                             bk_acc, self.trainer.test_loss[-1], bk_loss,
                             self.pruned_acc, self.pruned_bk_acc,
                             self.pruned_trainer.test_acc[-1],
                             fine_pruned_bk_acc])
