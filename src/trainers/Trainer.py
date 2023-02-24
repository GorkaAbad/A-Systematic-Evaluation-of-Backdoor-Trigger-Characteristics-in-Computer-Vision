from abc import ABC
from typing import Tuple
from datasets.MNIST import MNIST
from datasets.CIFAR10 import CIFAR10
from datasets.TinyImageNet import TinyImageNet
from models.Model import Model
import torch
from tqdm import tqdm
from datasets import Dataset
import os


class Trainer(ABC):
    """
    Trainer class
    """

    # Attributes
    # ----------
    model = None
    dataset = None
    trainloader = None
    testloader = None
    poisoned_dataset = None
    poisoned_trainloader = None
    poisoned_testloader = None
    optimizer = None
    lr = None
    momentum = None
    weight_decay = None
    loss = None
    amp = None
    device = None
    epochs = None
    batch_size = None
    scaler = None
    seed = None
    save_path = None

    # Results of the training
    # -----------------------
    train_loss = None
    train_acc = None
    test_loss = None
    test_acc = None
    bk_loss = None
    bk_acc = None

    def __init__(self, args) -> None:
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.dataset = self.get_dataset(args)
        self.model = self.get_model(
            args, n_classes=self.dataset.n_classes)
        self.batch_size = args.batch_size
        self.trainloader, self.testloader = self.get_dataloader()
        self.optimizer = self.get_optimizer(args)
        self.loss = self.get_loss(args)
        self.amp = args.amp
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None
        self.device = self.get_device()
        self.epochs = args.epochs
        self.seed = args.seed
        self.save_path = args.save_path

    def get_dataloader(self, clean=True) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Get dataloader
        """
        if clean:
            trainset = self.dataset.trainset
            testset = self.dataset.testset
        else:
            trainset = self.poisoned_dataset.trainset
            testset = self.poisoned_dataset.testset

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False)

        return trainloader, testloader

    def train(self, clean=True):
        """
        Train model
        """

        if clean:
            trainloader = self.trainloader
            dataset = self.dataset
        else:
            trainloader = self.poisoned_trainloader
            dataset = self.poisoned_dataset

        list_train_acc = []
        list_train_loss = []
        list_test_acc = []
        list_test_loss = []
        list_test_acc_bk = []
        list_test_loss_bk = []
        train_acc = 0
        train_loss = 0
        self.model.model.to(self.device)
        self.model.model.train()
        for epoch in range(self.epochs):
            for (data, target) in tqdm(trainloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model.model(data)
                if self.amp:
                    with torch.cuda.amp.autocast():
                        loss = self.loss(output, target)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.loss(output, target)
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_acc += pred.eq(target.view_as(pred)).sum().item()

            train_loss /= len(dataset.trainset)
            train_acc /= len(dataset.trainset)
            print('Train Epoch: {} \tLoss: {:.6f} \tAccuracy: {:.6f}'.format(
                epoch, train_loss, train_acc))

            # We always evaluate on clean test set
            test_acc, test_loss = self.evaluate(True)
            if not clean:
                bk_test_acc, bk_test_loss = self.evaluate(clean)
                list_test_acc_bk.append(bk_test_acc)
                list_test_loss_bk.append(bk_test_loss)

            list_train_acc.append(train_acc)
            list_train_loss.append(train_loss)
            list_test_acc.append(test_acc)
            list_test_loss.append(test_loss)

        self.train_loss = list_train_loss
        self.train_acc = list_train_acc
        self.test_loss = list_test_loss
        self.test_acc = list_test_acc
        self.bk_loss = list_test_loss_bk
        self.bk_acc = list_test_acc_bk

        return list_train_acc, list_train_loss, list_test_acc, list_test_loss, list_test_acc_bk, list_test_loss_bk

    def evaluate(self, clean=True) -> Tuple[float, float]:
        """
        Evaluate model
        """

        if clean:
            testloader = self.testloader
            dataset = self.dataset
        else:
            testloader = self.poisoned_testloader
            dataset = self.poisoned_dataset

        test_acc = 0
        test_loss = 0
        self.model.model.eval()
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model.model(data)
                test_loss += self.loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_acc += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(dataset.testset)
            test_acc /= len(dataset.testset)

            print('{} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                'Clean' if clean else 'Backdoor', test_loss, test_acc, len(
                    dataset.testset),
                100. * test_acc))

        return test_acc, test_loss

    def get_model(self, args, n_classes) -> Model:
        """
        Get model
        """
        return Model(args, n_classes=n_classes)

    def get_dataset(self, args) -> Dataset:
        if args.dataname == 'mnist':
            dataset = MNIST(args)
        elif args.dataname == 'cifar10':
            dataset = CIFAR10(args)
        elif args.dataname == 'tinyimagenet':
            dataset = TinyImageNet(args)
        else:
            raise Exception('Dataset not supported')
        return dataset

    def get_optimizer(self, args) -> torch.optim.Optimizer:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.model.parameters(), lr=self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.model.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            raise Exception('Optimizer not supported')
        return optimizer

    def get_loss(self, args) -> torch.nn.Module:
        if args.loss == 'cross':
            loss = torch.nn.CrossEntropyLoss()
        elif args.loss == 'mse':
            loss = torch.nn.MSELoss()
        else:
            raise Exception('Loss not supported')
        return loss

    def get_device(self) -> torch.device:
        """
        Get device
        """
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reset_optimizer(self) -> None:
        """
        Reset optimizer
        """
        if isinstance(self.optimizer, torch.optim.SGD):
            self.optimizer = torch.optim.SGD(self.model.model.parameters(), lr=self.lr,
                                             momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        elif isinstance(self.optimizer, torch.optim.Adam):
            self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=self.lr,
                                              weight_decay=self.weight_decay)
        else:
            raise Exception('Optimizer not supported')

    def save_trainer(self, path=None) -> None:
        """
        Save trainer
        """
        if path is None:
            path = self.save_path

        # Save the model based on unique name

        name = self.model.name + '_' + self.dataset.name + '_' + str(self.optimizer) + '_' + str(self.loss) + '_' + str(
            self.epochs) + '_' + str(self.lr) + '_' + str(self.momentum) + '_' + str(self.weight_decay) + '_' + str(
            self.batch_size) + '_' + str(self.amp) + '_' + str(self.seed)

        path = os.path.join(path, name, 'trainer.pt')
        torch.save(self, path)
