import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import get_dataset


class PoisonedDataset(Dataset):
    '''
    Poisoned dataset class

    Attributes:
        dataset (torch.utils.data.Dataset): dataset
        trigger_label (int): label of the target/objective class. The class to be changed to.
        mode (str): 'train' or 'test'
        epsilon (float): rate of poisoned data
        pos (str): position of the trigger. 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'
        shape (str): shape of the trigger. 'square' or 'random'
        color (str): color of the trigger. 'white' or 'random'
        trigger_size (float): size of the trigger as the percentage of the image size.
        device (torch.device): device
        dataname (str): name of the dataset

    Methods:
        __getitem__: returns the poisoned data and the corresponding label
        __len__: returns the length of the dataset
        __shape_info__: returns the shape of the dataset
        reshape: reshapes the dataset
        norm: normalizes the dataset
        add_trigger: adds the trigger to the dataset
        reshape_back: reshapes the dataset back to the original shape

    '''

    def __init__(self, dataset, trigger_label=0, mode='train', epsilon=0.1, pos='top-left', shape='square',
                 color='white',
                 trigger_size=0.1, device=torch.device('cuda'), dataname='minst'):

        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.transform = dataset.transform

        # TODO: Change the attributes of the imagenet to fit the same as MNIST
        self.data, self.targets = self.add_trigger(self.reshape(
            dataset.data, dataname), dataset.targets, trigger_label, epsilon, mode, pos, shape, color, trigger_size)
        self.channels, self.width, self.height = self.__shape_info__()
        self.data = self.reshape_back(self.data, dataname=self.dataname)

    def __getitem__(self, item):

        img = self.data[item]
        label_idx = int(self.targets[item])

        if self.transform:
            img = self.transform(img.astype(np.uint8))

        label = np.zeros(self.class_num)
        label[label_idx] = 1  # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, data, dataname='mnist'):

        if dataname == 'mnist':
            new_data = data.reshape(len(data), 1, 28, 28)
        elif dataname == 'cifar10':
            # new_data = data.reshape(len(data), 3, 32, 32)
            new_data = np.transpose(data, (0, 3, 1, 2))
        elif dataname == 'tinyimagenet':
            # new_data = np.transpose(data, (0, 3, 1, 2))
            new_data = data

        return np.array(new_data)

    def reshape_back(self, data, dataname='mnist'):

        if dataname == 'mnist':
            new_data = data.reshape(len(data), 28, 28)
        elif dataname == 'cifar10':
            new_data = np.transpose(data, (0, 2, 3, 1))
        elif dataname == 'tinyimagenet':
            new_data = np.transpose(data, (0, 2, 3, 1))

        return new_data

    def norm(self, data):
        offset = np.mean(data, 0)
        scale = np.std(data, 0).clip(min=1)
        return (data - offset) / scale

    def add_trigger(self, data, targets, trigger_label, epsilon, mode, pos, shape, color, trigger_size):
        '''
        Adds the trigger to the dataset

        Parameters:
            data (torch.tensor): dataset
            targets (torch.tensor): targets
            trigger_label (int): label of the target/objective class. The class to be changed to.
            epsilon (float): rate of poisoned data
            mode (str): 'train' or 'test'
            pos (str): position of the trigger. 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'
            shape (str): shape of the trigger. 'square' or 'random'
            color (str): color of the trigger. 'black', 'white', or 'random'
            trigger_size (float): size of the trigger as the percentage of the image size.

        Returns:
            poisoned_data (torch.tensor): poisoned dataset

        '''

        print("## generate " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)

        # Fixes some bugs
        if not torch.is_tensor(new_targets):
            new_targets = torch.Tensor(new_targets)

        # Choose a random subset of samples to be poisoned
        perm = np.random.permutation(len(new_data))[
            0: int(len(new_data) * epsilon)]
        _, width, height = new_data.shape[1:]

        # Swap every samples to the target class
        new_targets[perm] = trigger_label

        patch = None
        # Add the trigger to the dataset
        if color == 'white':
            # MNIST case 1 dimension
            if new_data[0].shape[0] == 1:
                value = 255
            else:
                value = [[[255]], [[255]], [[255]]]
        elif color == 'black':
            # MNIST case 1 dimension
            if new_data[0].shape[0] == 1:
                value = 0
            else:
                value = [[[0]], [[0]], [[0]]]
        elif color == 'green':
            # MNIST case 1 dimension
            if new_data[0].shape[0] == 1:
                value = 0
            else:
                value = [[[102]], [[179]], [[92]]]
        elif color == 'red':
            # MNIST case 1 dimension
            if new_data[0].shape[0] == 1:
                raise ValueError('Red color is not available for MNIST')
            else:
                value = [[[255]], [[0]], [[0]]]
        elif color == 'random':
            # Put this here to have the same trigger for training and testing
            # dataset, as they are poisoned separatedly.
            np.random.seed(42)
            # MNIST case 1 dimension
            if new_data[0].shape[0] == 1:
                value = np.random.randint(0, 256)
            else:
                value = [[[np.random.randint(0, 256)]], [[np.random.randint(0, 256)]], [
                    [np.random.randint(0, 256)]]]

        size_width = int(trigger_size * width)
        size_height = int(trigger_size * height)

        if size_height == 0:
            size_height = 1
        if size_width == 0:
            size_width = 1

        if patch is not None:
            # Resize the patch
            patch.thumbnail((size_width, size_height))

        if shape == 'square':
            if pos == 'top-left':
                x_begin = 0
                x_end = size_width
                y_begin = 0
                y_end = size_height

            elif pos == 'top-right':
                x_begin = int(width - size_width)
                x_end = width
                y_begin = 0
                y_end = size_height

            elif pos == 'bottom-left':
                x_begin = 0
                x_end = size_width
                y_begin = int(height - size_height)
                y_end = height
            elif pos == 'bottom-right':
                x_begin = int(width - size_width)
                x_end = width
                y_begin = int(height - size_height)
                y_end = height

            elif pos == 'middle':
                x_begin = int((width - size_width) / 2)
                x_end = int((width + size_width) / 2)
                y_begin = int((height - size_height) / 2)
                y_end = int((height + size_height) / 2)

            elif pos == 'random':
                # Note that every sample gets the same (random) trigger position
                # We can easily implement random trigger position for each sample by using the following code
                ''' TODO:
                 new_data[perm, :, np.random.randint(
                    0, height, size=len(perm)), np.random.randint(0, width, size=(perm))] = value
                '''
                # Put this here for reproducible experiments
                np.random.seed(42)
                x_begin = np.random.randint(0, width)
                x_end = x_begin + size_width
                y_begin = np.random.randint(0, height)
                y_end = y_begin + size_height

        elif shape == 'random':
            # Something along this lines could be useful: https://www.blog.pythonlibrary.org/2021/02/23/drawing-shapes-on-images-with-python-and-pillow/
            raise NotImplementedError('Random shape not implemented yet')

        # Add the trigger to the dataset
        if patch is not None:
            for i in range(y_begin, y_end):
                for j in range(x_begin, x_end):
                    new_data[perm, :, i, j] = patch.getpixel((j, i))
        else:
            new_data[perm, :, y_begin:y_end, x_begin:x_end] = value

        print(
            f'Injecting Over: Bad Imgs: {len(perm)}. Clean Imgs: {len(new_data) - len(perm)}. Epsilon: {epsilon}')

        return new_data, new_targets


def create_backdoor_data_loader(dataname, trigger_label, epsilon, pos, shape, color, trigger_size,
                                batch_size_train, batch_size_test, device, args):
    '''
    Creates the data loader for the backdoor training dataset, a clean test dataset, and a fully poisoned test dataset.

    Parameters:
        dataname (str): name of the dataset
        trigger_label (int): label of the target/objective class. The class to be changed to.
        epsilon (float): rate of poisoned data
        pos (str): position of the trigger. 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'
        shape (str): shape of the trigger. 'square' or 'random'
        color (str): color of the trigger. 'red', 'green', 'blue', 'random'
        trigger_size (float): size of the trigger as the percentage of the image size.
        batch_size_train (int): batch size for training
        batch_size_test (int): batch size for testing
        device (torch.device): device to use
        args (argparse.Namespace): arguments

    Returns:
        train_data_loader (torch.utils.data.DataLoader): training data loader
        test_data_ori_loader (torch.utils.data.DataLoader): clean test data loader
        test_data_tri_loader (torch.utils.data.DataLoader): poisoned test data loader

    '''

    # Get the dataset
    train_data, test_data = get_dataset(dataname, args)

    train_data = PoisonedDataset(
        train_data, trigger_label, mode='train', epsilon=epsilon, device=device,
        pos=pos, shape=shape, color=color, trigger_size=trigger_size, dataname=dataname)

    test_data_ori = PoisonedDataset(test_data, trigger_label, mode='test', epsilon=0,
                                    device=device, pos=pos, color=color, shape=shape,
                                    trigger_size=trigger_size, dataname=dataname)

    test_data_tri = PoisonedDataset(test_data, trigger_label, mode='test', epsilon=1,
                                    device=device, pos=pos, shape=shape, color=color,
                                    trigger_size=trigger_size, dataname=dataname)

    train_data_loader = DataLoader(
        dataset=train_data, batch_size=batch_size_train, shuffle=True)
    test_data_ori_loader = DataLoader(
        dataset=test_data_ori, batch_size=batch_size_test, shuffle=True)
    test_data_tri_loader = DataLoader(
        dataset=test_data_tri, batch_size=batch_size_test, shuffle=True)

    return train_data_loader, test_data_ori_loader, test_data_tri_loader
