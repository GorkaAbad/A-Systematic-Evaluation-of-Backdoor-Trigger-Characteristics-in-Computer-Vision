from abc import ABC, abstractmethod
import os
<<<<<<< HEAD
from datetime import datetime
import torch
=======
import datetime
>>>>>>> 8f531d6ebae9992cffaae58c91263da2615fa136


class Attack(ABC):
    """
    Attack
    """
    trainer = None
    name = None
    id = None
    target_label = 0

    def __init__(self, trainer, target_label) -> None:
        """
        Constructor

        Parameters
        ----------
        trainer : Trainer
            Trainer

        Returns
        -------
        None
        """
        super().__init__()
        self.name = self.__class__.__name__
        self.target_label = target_label
        self.trainer = trainer
        self.id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    @abstractmethod
    def execute_attack(self) -> None:
        """
        Execute attack
        """
        pass

    @abstractmethod
    def save_results(self, path=None) -> None:
        """
        Save attack results
        """
        pass

    def get_path(self, path) -> str:
        """
        Save attack results
        """

        if not os.path.exists(path):
            os.makedirs(path)

        # Create a folder per attack
        path = os.path.join(path, self.name)

        if not os.path.exists(path):
            os.makedirs(path)

        # Create a folder per seed
        path = os.path.join(path, str(self.trainer.seed))

        if not os.path.exists(path):
            os.makedirs(path)

        # Create the csv file
        path = os.path.join(path, 'results.csv')

        return path

    def save_attack(self, path=None) -> None:
        """
        Save attack object
        """

        if path is None:
            path = self.trainer.save_path

        if not os.path.exists(path):
            os.makedirs(path)

        # Create a folder per attack
        path = os.path.join(path, self.name)

        if not os.path.exists(path):
            os.makedirs(path)

        # Create a folder per seed
        path = os.path.join(path, str(self.trainer.seed))

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, self.id + '.pkl')
        torch.save(self, path)
