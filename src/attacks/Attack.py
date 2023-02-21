from abc import ABC, abstractmethod
import os


class Attack(ABC):
    """
    Attack
    """
    trainer = None
    name = None

    def __init__(self, trainer) -> None:
        """
        Constructor
        """
        super().__init__()
        self.name = self.__class__.__name__
        self.trainer = trainer

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
