from abc import ABC, abstractmethod
import os


class Defense(ABC):
    """
    Defense
    """
    name = None
    trainer = None

    def __init__(self, trainer) -> None:
        """
        Constructor

        Parameters
        ----------
        trainer : Trainer

        Returns
        -------
        None
        """
        super().__init__()

        self.trainer = trainer
        self.name = self.__class__.__name__

    @abstractmethod
    def execute_defense(self, defense) -> None:
        """
        Get defense
        """
        pass

    @abstractmethod
    def save_results(self, path=None) -> None:
        """
        Save defense results
        """
        pass

    def get_path(self, path) -> str:
        """
        Save defense results
        """

        if not os.path.exists(path):
            os.makedirs(path)

        # Create a folder per defense
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
