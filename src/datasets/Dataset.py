from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Dataset
    """
    trainset = None
    testset = None
    n_classes = None
    name = None

    def __init__(self, args) -> None:
        """
        Constructor

        Parameters
        ----------
        args : argparse.Namespace
            Arguments

        Returns
        -------
        None

        """
        super().__init__()
        self.trainset, self.testset = self.get_dataset(args)
        self.n_classes = self.get_n_classes()
        self.name = self.__class__.__name__

    @abstractmethod
    def get_dataset(self, args):
        """
        Get dataset
        """
        pass

    @abstractmethod
    def get_n_classes(self) -> int:
        """
        Get number of classes
        """
        pass
