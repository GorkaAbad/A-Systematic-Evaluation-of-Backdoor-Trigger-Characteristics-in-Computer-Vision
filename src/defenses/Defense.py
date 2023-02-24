from abc import ABC, abstractmethod


class Defense(ABC):
    """
    Defense
    """
    name = None

    def __init__(self, args) -> None:
        """
        Constructor
        """
        super().__init__()
        self.name = self.__class__.__name__

    @abstractmethod
    def execute_defense(self, defense) -> None:
        """
        Get defense
        """
        pass
