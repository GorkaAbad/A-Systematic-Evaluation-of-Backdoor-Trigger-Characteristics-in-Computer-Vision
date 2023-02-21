from abc import ABC, abstractmethod


class Defense(ABC):
    """
    Defense
    """
    defense = None

    def __init__(self, defense) -> None:
        """
        Constructor
        """
        super().__init__()

    @abstractmethod
    def execute_defense(self, defense) -> None:
        """
        Get defense
        """
        pass
