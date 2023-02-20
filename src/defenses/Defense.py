from abc import ABC, abstractmethod


class Defense(ABC):
    """
    Defense
    """
    defense = None

    def __init__(self, defense):
        """
        Constructor
        """
        self.defense = self.get_defense(defense)

    @abstractmethod
    def get_defense(self, defense):
        """
        Get defense
        """
        pass
