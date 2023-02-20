from abc import ABC, abstractmethod


class Attack(ABC):
    """
    Attack
    """
    attack = None

    def __init__(self, attack):
        """
        Constructor
        """
        self.attack = self.get_attack(attack)

    @abstractmethod
    def get_attack(self, attack):
        """
        Get attack
        """
        pass
