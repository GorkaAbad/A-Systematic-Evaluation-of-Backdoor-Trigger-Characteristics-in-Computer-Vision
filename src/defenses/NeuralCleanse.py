from defenses.Defense import Defense


class NeuralCleanse(Defense):
    """
    Neural Cleanse
    """

    # Attributes
    # ----------
    defense = None

    # Methods
    # -------
    def __init__(self, defense):
        """
        Constructor
        """
        super().__init__(defense)

    def execute_defense(self):
        """
        Get defense
        """
        print('Neural Cleanse Defense')
