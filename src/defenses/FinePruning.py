from defenses.Defense import Defense


class FinePruning(Defense):
    """
    Fine Pruning
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
        print('Fine Pruning Defense')
