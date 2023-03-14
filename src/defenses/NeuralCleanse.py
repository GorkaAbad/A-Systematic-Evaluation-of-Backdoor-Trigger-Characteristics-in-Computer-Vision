from defenses.Defense import Defense


class NeuralCleanse(Defense):
    """
    Neural Cleanse
    """

    # Attributes
    # ----------
    anomaly_threshold = None
    cost_multiplier = None
    asr_threshold = None
    patience = None

    # Methods
    # -------
    def __init__(self, args, trainer):
        """
        Constructor
        """
        super().__init__(trainer)
        self.anomaly_threshold = args.anomaly_threshold
        self.cost_multiplier = args.cost_multiplier
        self.asr_threshold = args.asr_threshold
        self.patience = args.patience

    def execute_defense(self):
        """
        Get defense
        """

        print('Neural Cleanse Defense')

    def optimize_trigger(self):
        """
        Optimize the trigger
        """
