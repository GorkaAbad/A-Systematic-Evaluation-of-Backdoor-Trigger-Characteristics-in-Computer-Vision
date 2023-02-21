from attacks.Attack import Attack


class SSBA(Attack):
    """
    SSBA Attack
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

    def execute_attack(self):
        """
        Get attack
        """
        print('SSBA Attack')
