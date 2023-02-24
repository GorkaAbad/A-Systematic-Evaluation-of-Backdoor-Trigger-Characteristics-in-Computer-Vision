from attacks.Attack import Attack


class WaNet(Attack):
    """
    WaNet Attack
    """

    def __init__(self, args):
        super().__init__()

    def execute_attack(self):
        """
        Get attack
        """
        print('WaNet Attack')
