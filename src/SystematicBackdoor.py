from trainers.Trainer import Trainer
from attacks.Attack import Attack
from defenses.Defense import Defense
from attacks.BadNets import BadNets
from attacks.SSBA import SSBA
from attacks.WaNet import WaNet
from defenses.NeuralCleanse import NeuralCleanse
from defenses.FinePruning import FinePruning

import numpy as np
import torch


class SystematicBackdoor():
    """
    Systematic Backdoor Attack
    """

    # Attributes
    # ----------

    trainer = None
    attack = None
    defense = None

    # Methods
    # -------

    def __init__(self, args) -> None:
        """
        Constructor
        """
        print('Systematic Backdoor Attack')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self.trainer = self.get_trainer(args)
        self.attack = self.get_attack(args)
        self.defense = self.get_defense(args)

    def get_trainer(self, args) -> Trainer:
        """
        Get trainer
        """
        return Trainer(args)

    def get_attack(self, args) -> Attack:
        """
        Get attack
        """
        att = None
        if args.attack == 'badnets':
            att = BadNets(args, self.trainer)
        elif args.attack == 'ssba':
            att = SSBA(args)
        elif args.attack == 'wanet':
            att = WaNet()
        elif args.attack == None:
            att = None
        else:
            raise ValueError('Invalid attack')

        return att

    def get_defense(self, args) -> Defense:
        """
        Get defense
        """
        df = None
        if args.defense == 'neuralcleanse':
            df = NeuralCleanse(args)
        elif args.defense == 'finepruning':
            df = FinePruning(args)
        elif args.defense == None:
            df = None
        else:
            raise ValueError('Invalid defense')

        return df
