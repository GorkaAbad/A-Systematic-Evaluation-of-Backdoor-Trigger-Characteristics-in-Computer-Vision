from trainers.Trainer import Trainer
from attacks.Attack import Attack
from defenses.Defense import Defense
from attacks.BadNets import BadNets
from attacks.SSBA import SSBA
from attacks.WaNet import WaNet
from defenses.NeuralCleanse import NeuralCleanse
from defenses.FinePruning import FinePruning
from Helper import Helper

import numpy as np
import torch
import random


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

        Parameters
        ----------
        args : argparse.Namespace
            Arguments

        Returns
        -------
        None
        """
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)

        self.trainer = self.get_trainer(args)

        if args.load_attack is not None:
            self.attack = Helper(args).load_attack(args.load_attack)
            self.trainer = self.attack.trainer

        if args.mode == 'attack':
            self.attack = self.get_attack(args)
        elif args.mode == 'defense':
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
        if args.type == 'badnets':
            att = BadNets(args, self.trainer)
        elif args.type == 'ssba':
            att = SSBA(args, self.trainer)
        elif args.type == 'wanet':
            att = WaNet(args)
        elif args.type == None:
            att = None
        else:
            raise ValueError('Invalid attack')

        return att

    def get_defense(self, args) -> Defense:
        """
        Get defense
        """
        df = None
        if args.type == 'neuralcleanse':
            df = NeuralCleanse(args, self.trainer)
        elif args.type == 'fine-pruning':
            df = FinePruning(args, self.trainer)
        elif args.type == None:
            df = None
        else:
            raise ValueError('Invalid defense')

        return df
