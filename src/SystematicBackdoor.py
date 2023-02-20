from models.Model import Model
from datasets.Dataset import Dataset
from attacks.Attack import Attack
from defenses.Defense import Defense


class SystematicBackdoor():
    """
    Systematic Backdoor Attack
    """

    # Attributes
    # ----------

    model = None
    dataset = None
    attack = None
    defense = None

    # Methods
    # -------

    def __init__(self, model, dataset, attack, defense):
        """
        Constructor
        """
        print('Systematic Backdoor Attack')
        self.model = Model(model)
        self.dataset = Dataset.Dataset(dataset)
        self.attack = Attack(attack)
        self.defense = Defense(defense)
