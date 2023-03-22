from attacks.Attack import Attack
import random
import torch, torchvision
import kornia.augmentation as A


class WaNet(Attack):
    """
    WaNet Attack
    """

    def __init__(self, args, trainer):
        super().__init__(trainer, args.target_label)
        self.identity_grid =  args.identity_grid
        self.s = args.s
        self.noise_grid = args.noise_grid
        self.input_height = args.input_height
        self.grid_rescale = args.grid_rescale
        self.cross_ratio = args.cross_ratio
        self.device = args.device
        self.input_width = args.input_width
        self.random_crop = args.random_crop
        self.random_rotation = args.random_rotation

    def execute_attack(self):
        """
        Get attack
        """
        transforms = PostTensorTransform().to(self.device)
        # Create a copy of the orginal training set and test set
        poisoned_trainset = deepcopy(self.trainer.dataset.trainset)
        poisoned_testset = deepcopy(self.trainer.dataset.testset)
        num_test = len(poisoned_testset)

        # Get a random subset of the training set
        perm = torch.randperm(len(poisoned_trainset))
        idx = perm[:int(len(poisoned_trainset) * self.epsilon)]
        num_bd = len(idx)    
        num_cross = int(num_bd * self.cross_ratio)    
        idx_cross = perm[num_bd:num_bd+num_cross]

        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)        

        ins = torch.rand(num_cross, self.input_height, self.input_height, 2).to(self.device) * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / self.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        inputs_bd = F.grid_sample(poisoned_trainset[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        inputs_cross = F.grid_sample(poisoned_trainset[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)
        poisoned_trainset.data[idx] = inputs_bd
        poisoned_trainset.data[idx_cross] = inputs_cross
        poisoned_trainset.data = transforms(poisoned_trainset.data)

        # Change the label to the target label
        poisoned_trainset.targets = torch.as_tensor(poisoned_trainset.targets)
        poisoned_trainset.targets[idx] = self.target_label

        # Poison the test set
        inputs_bd = F.grid_sample(poisoned_testset, grid_temps.repeat(num_test, 1, 1, 1), align_corners=True)
        poisoned_testset.data[:] = inputs_bd
        poisoned_testset.targets = torch.as_tensor(poisoned_trainset.targets)
        poisoned_testset.targets[:] = self.target_label

        # Create a new trainer with the poisoned training set
        self.trainer.poisoned_dataset = deepcopy(self.trainer.dataset)
        self.trainer.poisoned_dataset.trainset = poisoned_trainset
        self.trainer.poisoned_dataset.testset = poisoned_testset
        self.trainer.poisoned_trainloader, self.trainer.poisoned_testloader = self.trainer.get_dataloader(
            clean=False)

        print(
            f'Successfully created a poisoned dataset with {self.epsilon * 100}% of the training set')
        print(f'Original training set: {len(self.trainer.dataset.trainset)}')
        print(
            f'Poisoned training set clean: {len(self.trainer.poisoned_dataset.trainset) - len(num_bd)} Backdoor: {len(num_bd)}')

        self.backdoor_train()
        print('WaNet Attack')


    def backdoor_train(self) -> None:
        """
        Train the model with the poisoned training set
        """
        self.trainer.train(clean=False)


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return 
    
class PostTensorTransform(torch.nn.Module):
    def __init__(self):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((self.input_height, self.input_width), padding=self.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(self.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
