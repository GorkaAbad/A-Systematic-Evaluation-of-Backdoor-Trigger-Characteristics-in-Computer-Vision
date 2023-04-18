# A Systematic Evaluation framework of Backdoor Attacks on Deep Neural Networks

## How to use
The tools are divided into modules:

- attacks: contains the backdoor attack methods
  - Attack.py: the base class of all attacks (abstract)
  - BadNets.py: the implementation of BadNets attack
  - SSBA.py: the implementation of SSBA attack
  - WaNet.py: the implementation of WaNet attack

- datasets: contains the dataset classes
  - Dataset.py: the base class of all datasets (abstract)
  - CIFAR10.py: the implementation of CIFAR10 dataset
  - MNIST.py: the implementation of MNIST dataset
  - TinyImageNet.py: the implementation of TinyImageNet dataset

- defenses: contains the defense methods
   - Defense.py: the base class of all defenses (abstract)
   - NeuralCleanse.py: the implementation of NeuralCleanse defense
   - Fine-Pruning.py: the implementation of Fine-Pruning defense

- models: contains the model classes
  - Model.py: the base class of all models

- trainers: helper class for training the model
  - Trainer.py: the base class containing all the functions

- Helper.py: contains the helper functions
- SystematicBackdoor.py: cantains the high level logic (abstract)
- main.py: the main file to run the framework


## How to run
The framework can be executed in 3 modes:
  - clean: Train clean models. You don't have to specify any positional arguments (default)
  - attack: Train models with backdoor attacks. You have to specify the attack method and its parameters
  - defense: Train models with backdoor defenses. You have to specify the defense method and its parameters

Note that you cannot use different modes in the same run. You have to run the framework multiple times for each mode.

See [How to use](./how_to.md) for more details.

### Clean

```bash
python main.py --help

usage: main.py [-h] [--dataname {mnist,cifar10,tinyimagenet}] [--model {resnet,googlenet,vgg,alexnet}] [--pretrained]
               [--lr LR] [--loss {mse,cross}] [--optimizer {adam,sgd}] [--momentum MOMENTUM]
               [--weight_decay WEIGHT_DECAY] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--seed SEED]
               [--datadir DATADIR] [--amp] [--save_path SAVE_PATH] [--load_model LOAD_MODEL]
               [--load_attack LOAD_ATTACK]
               {attack,defense} ...

Systematic Backdoor Attack

positional arguments:
  {attack,defense}
    attack              Attack help
    defense             Defense help

optional arguments:
  -h, --help            show this help message and exit
  --dataname {mnist,cifar10,tinyimagenet}
                        The dataset to use
  --model {resnet,googlenet,vgg,alexnet}
                        The model to use
  --pretrained          Use pretrained model
  --lr LR               Learning rate
  --loss {mse,cross}    The loss function to use
  --optimizer {adam,sgd}
                        The optimizer to use
  --momentum MOMENTUM   Momentum for SGD optimizer
  --weight_decay WEIGHT_DECAY
                        Weight decay for SGD optimizer
  --batch_size BATCH_SIZE
                        Train batch size
  --epochs EPOCHS       Number of epochs
  --seed SEED           Random seed
  --datadir DATADIR     path to save downloaded data
  --amp                 Use automatic mixed precision
  --save_path SAVE_PATH
                        path to save training results
  --load_model LOAD_MODEL
                        path to load model
  --load_attack LOAD_ATTACK
                        path to load attack
```

### Attack

```bash
python main.py attack --help

usage: main.py attack [-h] [--type {badnets,ssba,wanet}] [--target_label TARGET_LABEL] [--epsilon EPSILON]
                      [--pos {top-left,top-right,bottom-left,bottom-right,middle,random}]
                      [--color {white,black,green}] [--trigger_size TRIGGER_SIZE] [--s S] [--cross_ratio CROSS_RATIO]
                      [--grid_rescale GRID_RESCALE] [--device DEVICE] [--random_crop RANDOM_CROP]
                      [--random_rotation RANDOM_ROTATION] [--k K] [--ckpt_path CKPT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --type {badnets,ssba,wanet}
                        Type of the attack
  --target_label TARGET_LABEL
                        The label of the target/objective class. The class to be changed to.
  --epsilon EPSILON     The rate of poisoned data

Badnets:
  --pos {top-left,top-right,bottom-left,bottom-right,middle,random}
                        The position of the trigger
  --color {white,black,green}
                        The color of the trigger
  --trigger_size TRIGGER_SIZE
                        The size of the trigger in percentage of the image size

WANet:
  --s S                 the parameter used to define the strength of P(backward warping field)
  --cross_ratio CROSS_RATIO
  --grid_rescale GRID_RESCALE
                        scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98
  --device DEVICE
  --random_crop RANDOM_CROP
  --random_rotation RANDOM_ROTATION
  --k K                 size of uniform grid
  --ckpt_path CKPT_PATH
```

### Defense

```bash
python main.py defense --help

usage: main.py defense [-h] [--type {neuralcleanse,fine-pruning}] [--attack_id ATTACK_ID] [--nc_lr NC_LR]
                       [--nc_init_cost NC_INIT_COST] [--nc_atk_succ_threshold NC_ATK_SUCC_THRESHOLD] [--nc_early_stop]
                       [--nc_early_stop_threshold NC_EARLY_STOP_THRESHOLD]
                       [--nc_early_stop_patience NC_EARLY_STOP_PATIENCE] [--nc_patience NC_PATIENCE]
                       [--nc_cost_multiplier NC_COST_MULTIPLIER] [--nc_epochs NC_EPOCHS] [--nc_epsilon NC_EPSILON]
                       [--nc_n_times_test NC_N_TIMES_TEST] [--pruning_rate PRUNING_RATE] [--fp_epochs FP_EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --type {neuralcleanse,fine-pruning}
                        Type of the defense
  --attack_id ATTACK_ID
                        id of the attack

NeuralCleanse:
  --nc_lr NC_LR
  --nc_init_cost NC_INIT_COST
  --nc_atk_succ_threshold NC_ATK_SUCC_THRESHOLD
  --nc_early_stop
  --nc_early_stop_threshold NC_EARLY_STOP_THRESHOLD
  --nc_early_stop_patience NC_EARLY_STOP_PATIENCE
  --nc_patience NC_PATIENCE
  --nc_cost_multiplier NC_COST_MULTIPLIER
  --nc_epochs NC_EPOCHS
  --nc_epsilon NC_EPSILON
  --nc_n_times_test NC_N_TIMES_TEST

Fine-pruning:
  --pruning_rate PRUNING_RATE
                        The rate of neurons to be pruned
  --fp_epochs FP_EPOCHS
                        The number of epochs to train the pruned model
```
