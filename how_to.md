# How to run the code

Each clean, attack, or defense will store the results in a .csv file. Plus, a .pth file will all the run information will be also stored.

## Clean training
For clean training, you can run the following command:

```bash
python main.py --dataname cifar10 --model resnet --pretrained --epochs 10
```

## Attack
For attack, e.g. BadNets, you can run the following command:

```bash
python main.py --dataname cifar10 --model resnet --pretrained --epochs 10 attack --type badnets --epsilon 0.1
```

### WaNet
```bash
python src/main.py --model googlenet --datadir <data_path> --seed 0 --pretrained attack --type wanet --epsilon 0.05
```

Note that each Attack has its own attack_id. This is later used to pass as an argument to the defense. The defense will then load all the attack information from the attack_id. You can check the attack_id in the .csv file.

## Defense
For defense, e.g. NeuralCleanse, you can run the following command:

```bash
python main.py defense --dataname cifar10 --model resnet --pretrained --epochs 10 --type neuralcleanse --attack_id <attack_id>
```
