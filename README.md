# A-Systematic-Evaluation-of-Backdoor-Trigger-Characteristics-in-Computer-Vision
This repository contains the code for the paper "A Systematic Evaluation of Backdoor Trigger Characteristics in Computer Vision" submited to CCS'23.

## How to run
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
For MNIST and CIFAR10 no dataset has to be donwloaded manaully. For TinyImagenet execute the following script:
```bash
./tinyimagenet.sh
```

### 3. Run the experiments
Get help for the arguments:
```bash
python main.py --help

usage: main.py [-h] [--dataname {mnist,cifar10,tinyimagenet}] [--model {resnet,googlenet,vgg,alexnet}] [--pretrained]
               [--load-model] [--epsilon EPSILON] [--pos {top-left,top-right,bottom-left,bottom-right,middle,random}]
               [--shape {square,random}] [--color {white,black,green,random}] [--trigger_size TRIGGER_SIZE]
               [--trigger_label TRIGGER_LABEL] [--lr LR] [--loss {mse,cross}] [--optimizer {adam,sgd}]
               [--batch_size BATCH_SIZE] [--batch_size_test BATCH_SIZE_TEST] [--epochs EPOCHS] [--device DEVICE]
               [--seed SEED] [--datadir DATADIR] [--pretrained_path PRETRAINED_PATH] [--save_path SAVE_PATH]

Backdoor attack

optional arguments:
  -h, --help            show this help message and exit
  --dataname {mnist,cifar10,tinyimagenet}
                        The dataset to use
  --model {resnet,googlenet,vgg,alexnet}
                        The model to use
  --pretrained          Use pretrained weights
  --load-model          Load a saved model from the corresponding folder
  --epsilon EPSILON     The rate of poisoned data
  --pos {top-left,top-right,bottom-left,bottom-right,middle,random}
                        The position of the trigger
  --shape {square,random}
                        The shape of the trigger
  --color {white,black,green,random}
                        The color of the trigger
  --trigger_size TRIGGER_SIZE
                        The size of the trigger in percentage of the image size
  --trigger_label TRIGGER_LABEL
                        The label of the target/objective class. The class to be changed to.
  --lr LR               Learning rate
  --loss {mse,cross}    The loss function to use
  --optimizer {adam,sgd}
                        The optimizer to use
  --batch_size BATCH_SIZE
                        Train batch size
  --batch_size_test BATCH_SIZE_TEST
                        Test batch size
  --epochs EPOCHS       Number of epochs
  --device DEVICE       Device to use
  --seed SEED           Random seed
  --datadir DATADIR     path to save downloaded data
  --pretrained_path PRETRAINED_PATH
                        path to save downloaded pretrained model
  --save_path SAVE_PATH
                        path to save training results
```

For example, use CIFAR10 with Resnet and pretrained weights, using a black trigger in the top-left corner of the image, with a size of 10% of the image size, and a poisoning rate of 0.1:

```bash
    python main.py --dataname cifar10 --model resnet --pretrained --pos top-left --shape square --color black --trigger_size 0.1 --epsilon 0.1
```