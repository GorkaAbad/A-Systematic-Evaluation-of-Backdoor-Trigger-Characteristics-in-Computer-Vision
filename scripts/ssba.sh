epsilons="0.0 0.005 0.01 0.015 0.02"
seeds="0 1 2 3 4"
models="resnet googlenet alexnet vgg"

for seed in $seeds; do
    for model in $models; do
        for epsilon in $epsilons; do
            python src/main.py --model $model --seed $seed --pretrained attack --type ssba --epsilon $epsilon
        done
    done
done