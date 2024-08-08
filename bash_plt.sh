#!/bin/bash

# Define arrays
learning_rates=(0.5  0.8  1.0)
#datasets=("cifar10-1k" "cifar10-2k" "cifar10-5k" "cifar10-10k" "cifar10-20k" "mnist-whole" "mnist-20k" "mnist-5k" "mnist-1k")
datasets=("mnist-20k")
activations=("fc-relu-depth4")
#activations=("fc-relu-depth4" "fc-relu-depth6")

# Loop over every combination
for dataset in "${datasets[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for arch in "${activations[@]}"; do
            echo "Processing $dataset with $arch at learning rate $lr"
            python plt.py --dataset "$dataset" --lr "$lr" --arch "$arch"
        done
    done
done
