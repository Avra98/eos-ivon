#!/bin/bash

FILE="./output"
mkdir -p $FILE

COUNTER=0
gpu_arr=(1 2 4 5 6 7)
LEN=${#gpu_arr[@]}

h0_values=(0.1 0.01 1.0 10.0)
beta2_values=(0.9 0.99 0.999)
learning_rates=(0.1 1.0 0.01 0.0001)
datasets=("cifar10-10k" "mnist-whole")

# Loop over each combination of hyperparameters
for h0 in "${h0_values[@]}"; do
    for beta2 in "${beta2_values[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for dataset in "${datasets[@]}"; do
                python src/gd.py \
                    --dataset $dataset \
                    --lr $lr \
                    --h0 $h0 \
                    --arch_id="fc-relu" \
                    --max_steps=500 \
                    --beta2 $beta2 \
                    --beta=1e-2 \
                    --opt="ivon" \
                    --device_id ${gpu_arr[$((COUNTER % LEN))]} \
                    >> "$FILE"/"h0${h0}_beta2${beta2}_lr${lr}_dataset${dataset}.out" &
                COUNTER=$((COUNTER + 1))
                if [ $((COUNTER % 12)) -eq 0 ]; then
                    wait
                fi
            done
        done
    done
done

# Wait for all background jobs to finish
wait

echo "All jobs completed."
