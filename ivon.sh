# #!/bin/bash
# FILE="./ivon_sharp_correct-alphas"
# mkdir -p $FILE

# COUNTER=0
# gpu_arr=(1 2 3 4 5 6)
# LEN=${#gpu_arr[@]}

# # Define h0 values based on arch_id and loss combinations
# h0_fc_tanh_ce=(1.0)
# h0_resnet_ce=(0.03 0.1 0.5 1.0 5.0)
# h0_fc_tanh_mse=(1.0)
# h0_resnet_mse=(0.1 0.01 0.5 5.0) 
# alphas=(1 2 2.5 3 10 100)
# beta2_values=(1.0)
# learning_rates=(0.05)
# post_samples=(1)
# datasets=("cifar10-10k")
# losses=("mse")
# arch_ids=("fc-tanh")

# # Loop over arch_id first
# for arch_id in "${arch_ids[@]}"; do
#     # Determine the counter limit based on arch_id
#     if [ "$arch_id" == "fc-tanh" ]; then
#         COUNTER_LIMIT=3
#     else
#         COUNTER_LIMIT=3
#     fi
    
#     # Loop over each combination of hyperparameters
#     for post_sample in "${post_samples[@]}"; do
#         for beta2 in "${beta2_values[@]}"; do
#             for lr in "${learning_rates[@]}"; do
#                 for dataset in "${datasets[@]}"; do
#                     for loss in "${losses[@]}"; do
#                         for alpha in "${alphas[@]}"; do
                        
#                             # Select h0 values based on arch_id and loss
#                             if [ "$arch_id" == "fc-tanh" ] && [ "$loss" == "ce" ]; then
#                                 h0_values=("${h0_fc_tanh_ce[@]}")
#                             elif [ "$arch_id" == "resnet32" ] && [ "$loss" == "ce" ]; then
#                                 h0_values=("${h0_resnet_ce[@]}")
#                             elif [ "$arch_id" == "fc-tanh" ] && [ "$loss" == "mse" ]; then
#                                 h0_values=("${h0_fc_tanh_mse[@]}")
#                             elif [ "$arch_id" == "resnet32" ] && [ "$loss" == "mse" ]; then
#                                 h0_values=("${h0_resnet_mse[@]}")
#                             fi

#                             for h0 in "${h0_values[@]}"; do
#                                 python src/gd.py \
#                                     --dataset $dataset \
#                                     --lr $lr \
#                                     --h0 $h0 \
#                                     --arch_id=$arch_id \
#                                     --max_steps=10000 \
#                                     --beta2 $beta2 \
#                                     --beta=0.0 \
#                                     --opt="ivon" \
#                                     --eig_freq=50 \
#                                     --post_samples $post_sample \
#                                     --loss $loss \
#                                     --alpha $alpha \
#                                     --device_id ${gpu_arr[$((COUNTER % LEN))]} \
#                                     >> "$FILE"/"post${post_sample}_h0${h0}_beta2${beta2}_alpha${alpha}_lr${lr}_dataset${dataset}_loss${loss}_arch${arch_id}.out" &
#                                 COUNTER=$((COUNTER + 1))
#                                 if [ $((COUNTER % COUNTER_LIMIT)) -eq 0 ]; then
#                                     wait
#                                 fi
#                             done    
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# # Wait for all background jobs to finish
# wait

# echo "All jobs completed."



#!/bin/bash
FILE="./ivon_actual_cifarwhole"
mkdir -p $FILE

COUNTER=0
gpu_arr=(0 1 3 4 5 6 7)
LEN=${#gpu_arr[@]}

# Define h0 values based on arch_id and loss combinations
h0_fc_tanh_ce=(0.01 0.07 0.1 0.5 1.0 5.0 10.0)
#h0_resnet_ce=(0.03 0.1 0.5 1.0 5.0)
h0_fc_tanh_mse=(0.01 0.07 0.1 0.5 1.0 5.0 10.0)
#h0_resnet_mse=(0.1 0.01 0.5 1.0 10.0) 
alphas=(100)
beta2_values=(1.0)
ess=(1e5)
learning_rates=(5e-2)
post_samples=(10)
datasets=("cifar10")
losses=("ce" "mse")
arch_ids=("fc-tanh")

# Loop over arch_id first
for arch_id in "${arch_ids[@]}"; do
    # Determine the counter limit based on arch_id
    if [ "$arch_id" == "fc-tanh" ]; then
        COUNTER_LIMIT=6
    else
        COUNTER_LIMIT=3
    fi
    
    # Loop over each combination of hyperparameters
    for post_sample in "${post_samples[@]}"; do
        for beta2 in "${beta2_values[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for dataset in "${datasets[@]}"; do
                    for loss in "${losses[@]}"; do
                        for alpha in "${alphas[@]}"; do
                            for ess in "${ess[@]}"; do
                        
                                # Select h0 values based on arch_id and loss
                                if [ "$arch_id" == "fc-tanh" ] && [ "$loss" == "ce" ]; then
                                    h0_values=("${h0_fc_tanh_ce[@]}")
                                elif [ "$arch_id" == "resnet32" ] && [ "$loss" == "ce" ]; then
                                    h0_values=("${h0_resnet_ce[@]}")
                                elif [ "$arch_id" == "fc-tanh" ] && [ "$loss" == "mse" ]; then
                                    h0_values=("${h0_fc_tanh_mse[@]}")
                                elif [ "$arch_id" == "resnet32" ] && [ "$loss" == "mse" ]; then
                                    h0_values=("${h0_resnet_mse[@]}")
                                fi

                                for h0 in "${h0_values[@]}"; do
                                    python src/gd.py \
                                        --dataset $dataset \
                                        --lr $lr \
                                        --h0 $h0 \
                                        --arch_id=$arch_id \
                                        --max_steps=30000 \
                                        --beta2 $beta2 \
                                        --beta=0.0 \
                                        --opt="ivon" \
                                        --eig_freq=50 \
                                        --post_samples $post_sample \
                                        --loss $loss \
                                        --alpha $alpha \
                                        --ess $ess \
                                        --device_id ${gpu_arr[$((COUNTER % LEN))]} \
                                        >> "$FILE"/"post${post_sample}_h0${h0}_beta2${beta2}_alpha${alpha}_lr${lr}_dataset${dataset}_loss${loss}_ess${ess}_arch${arch_id}.out" &
                                    COUNTER=$((COUNTER + 1))
                                    if [ $((COUNTER % COUNTER_LIMIT)) -eq 0 ]; then
                                        wait
                                    fi
                                done    
                            done    
                        done
                    done
                done
            done
        done
    done
done

# Wait for all background jobs to finish
wait

echo "All jobs completed."