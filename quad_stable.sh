#!/bin/bash

# Bash script to run the Python script for lr ranging from 1.0 to 2.0, excluding 1.0 and 2.0

# Define the Python script to be executed
PYTHON_SCRIPT="range_ivon_stability.py"  # Replace with the actual path to your script

# Loop over the learning rates from 1.1 to 1.9 with a step of 0.1
for lr in $(seq 1.1 0.1 1.9); do
    echo "Running experiment with lr=$lr"
    python $PYTHON_SCRIPT --lr $lr --max_steps 500 --seed 0 --opt ivon --device_id 1
    echo "Completed experiment with lr=$lr"
    echo "-----------------------------------"
done

echo "All experiments completed."

