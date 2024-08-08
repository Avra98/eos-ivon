import torch
import numpy as np
import matplotlib.pyplot as plt
from os import environ
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description="Plot training results")
parser.add_argument('--dataset', type=str, required=True, help='Dataset to process')
parser.add_argument('--lr', type=float, required=True, help='Learning rate')
parser.add_argument('--arch', type=str, required=True, help='Architecture')

args = parser.parse_args()

# Load the data
dataset = args.dataset
lr = args.lr
arch = args.arch
path = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/mse/gd/lr_{lr}"

gd_train_loss = torch.load(f"{path}/train_loss_final")
gd_train_acc = torch.load(f"{path}/train_acc_final")
gd_sharpness = torch.load(f"{path}/eigs_final")[:,0]
singular_value_storage = torch.load(f"{path}/singular_value_storage")
subspace_distances = torch.load(f"{path}/subspace_distances")

# Prepare for plotting
num_layers = len(singular_value_storage)  # Adjusted to include the last layer
num_main_plots = 3  # Training loss, accuracy, sharpness
total_plots = num_main_plots + num_layers * 2  # Including subspace distances
num_cols = 3  # Desired number of columns
num_rows = 3  # Number of rows

fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), dpi=100)
axs = axs.flatten()  # Flatten to simplify indexing

title_fontsize = 16
label_fontsize = 14
legend_fontsize = 14
tick_fontsize = 14  # Font size for tick labels

# Plot training loss
axs[0].plot(gd_train_loss, label="Train Loss")
axs[0].set_title("Train Loss", fontsize=title_fontsize)
axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
axs[0].set_ylabel("Loss", fontsize=label_fontsize)
axs[0].legend(fontsize=legend_fontsize)
axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[0].grid(True)

# Plot training accuracy
axs[1].plot(gd_train_acc, label="Train Accuracy")
axs[1].set_title("Train Accuracy", fontsize=title_fontsize)
axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[1].legend(fontsize=legend_fontsize)
axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[1].grid(True)

# Plot sharpness
axs[2].scatter(np.arange(len(gd_sharpness)) * 5, gd_sharpness, label="Sharpness", s=10)
axs[2].axhline(2. / lr, linestyle='dotted', label=r"$2/\eta$")
axs[2].set_title("Sharpness", fontsize=title_fontsize)
axs[2].set_xlabel("Iteration", fontsize=label_fontsize)
axs[2].set_ylabel("Sharpness Value", fontsize=label_fontsize)
axs[2].legend(fontsize=legend_fontsize)
axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[2].grid(True)

# Plot singular values and subspace distances for each layer
for i, layer_data in enumerate(singular_value_storage):
    if i >= 3:  # Limit to 3 layers
        break

    singular_values = layer_data['singular_values']
    plot_index_singular_values = 3 + i
    if singular_values:
        all_singular_values = np.stack(singular_values)
        print(f"Layer {i+1}: all_singular_values shape = {all_singular_values.shape}")
        if all_singular_values.ndim > 1:  # Ensure it has the right dimensions
            num_singular_values = all_singular_values.shape[1]
            indices_top_5 = range(5)
            indices_last_5 = range(max(0, num_singular_values - 5), num_singular_values)
            iterations_scaled = np.arange(all_singular_values.shape[0]) * 5
            for idx in indices_top_5:
                axs[plot_index_singular_values].plot(iterations_scaled, all_singular_values[:, idx], label=fr'$\sigma_{{{idx+1}}}$')
            for idx in indices_last_5:
                axs[plot_index_singular_values].plot(iterations_scaled, all_singular_values[:, idx], label=fr'$\sigma_{{{num_singular_values - (num_singular_values - idx - 1)}}}$')
            axs[plot_index_singular_values].set_title(f'Layer {i+1} Singular Value Evolution', fontsize=title_fontsize)
            axs[plot_index_singular_values].set_xlabel('Iteration', fontsize=label_fontsize)
            axs[plot_index_singular_values].set_ylabel('Singular Value', fontsize=label_fontsize)
            axs[plot_index_singular_values].legend(loc='lower right', fontsize=legend_fontsize, ncol=2)  # Adjusted legend to have 2 columns
            axs[plot_index_singular_values].tick_params(axis='both', which='major', labelsize=tick_fontsize)
            axs[plot_index_singular_values].grid(True)
        else:
            print(f"Skipping Layer {i+1}: all_singular_values has incorrect shape {all_singular_values.shape}")
            axs[plot_index_singular_values].axis('off')
    else:
        print(f"Skipping Layer {i+1}: No singular values recorded.")
        axs[plot_index_singular_values].axis('off')  # Turn off unused subplots

    # Plot subspace distances for each layer
    plot_index_subspace = 6 + i
    if subspace_distances[i]:
        subspace_distances_layer = np.array(subspace_distances[i])
        print(f"Layer {i+1}: subspace_distances_layer shape = {subspace_distances_layer.shape}")
        iterations_scaled = np.arange(len(subspace_distances_layer)) * 5
        axs[plot_index_subspace].plot(iterations_scaled, subspace_distances_layer)
        axs[plot_index_subspace].set_title(f'Layer {i+1} Subspace Distance Evolution', fontsize=title_fontsize)
        axs[plot_index_subspace].set_xlabel('Iteration', fontsize=label_fontsize)
        axs[plot_index_subspace].set_ylabel('Subspace Distance', fontsize=label_fontsize)
        axs[plot_index_subspace].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[plot_index_subspace].grid(True)
    else:
        print(f"Skipping Layer {i+1}: No subspace distances recorded.")
        axs[plot_index_subspace].axis('off')  # Turn off unused subplots

# Hide any unused axes
for ax in axs[total_plots:]:
    ax.axis('off')

plt.tight_layout()
plt.savefig(f"complete_fig_data_{dataset}_lr_{lr}_arch_{arch}.png")
plt.savefig(f"complete_fig_data_{dataset}_lr_{lr}_arch_{arch}.svg")
plt.show()
