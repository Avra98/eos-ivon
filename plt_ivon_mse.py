import torch
import numpy as np
import matplotlib.pyplot as plt

arch = "resnet32"
# Paths to the specified data
base_path1 = f"RESULTS/cifar10-10k/{arch}/seed_0/mse/ivon/lr_0.05_h0_0.01_post_2_ivon"
base_path2 = f"RESULTS/cifar10-10k/{arch}/seed_0/mse/ivon/lr_0.05_h0_0.01_post_10_ivon"
base_path3 = f"RESULTS/cifar10-10k/{arch}/seed_0/mse/ivon/lr_0.05_h0_0.1_post_2_ivon"
base_path4 = f"RESULTS/cifar10-10k/{arch}/seed_0/mse/ivon/lr_0.05_h0_0.1_post_10_ivon"
base_path5 = f"RESULTS/cifar10-10k/{arch}/seed_0/mse/ivon/lr_0.05_h0_10.0_post_2_ivon"
base_path6 = f"RESULTS/cifar10-10k/{arch}/seed_0/mse/ivon/lr_0.05_h0_10.0_post_10_ivon"

# Store paths in a list
paths = [base_path1, base_path2, base_path3, base_path4, base_path5, base_path6]

# Initialize lists to store the data
train_losses = []
train_accuracies = []
sharpnesses = []

# Load the data for each location
for path in paths:
    train_losses.append(torch.load(f"{path}/train_loss_final"))
    train_accuracies.append(torch.load(f"{path}/train_acc_final"))
    sharpnesses.append(torch.load(f"{path}/eigs_final")[:, 0])

# Prepare for plotting with vertical subplots
fig, axs = plt.subplots(3, 1, figsize=(15, 18), dpi=100)

title_fontsize = 16
label_fontsize = 12
legend_fontsize = 10  # Smaller font size for the legend
tick_fontsize = 14  # Font size for tick labels

# Define labels and styles for the paths
labels = [
    "Ivon h=0.01, post=2", "Ivon h=0.01, post=10", 
    "Ivon h=0.1, post=2", "Ivon h=0.1, post=10",
    "Ivon h=10.0, post=2", "Ivon h=10.0, post=10"
]

markers = ['o', '^', 's', 'D', 'v', 'p']  # Markers for the plots
colors = ['navy', 'darkorange', 'forestgreen', 'darkred', 'purple', 'brown']
linestyles = ['-', '--', '-.', ':', '-', '--']  # Linestyles for the plots

# Plot training loss
for i, train_loss in enumerate(train_losses):
    axs[0].plot(train_loss, label=f"Train Loss - {labels[i]}", marker=markers[i], color=colors[i], linestyle=linestyles[i], linewidth=1.5, markersize=4)
axs[0].set_title("Train Loss", fontsize=title_fontsize)
axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
axs[0].set_ylabel("Loss", fontsize=label_fontsize)
axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[0].grid(True)

# Plot training accuracy
for i, train_acc in enumerate(train_accuracies):
    axs[1].plot(train_acc, label=f"Train Accuracy - {labels[i]}", marker=markers[i], color=colors[i], linestyle=linestyles[i], linewidth=1.5, markersize=4)
axs[1].set_title("Train Accuracy", fontsize=title_fontsize)
axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[1].grid(True)

# Plot sharpness (eigenvalues)
for i, sharpness in enumerate(sharpnesses):
    axs[2].scatter(np.arange(len(sharpness)) * 5, sharpness, label=f"Sharpness - {labels[i]}", color=colors[i], marker=markers[i], s=40, edgecolors='black', linewidths=0.5)
axs[2].axhline(2. / 0.05, linestyle='dotted', label=r"$2/\eta$", color='black')
axs[2].set_title("Sharpness", fontsize=title_fontsize)
axs[2].set_xlabel("Iteration", fontsize=label_fontsize)
axs[2].set_ylabel("Sharpness Value", fontsize=label_fontsize)
axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[2].grid(True)

# Move legends outside the plots for all subplots
for ax in axs:
    ax.legend(fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig(f"sharpness_ivon_Resnet_mse.png", bbox_inches='tight')
plt.savefig(f"sharpness_ivon_Resnet_mse.svg", bbox_inches='tight')
plt.show()
