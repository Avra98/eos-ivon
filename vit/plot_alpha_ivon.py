import torch
import numpy as np
import matplotlib.pyplot as plt

# File paths for alpha = 3, 4, 5, 8, 12
base_paths = [
    "RESULTS/ivon/vit/1.0/3.0",
    "RESULTS/ivon/vit/1.0/4.0",
  #  "RESULTS/ivon/vit/1.0/5.0",
    "RESULTS/ivon/vit/1.0/8.0",
    "RESULTS/ivon/vit/1.0/12.0",
]

# Labels for the legends
labels = [
    r"$\alpha=3.0$", r"$\alpha=4.0$", #r"IVON $\alpha=5.0$",
    r"$\alpha=8.0$", r"$\alpha=12.0$"
]

# Extended arrays for colors, markers, and linestyles
colors = ['blue', 'orange', 'green', 'red', 'purple']
markers = ['o', 's', '^', 'v', 'D']
linestyles = ['-', '--', '-.', ':', '-']

# Initialize empty lists to load data
train_accuracies = []
test_accuracies = []
eigenvalues = []

# Load and process data
for path in base_paths:
    try:
        train_accuracies.append(np.array(torch.load(f"{path}/train_acc.pt")))
        test_accuracies.append(np.array(torch.load(f"{path}/test_acc.pt")))
        eigenvalues.append(np.array(torch.load(f"{path}/eigenvalues.pt")))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Plot settings
fig, axs = plt.subplots(3, 1, figsize=(20, 20), dpi=120)

title_fontsize = 28
label_fontsize = 24
legend_fontsize = 24
tick_fontsize = 24

# Training Accuracy
for i, train_acc in enumerate(train_accuracies):
    axs[0].plot(
        np.arange(len(train_acc)) * 50, train_acc, label=f"{labels[i]}",
        marker=markers[i % len(markers)], color=colors[i % len(colors)], 
        linestyle=linestyles[i % len(linestyles)], linewidth=2, markersize=6
    )
axs[0].set_title("Training Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
#axs[0].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[0].grid(True)
axs[0].legend(fontsize=legend_fontsize)

# Test Accuracy
for i, test_acc in enumerate(test_accuracies):
    axs[1].plot(
        np.arange(len(test_acc)) * 50, test_acc, label=f"{labels[i]}",
        marker=markers[i % len(markers)], color=colors[i % len(colors)], 
        linestyle=linestyles[i % len(linestyles)], linewidth=2, markersize=6
    )
axs[1].set_title("Test Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
#axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)

# Customize x and y ticks for Test Accuracy
x_ticks = np.arange(0, 110000, 20000)  # X-axis ticks every 20,000
y_ticks = np.arange(0.58, 0.64, 0.01)  # Y-axis ticks every 0.01
axs[1].set_xticks(x_ticks)
axs[1].set_yticks(y_ticks)

axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[1].grid(True, which='major', linewidth=1)
axs[1].legend(fontsize=legend_fontsize)
axs[1].set_ylim(0.56, 0.61)

# Eigenvalues (Sharpness)
for i, eigs in enumerate(eigenvalues):
    axs[2].scatter(
        np.arange(len(eigs)) * 200, eigs, label=f"{labels[i]}",
        color=colors[i % len(colors)], marker=markers[i % len(markers)], s=40, alpha=0.8
    )
axs[2].axhline(400, color='black', linestyle='dashed', linewidth=2, label=r"$\frac{2}{\rho}$")
axs[2].set_title(r"Sharpness", fontsize=title_fontsize, fontweight='bold')
axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
#axs[2].set_ylabel("Eigenvalue Magnitude", fontsize=label_fontsize)
axs[2].set_ylim(0, 500)
axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[2].grid(True)
axs[2].legend(fontsize=legend_fontsize)

# Save and show the plot
output_path = "vit/vl_plots_alpha.png"
plt.tight_layout()
plt.savefig(output_path, format='png', bbox_inches='tight')
plt.show()
