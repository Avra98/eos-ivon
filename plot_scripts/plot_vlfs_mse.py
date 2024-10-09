
import torch
import numpy as np
import matplotlib.pyplot as plt

arch = "fc-tanh"
loss = "mse"
post = 10


# Updated paths to data with different h0 and post values
base_paths = [
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.07_post_{post}_ivon",
    #f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05/h0_1.0/ess_500000.0/post_{post}/beta2_1.0/beta_0.0_alpha_10.0_ivon",
    "RESULTS/cifar10-10k/fc-tanh/seed_0/mse/gd/lr_0.05"
]

# Initialize lists to store the data
train_accuracies = []
test_accuracies = []
sharpnesses = []

# Define labels based on the varied parameters
labels = [
    r"$IVON:h_0=0.07$",
    #r"$h_0=1.0$",
    r"GD"
]

# Try to load the data for each file
for path in base_paths:
    try:
        train_accuracies.append(torch.load(f"{path}/train_acc_final"))
        test_accuracies.append(torch.load(f"{path}/test_acc_final"))
        sharpnesses.append(torch.load(f"{path}/eigs_final")[:, 0])
    except FileNotFoundError:
        print(f"Missing data for path: {path}")
        continue

# Prepare for plotting with three vertical subplots (Train Accuracy, Test Accuracy, Sharpness)
fig, axs = plt.subplots(3, 1, figsize=(14, 18), dpi=100)

title_fontsize = 20
label_fontsize = 26
tick_fontsize = 28
legend_fontsize = 24  # Larger font size for legends

# Define distinct colors and markers for better differentiation
colors = ['blue', 'orange','red']
linestyles = ['-', '--',':']
markers = ['o', 's','d']

# Update style for lighter gray x/y ticks, axes, and grids
light_gray = '#d3d3d3'  # Light gray color for axis ticks and labels
dark_gray = '#a9a9a9'
middle_gray = '#808080'
gray_c = middle_gray
# Set thicker plot lines
linewidth = 3

# Plot training accuracy
for i, train_acc in enumerate(train_accuracies):
    axs[0].plot(train_acc, label=f"{labels[i]}", color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
axs[0].set_title("Train Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
#axs[0].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
axs[0].spines['top'].set_color('none')  # Remove top spine
axs[0].spines['right'].set_color('none')  # Remove right spine
axs[0].spines['left'].set_color(gray_c)  # Make left spine light gray
axs[0].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
axs[0].grid(True)

# Position the legend in the bottom right corner for the first plot
axs[0].legend(fontsize=legend_fontsize, loc='lower right')

# Plot test accuracy
for i, test_acc in enumerate(test_accuracies):
    axs[1].plot(test_acc, label=f"{labels[i]}", color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
axs[1].set_title("Test Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
#axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
axs[1].set_ylim(0.25, 0.45)
axs[1].spines['top'].set_color('none')  # Remove top spine
axs[1].spines['right'].set_color('none')  # Remove right spine
axs[1].spines['left'].set_color(gray_c)  # Make left spine light gray
axs[1].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
axs[1].grid(True)

# Position the legend in the bottom right corner for the second plot
axs[1].legend(fontsize=legend_fontsize, loc='lower right')

# Plot sharpness (eigenvalues)

## step ratio is 4
for i, sharpness in enumerate(sharpnesses):
    step = 50 if i == 0 else 200  
    if i == 0:
        # Take every 5th index for i == 0
        indices = np.arange(0, len(sharpness), 20)
        axs[2].scatter(indices * step, sharpness[indices], label=f"{labels[i]}", color=colors[i], s=100, edgecolor='black', linewidth=1, alpha=0.8)
    else:
        indices = np.arange(0, len(sharpness), 5)
        axs[2].scatter(indices * step, sharpness[indices], label=f"{labels[i]}", color=colors[i], s=100, edgecolor='black', linewidth=1, alpha=0.8)
    #axs[2].scatter(np.arange(len(sharpness)) * step, sharpness, label=f"{labels[i]}", color=colors[i], s=100, edgecolor='black', linewidth=1, alpha=0.8)
axs[2].axhline(2. / 0.05, linestyle='dotted', label=r"$2/\eta$", color='black')
axs[2].set_title(r"Sharpness (Eigenvalues) $\|\nabla^2 L\|_{2}$", fontsize=title_fontsize, fontweight='bold')
axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
#axs[2].set_ylabel("Sharpness", fontsize=label_fontsize)
axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
axs[2].spines['top'].set_color('none')  # Remove top spine
axs[2].spines['right'].set_color('none')  # Remove right spine
axs[2].spines['left'].set_color(gray_c)  # Make left spine light gray
axs[2].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
axs[2].grid(True)

# Add legend inside for the sharpness plot
axs[2].legend(fontsize=legend_fontsize, loc='center right')

# Adjust layout
plt.tight_layout()
plt.savefig(f"{arch}_{loss}_sharpness_post_{post}_simplified.png", bbox_inches='tight')
plt.show()

