import torch 
import numpy as np
import matplotlib.pyplot as plt

arch = "fc-tanh"
loss = "ce"
data='cifar10'
post=10
ess=100000.0

base_paths = [
   # f"RESULTS/{data}/{arch}/seed_0/{loss}/ivon/lr_0.05/h0_10.0/ess_{ess}/post_10/beta2_1.0/beta_0.0_alpha_100.0_ivon",
    f"RESULTS/{data}/{arch}/seed_0/{loss}/ivon/lr_0.05/h0_5.0/ess_{ess}/post_10/beta2_1.0/beta_0.0_alpha_100.0_ivon",
    f"RESULTS/{data}/{arch}/seed_0/{loss}/ivon/lr_0.05/h0_1.0/ess_{ess}/post_10/beta2_1.0/beta_0.0_alpha_100.0_ivon",
    f"RESULTS/{data}/{arch}/seed_0/{loss}/ivon/lr_0.05/h0_0.5/ess_{ess}/post_10/beta2_1.0/beta_0.0_alpha_100.0_ivon",
    f"RESULTS/{data}/{arch}/seed_0/{loss}/ivon/lr_0.05/h0_0.1/ess_{ess}/post_10/beta2_1.0/beta_0.0_alpha_100.0_ivon",
   # f"RESULTS/{data}/{arch}/seed_0/{loss}/ivon/lr_0.05/h0_0.07/ess_{ess}/post_10/beta2_1.0/beta_0.0_alpha_100.0_ivon",
    f"RESULTS/{data}/{arch}/seed_0/{loss}/ivon/lr_0.05/h0_0.01/ess_{ess}/post_10/beta2_1.0/beta_0.0_alpha_100.0_ivon",
]

# Initialize lists to store the data
train_accuracies = []
test_accuracies = []
sharpnesses = []  # Change pre_sharpnesses to sharpnesses since we're reading eigs now

# Define labels based on the varied parameters
labels = [
    #r"$h_0=10.0$",
    r"$h_0=5.0$",
    r"$h_0=1.0$",
    r"$h_0=0.5$",
    r"$h_0=0.1$",
   # r"$h_0=0.07$",
    r"$h_0=0.01$"
]

# Try to load the data for each file
for path in base_paths:
    try:
        train_accuracies.append(torch.load(f"{path}/train_acc_final"))
        test_accuracies.append(torch.load(f"{path}/test_acc_final"))
        sharpnesses.append(torch.load(f"{path}/eigs_final")[:, 0])  # Loading eigs instead of pre-eigs
    except FileNotFoundError:
        print(f"Missing data for path: {path}")
        continue

# Prepare for plotting with three vertical subplots (Train Accuracy, Test Accuracy, Sharpness)
fig, axs = plt.subplots(3, 1, figsize=(18, 20), dpi=100)  # Adjust figsize to accommodate 3 plots

title_fontsize = 18
label_fontsize = 16
legend_fontsize = 24  # Smaller font size for the legend
tick_fontsize = 16  # Font size for tick labels

# Define distinct colors and markers for better differentiation
colors = [
    'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 
    'cyan', 'magenta', 'yellow', 'lime', 'indigo', 'teal', 'olive', 'navy'
]
markers = [
    'o', 's', 'D', '^', 'v', 'p', '*', 'h', 
    'X', '+', '|', '_', '<', '>', '1', '2'
]
linestyles = [
    '-', '--', '-.', ':', '-', '--', '-.', ':', 
    (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (5, 1))
]

# Plot training accuracy
for i, train_acc in enumerate(train_accuracies):
    axs[0].plot(train_acc[0:10000], label=f"{labels[i]}", 
                marker=markers[i % len(markers)], 
                color=colors[i % len(colors)], 
                linestyle=linestyles[i % len(linestyles)], 
                linewidth=2.5,  # Thicker lines
                markersize=5)  # Adjust marker size if needed
axs[0].set_title("Train Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
axs[0].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[0].grid(True)

# Plot test accuracy
for i, test_acc in enumerate(test_accuracies):
    axs[1].plot(test_acc[0:10000], label=f"{labels[i]}", 
                marker=markers[i % len(markers)], 
                color=colors[i % len(colors)], 
                linestyle=linestyles[i % len(linestyles)], 
                linewidth=2.5,  
                markersize=5)
axs[1].set_title("Test Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
#axs[1].set_ylim(0.35,0.45)  
axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[1].grid(True)

# Plot sharpness (eigenvalues)
for i, sharpness in enumerate(sharpnesses):
    if i==0 or i==2 or i==3:
        step=50
    else:
        step=50    
    axs[2].scatter(np.arange(len(sharpness[0:600])) * step, sharpness[0:600], 
                label=rf"{labels[i]}", 
                color=colors[i % len(colors)], 
                marker=markers[i % len(markers)], 
                s=60, edgecolors='black', linewidths=1, alpha=0.8)
axs[2].axhline(2. / 0.05, linestyle='dotted', label=r"$2/\eta$", color='black')
axs[2].set_title(r"Sharpness (Eigenvalues) $\|\nabla^2 L\|_{2}$", fontsize=title_fontsize, fontweight='bold')
axs[2].set_xlabel("Iteration", fontsize=label_fontsize)
axs[2].set_ylabel("Sharpness Value", fontsize=label_fontsize)
#axs[2].set_ylim(0, 6e1)  # Example y-axis limit
axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[2].grid(True)

# Move legends outside the plots for all subplots
for ax in axs:
    ax.legend(fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig(f"plot_scripts/plot_cifar_whole/{arch}_{loss}_{data}_{ess}_sharpness_post_{post}.png", bbox_inches='tight')
plt.show()
