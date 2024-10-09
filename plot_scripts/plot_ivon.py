

### Pre-conditioning plots (final)
import torch
import numpy as np
import matplotlib.pyplot as plt

arch = "fc-tanh"
loss = "mse"
post = 50


# base_paths = [
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/adam/lr_0.0005_adam_beta_0.0_beta2_0.999",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005_h0_1.0_post_{post}_alpha_100.0_beta2_0.999_beta_0.0_ess_100000.0_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005_h0_1.0_post_{post}_alpha_100.0_beta2_0.999_beta_0.0_ess_500000.0_ivon",
#     #f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005_h0_1.0_post_{post}_alpha_100.0_beta2_0.999_beta_0.0_ess_1000000.0_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005_h0_1.0_post_{post}_alpha_100.0_beta2_0.999_beta_0.0_ess_2000000.0_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005_h0_1.0_post_{post}_alpha_100.0_beta2_0.999_beta_0.0_ess_3000000.0_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005_h0_1.0_post_{post}_alpha_100.0_beta2_0.999_beta_0.0_ess_5000000.0_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005_h0_1.0_post_{post}_alpha_100.0_beta2_0.999_beta_0.0_ess_10000000.0_ivon"
# ]
# Explicit paths for each combination of h0, beta2, and ess
base_paths = [
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/adam/lr_0.0005_adam_beta_0.0_beta2_0.999",
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005/h0_1.0/ess_1000000.0/post_{post}/beta2_0.999/beta_0.0_alpha_100.0_ivon",
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005/h0_1.0/ess_2000000.0/post_{post}/beta2_0.999/beta_0.0_alpha_100.0_ivon",
    # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005/h0_1.0/ess_3000000.0/post_{post}/beta2_0.999/beta_0.0_alpha_100.0_ivon",
    # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005/h0_1.0/ess_5000000.0/post_{post}/beta2_0.999/beta_0.0_alpha_100.0_ivon",
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005/h0_1.0/ess_10000000.0/post_{post}/beta2_0.999/beta_0.0_alpha_100.0_ivon",
    # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005/h0_1.0/ess_20000000.0/post_{post}/beta2_0.999/beta_0.0_alpha_100.0_ivon",
    # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005/h0_1.0/ess_30000000.0/post_{post}/beta2_0.999/beta_0.0_alpha_100.0_ivon",
    #f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005/h0_1.0/ess_50000000.0/post_{post}/beta2_0.999/beta_0.0_alpha_100.0_ivon",
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005/h0_1.0/ess_100000000.0/post_{post}/beta2_0.999/beta_0.0_alpha_100.0_ivon"
]

# Initialize lists to store the data
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
pre_sharpnesses = []

# Define labels based on the varied parameters (\lambda for LaTeX form)
# labels = [
#     "ADAM",
#     r"IVON $\lambda=1e6$", 
#     r"IVON $\lambda=2e6$",
#     # r"IVON $\lambda=3e6$",
#     # r"IVON $\lambda=5e6$",
#     r"IVON $\lambda=1e7$" ,
#     # r"$\lambda=2e7$", 
#     # r"$\lambda=3e7$", 
#     # r"IVON $\lambda=5e7$", 
#     r"IVON $\lambda=1e8$"
# ]

labels = [
    "ADAM",
    r"IVON $\tau=0.5$", 
    r"IVON $\tau=0.25$",
    # r"IVON $\lambda=3e6$",
    # r"IVON $\lambda=5e6$",
    r"IVON $\tau=0.05$" ,
    # r"$\lambda=2e7$", 
    # r"$\lambda=3e7$", 
    # r"IVON $\lambda=5e7$", 
    r"IVON $\tau=0.005$"
]
# Try to load the data for each file
for path in base_paths:
    try:
        train_losses.append(torch.load(f"{path}/train_loss_final"))
        train_accuracies.append(torch.load(f"{path}/train_acc_final"))
        test_losses.append(torch.load(f"{path}/test_loss_final"))
        test_accuracies.append(torch.load(f"{path}/test_acc_final"))
        pre_sharpnesses.append(torch.load(f"{path}/pre-eigs_final")[:, 0])
    except FileNotFoundError:
        print(f"Missing data for path: {path}")
        continue

# Prepare for plotting with five vertical subplots (Train Loss, Train Accuracy, Test Loss, Test Accuracy, Pre-Sharpness)
fig, axs = plt.subplots(3, 1, figsize=(14, 18), dpi=100)

title_fontsize = 20
label_fontsize = 24
legend_fontsize = 22  # Smaller font size for the legend
tick_fontsize = 24  # Font size for tick labels

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

# Update style for lighter gray x/y ticks, axes, and grids
light_gray = '#d3d3d3'  # Light gray color for axis ticks and labels
dark_gray = '#a9a9a9'
middle_gray = '#808080'
gray_c = middle_gray
# Set thicker plot lines
linewidth = 3


# # Plot training loss
# for i, train_loss in enumerate(train_losses):
#     axs[0].plot(train_loss[0:30000], label=f"{labels[i]}", marker=markers[i % len(markers)], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=4, markersize=1)
# axs[0].set_title("Train Loss", fontsize=title_fontsize,fontweight='bold')
# axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[0].set_ylabel("Loss", fontsize=label_fontsize)
# axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axs[0].grid(True)

# Plot training accuracy
for i, train_acc in enumerate(train_accuracies):
    axs[0].plot(train_acc[0:30000], label=f" {labels[i]}", marker=markers[i % len(markers)], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=4, markersize=1)
axs[0].set_title("Train Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
axs[0].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
axs[0].spines['top'].set_color('none')  # Remove top spine
axs[0].spines['right'].set_color('none')  # Remove right spine
axs[0].spines['left'].set_color(gray_c)  # Make left spine light gray
axs[0].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
axs[0].grid(True)
axs[0].legend(fontsize=legend_fontsize, loc='lower right')
# # Plot test loss
# for i, test_loss in enumerate(test_losses):
#     axs[2].plot(test_loss[0:30000], label=f"{labels[i]}", marker=markers[i % len(markers)], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=4, markersize=1)
# axs[2].set_title("Test Loss", fontsize=title_fontsize, fontweight='bold')
# axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[2].set_ylabel("Loss", fontsize=label_fontsize)
# axs[2].set_ylim(0.38, 0.6) 
# axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axs[2].grid(True)

# Plot test accuracy
for i, test_acc in enumerate(test_accuracies):
    axs[1].plot(test_acc[0:30000], label=f"{labels[i]}", marker=markers[i % len(markers)], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=4, markersize=1)
axs[1].set_title("Test Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[1].set_ylim(0.3, 0.5) 
axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
axs[1].spines['top'].set_color('none')  # Remove top spine
axs[1].spines['right'].set_color('none')  # Remove right spine
axs[1].spines['left'].set_color(gray_c)  # Make left spine light gray
axs[1].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
axs[1].grid(True)
axs[1].legend(fontsize=legend_fontsize, loc='upper left')

# Plot pre-sharpness (pre-eigenvalues)
for i, pre_sharpness in enumerate(pre_sharpnesses):
    step = 200 if i == 0 else 50
    axs[2].scatter(np.arange(len(pre_sharpness[0:600])) * step, pre_sharpness[0:600], label=rf" {labels[i]}", color=colors[i % len(colors)], marker=markers[i % len(markers)], s=60, edgecolors='black', linewidths=1, alpha=0.8)
axs[2].axhline(2. / 0.0005, linestyle='dotted', label=r"$2/\eta$", color='black')
axs[2].set_title(r"Pre-Conditioned-Sharpness $\|\mathbf{P(t)}^{-1} \nabla^2 L (t)\|$", fontsize=title_fontsize, fontweight='bold')
axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
axs[2].set_ylabel("Sharpness Value", fontsize=label_fontsize)
axs[2].set_ylim(0, 6e3)  # Example y-axis limit
axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
axs[2].spines['top'].set_color('none')  # Remove top spine
axs[2].spines['right'].set_color('none')  # Remove right spine
axs[2].spines['left'].set_color(gray_c)  # Make left spine light gray
axs[2].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
axs[2].grid(True)
axs[2].legend(fontsize=legend_fontsize, loc='center left')

# # Move legends outside the plots for all subplots
# for ax in axs:
#     ax.legend(fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig(f"{arch}_{loss}_post_{post}_pre-eigs_h1.0.png", bbox_inches='tight')
plt.show()


# ### Pre-conditioning plots (final)
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# arch = "fc-tanh"
# loss = "mse"
# post = 1

# # Paths to data
# base_paths = [
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/adam/lr_0.0005_adam_beta_0.0_beta2_0.999",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005_h0_1.0_post_{post}_alpha_100.0_beta2_0.999_beta_0.0_ess_2000000.0_ivon",
# ]

# # Initialize lists to store the data
# train_accuracies = []
# test_accuracies = []
# pre_sharpnesses = []

# # Define labels based on the varied parameters (\lambda for LaTeX form)
# labels = [
#     "ADAM",
#     r"IVON $\tau=0.25$",
# ]

# # Try to load the data for each file
# for path in base_paths:
#     try:
#         train_accuracies.append(torch.load(f"{path}/train_acc_final"))
#         test_accuracies.append(torch.load(f"{path}/test_acc_final"))
#         pre_sharpnesses.append(torch.load(f"{path}/pre-eigs_final")[:, 0])
#     except FileNotFoundError:
#         print(f"Missing data for path: {path}")
#         continue

# # Prepare for plotting with three vertical subplots (Train Accuracy, Test Accuracy, Sharpness)
# fig, axs = plt.subplots(3, 1, figsize=(14, 18), dpi=100)

# title_fontsize = 20
# label_fontsize = 22
# tick_fontsize = 20
# legend_fontsize = 24  # Larger font size for legends

# # Define distinct colors and markers for better differentiation
# colors = ['blue', 'orange']
# linestyles = ['-', '--']
# markers = ['o', 's']

# # Update style for lighter gray x/y ticks, axes, and grids
# light_gray = '#d3d3d3'  # Light gray color for axis ticks and labels
# dark_gray = '#a9a9a9'
# middle_gray = '#808080'
# gray_c = middle_gray
# # Set thicker plot lines
# linewidth = 3

# # Plot training accuracy
# for i, train_acc in enumerate(train_accuracies):
#     axs[0].plot(train_acc, label=f"{labels[i]}", color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
# axs[0].set_title("Train Accuracy", fontsize=title_fontsize, fontweight='bold')
# axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[0].set_ylabel("Accuracy", fontsize=label_fontsize)
# axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
# axs[0].spines['top'].set_color('none')  # Remove top spine
# axs[0].spines['right'].set_color('none')  # Remove right spine
# axs[0].spines['left'].set_color(gray_c)  # Make left spine light gray
# axs[0].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
# axs[0].grid(True)

# # Position the legend in the bottom right corner for the first plot
# axs[0].legend(fontsize=legend_fontsize, loc='lower right')

# # Plot test accuracy
# for i, test_acc in enumerate(test_accuracies):
#     axs[1].plot(test_acc, label=f"{labels[i]}", color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
# axs[1].set_title("Test Accuracy", fontsize=title_fontsize, fontweight='bold')
# axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
# axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
# axs[1].set_ylim(0.3, 0.45)
# axs[1].spines['top'].set_color('none')  # Remove top spine
# axs[1].spines['right'].set_color('none')  # Remove right spine
# axs[1].spines['left'].set_color(gray_c)  # Make left spine light gray
# axs[1].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
# axs[1].grid(True)

# # Position the legend in the bottom right corner for the second plot
# axs[1].legend(fontsize=legend_fontsize, loc='best')

# # Plot sharpness (eigenvalues)
# for i, sharpness in enumerate(pre_sharpnesses):
#     step = 200 if i == 0 else 50 
#     axs[2].scatter(np.arange(len(sharpness)) * step, sharpness, label=f"{labels[i]}", color=colors[i], s=100, edgecolor='black', linewidth=1, alpha=0.8)
# axs[2].axhline(2. / 0.0005, linestyle='dotted', label=r"$2/\eta$", color='black')
# axs[2].set_title(r"Pre-Conditioned-Sharpness $\|\mathbf{P(t)}^{-1} \nabla^2 L (t)\|_{2}$", fontsize=title_fontsize, fontweight='bold')
# axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[2].set_ylabel("Sharpness", fontsize=label_fontsize)
# axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
# axs[2].spines['top'].set_color('none')  # Remove top spine
# axs[2].spines['right'].set_color('none')  # Remove right spine
# axs[2].spines['left'].set_color(gray_c)  # Make left spine light gray
# axs[2].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
# axs[2].grid(True)

# # Add legend inside for the sharpness plot
# axs[2].legend(fontsize=legend_fontsize, loc='center left')

# # Adjust layout
# plt.tight_layout()
# plt.savefig(f"{arch}_{loss}_post_{post}_pre-eigs_h1.0_main_simplified.png", bbox_inches='tight')
# plt.show()
