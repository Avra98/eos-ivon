
### This part of code if for fixed covariance plots and sharpness
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# arch = "fc-tanh"
# loss = "ce"
# post=10

# # Updated paths to data with different h0 and post values based on the figure you provided
# base_paths = [
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.01_post_{post}_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.03_post_{post}_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.05_post_{post}_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.07_post_{post}_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.1_post_{post}_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_{post}_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_5.0_post_{post}_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_10.0_post_{post}_ivon"
# ]

# # Initialize lists to store the data
# train_losses = []
# train_accuracies = []
# test_losses = []
# test_accuracies = []
# sharpnesses = []  # Change pre_sharpnesses to sharpnesses since we're reading eigs now

# # Define labels based on the varied parameters
# labels = [
#     # r"$h_0=0.01$",
#     # r"$h_0=0.03$",
#     # r"$h_0=0.05$",
#     r"$h_0=0.07$",
#     # r"$h_0=0.1$",
#     # r"$h_0=1.0$",
#     # r"$h_0=5.0$",
#     r"$h_0=10.0$"
# ]

# # Try to load the data for each file
# for path in base_paths:
#     try:
#         train_losses.append(torch.load(f"{path}/train_loss_final"))
#         train_accuracies.append(torch.load(f"{path}/train_acc_final"))
#         test_losses.append(torch.load(f"{path}/test_loss_final"))
#         test_accuracies.append(torch.load(f"{path}/test_acc_final"))
#         sharpnesses.append(torch.load(f"{path}/eigs_final")[:, 0])  # Loading eigs instead of pre-eigs
#     except FileNotFoundError:
#         print(f"Missing data for path: {path}")
#         continue

# # Prepare for plotting with five vertical subplots (Train Loss, Train Accuracy, Test Loss, Test Accuracy, Sharpness)
# fig, axs = plt.subplots(5, 1, figsize=(18, 30), dpi=100)  # Adjust figsize to accommodate 5 plots

# title_fontsize = 18
# label_fontsize = 16
# legend_fontsize = 24  # Smaller font size for the legend
# tick_fontsize = 16  # Font size for tick labels

# # Define distinct colors and markers for better differentiation
# colors = [
#     'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 
#     'cyan', 'magenta', 'yellow', 'lime', 'indigo', 'teal', 'olive', 'navy'
# ]
# markers = [
#     'o', 's', 'D', '^', 'v', 'p', '*', 'h', 
#     'X', '+', '|', '_', '<', '>', '1', '2'
# ]
# linestyles = [
#     '-', '--', '-.', ':', '-', '--', '-.', ':', 
#     (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (5, 1))
# ]

# # Plot training loss
# for i, train_loss in enumerate(train_losses):
#     axs[0].plot(train_loss[0:10000], label=f"{labels[i]}", 
#                 marker=markers[i % len(markers)], 
#                 color=colors[i % len(colors)], 
#                 linestyle=linestyles[i % len(linestyles)], 
#                 linewidth=2.5,  # Thicker lines
#                 markersize=5)  # Adjust marker size if needed
# axs[0].set_title("Train Loss", fontsize=title_fontsize, fontweight='bold')
# axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[0].set_ylabel("Loss", fontsize=label_fontsize)
# axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axs[0].grid(True)

# # Plot training accuracy
# for i, train_acc in enumerate(train_accuracies):
#     axs[1].plot(train_acc[0:10000], label=f"{labels[i]}", 
#                 marker=markers[i % len(markers)], 
#                 color=colors[i % len(colors)], 
#                 linestyle=linestyles[i % len(linestyles)], 
#                 linewidth=2.5,  # Thicker lines
#                 markersize=5)  # Adjust marker size if needed
# axs[1].set_title("Train Accuracy", fontsize=title_fontsize, fontweight='bold')
# axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
# axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axs[1].grid(True)

# # Plot test loss
# for i, test_loss in enumerate(test_losses):
#     axs[2].plot(test_loss[0:10000], label=f"{labels[i]}", 
#                 marker=markers[i % len(markers)], 
#                 color=colors[i % len(colors)], 
#                 linestyle=linestyles[i % len(linestyles)], 
#                 linewidth=2.5,  
#                 markersize=5)
# axs[2].set_title("Test Loss", fontsize=title_fontsize, fontweight='bold')
# axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[2].set_ylabel("Loss", fontsize=label_fontsize)
# #axs[2].set_ylim(1.6, 1.7)  
# axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axs[2].grid(True)

# # Plot test accuracy
# for i, test_acc in enumerate(test_accuracies):
#     axs[3].plot(test_acc[0:10000], label=f" {labels[i]}", 
#                 marker=markers[i % len(markers)], 
#                 color=colors[i % len(colors)], 
#                 linestyle=linestyles[i % len(linestyles)], 
#                 linewidth=2.5,  
#                 markersize=5)
# axs[3].set_title("Test Accuracy", fontsize=title_fontsize, fontweight='bold')
# axs[3].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[3].set_ylabel("Accuracy", fontsize=label_fontsize)
# axs[3].set_ylim(0.35,0.45)  
# axs[3].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axs[3].grid(True)

# # Plot sharpness (eigenvalues)
# for i, sharpness in enumerate(sharpnesses):
#     if i==0 or i==2 or i==3:
#         step=50
#     else:
#         step=19    
#     axs[4].scatter(np.arange(len(sharpness[0:600])) * step, sharpness[0:600], 
#                 label=rf"{labels[i]}", 
#                 color=colors[i % len(colors)], 
#                 marker=markers[i % len(markers)], 
#                 s=60, edgecolors='black', linewidths=1, alpha=0.8)
# axs[4].axhline(2. / 0.05, linestyle='dotted', label=r"$2/\eta$", color='black')
# axs[4].set_title(r"Sharpness (Eigenvalues) $\|\nabla^2 L\|_{2}$", fontsize=title_fontsize, fontweight='bold')
# axs[4].set_xlabel("Iteration", fontsize=label_fontsize)
# axs[4].set_ylabel("Sharpness Value", fontsize=label_fontsize)
# axs[4].set_ylim(0, 6e1)  # Example y-axis limit
# axs[4].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axs[4].grid(True)

# # Move legends outside the plots for all subplots
# for ax in axs:
#     ax.legend(fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

# plt.tight_layout()
# plt.savefig(f"{arch}_{loss}_sharpness_post_{post}_main.png", bbox_inches='tight')
# plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt

arch = "fc-tanh"
loss = "ce"
post = 10


# Updated paths to data with different h0 and post values
base_paths = [
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.07_post_{post}_ivon",
    #f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.1_post_{post}_ivon",
    "RESULTS/cifar10-10k/fc-tanh/seed_0/ce/gd/lr_0.05"

]

# Initialize lists to store the data
train_accuracies = []
test_accuracies = []
sharpnesses = []

# Define labels based on the varied parameters
labels = [
    r"IVON ($h_0=0.07)$",
    #r"$h_0=0.1$",
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
label_fontsize = 30
tick_fontsize = 28
legend_fontsize = 25  # Larger font size for legends

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
linewidth = 4

# Plot training accuracy
for i, train_acc in enumerate(train_accuracies):
    axs[0].plot(train_acc, label=f"{labels[i]}", color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
axs[0].set_title("Train Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
axs[0].set_ylabel("Accuracy", fontsize=label_fontsize)
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
axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
axs[1].set_ylim(0.35, 0.45)
axs[1].spines['top'].set_color('none')  # Remove top spine
axs[1].spines['right'].set_color('none')  # Remove right spine
axs[1].spines['left'].set_color(gray_c)  # Make left spine light gray
axs[1].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
axs[1].grid(True)

# Position the legend in the bottom right corner for the second plot
axs[1].legend(fontsize=legend_fontsize, loc='lower right')

# Plot sharpness (eigenvalues)
for i, sharpness in enumerate(sharpnesses):
    step = 50 if i == 0 else 50  
    axs[2].scatter(np.arange(len(sharpness)) * step, sharpness, label=f"{labels[i]}", color=colors[i], s=100, edgecolor='black', linewidth=1, alpha=0.8)
axs[2].axhline(2. / 0.05, linestyle='dotted', label=r"$2/\eta$", color='black')
axs[2].set_title(r"Sharpness (Eigenvalues) $\|\nabla^2 L\|_{2}$", fontsize=title_fontsize, fontweight='bold')
axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
axs[2].set_ylabel("Sharpness", fontsize=label_fontsize)
axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
axs[2].spines['top'].set_color('none')  # Remove top spine
axs[2].spines['right'].set_color('none')  # Remove right spine
axs[2].spines['left'].set_color(gray_c)  # Make left spine light gray
axs[2].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
axs[2].grid(True)

# Add legend inside for the sharpness plot
axs[2].legend(fontsize=legend_fontsize, loc='lower right')

# Adjust layout
plt.tight_layout()
plt.savefig(f"{arch}_{loss}_sharpness_post_{post}_simplified.png", bbox_inches='tight')
plt.show()

