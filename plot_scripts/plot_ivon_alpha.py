

# alpha plots for student-t distribution
import torch
import numpy as np
import matplotlib.pyplot as plt

arch = "fc-tanh"
loss = "ce"

# Updated paths to data with new alpha values
base_paths = [
   # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_1.0_ivon",
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_2.0_ivon",
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_2.2_ivon",
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_2.5_ivon",
    # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_2.7_ivon",
    # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_3.0_ivon",
    # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_4.0_ivon",
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_7.0_ivon",
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_100.0_ivon"
]

# Initialize lists to store the data
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
sharpnesses = []  # Change pre_sharpnesses to sharpnesses since we're reading eigs now

# Define labels based on the varied alpha values
labels = [
   # r"$\alpha=1.0$", 
    r"$\alpha=2.0$", 
    r"$\alpha=2.2$", 
    r"$\alpha=2.5$", 
    # r"$\alpha=2.7$", 
    # r"$\alpha=3.0$", 
    # r"$\alpha=4.0$", 
    r"$\alpha=7.0$", 
    r"$\alpha=100.0$"
]

# Try to load the data for each file
for path in base_paths:
    try:
        train_losses.append(torch.load(f"{path}/train_loss_final"))
        train_accuracies.append(torch.load(f"{path}/train_acc_final"))
        test_losses.append(torch.load(f"{path}/test_loss_final"))
        test_accuracies.append(torch.load(f"{path}/test_acc_final"))
        sharpnesses.append(torch.load(f"{path}/eigs_final")[:, 0])  # Loading eigs instead of pre-eigs
    except FileNotFoundError:
        print(f"Missing data for path: {path}")
        continue

# Prepare for plotting with five vertical subplots (Train Loss, Train Accuracy, Test Loss, Test Accuracy, Sharpness)
fig, axs = plt.subplots(3, 1, figsize=(14, 18), dpi=100)  # Adjust figsize to accommodate 5 plots

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
#     axs[0].plot(train_loss[0:5000], label=f"{labels[i]}", 
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

# Plot training accuracy
for i, train_acc in enumerate(train_accuracies):
    axs[0].plot(train_acc[0:5000], label=f"{labels[i]}", color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
axs[0].set_title("Train Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
axs[0].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
axs[0].spines['top'].set_color('none')  # Remove top spine
axs[0].spines['right'].set_color('none')  # Remove right spine
axs[0].spines['left'].set_color(gray_c)  # Make left spine light gray
axs[0].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
axs[0].set_ylim(0.2, 1.0)
axs[0].grid(True)

# # Plot test loss
# for i, test_loss in enumerate(test_losses):
#     axs[2].plot(test_loss[0:5000], label=f" {labels[i]}", 
#                 marker=markers[i % len(markers)], 
#                 color=colors[i % len(colors)], 
#                 linestyle=linestyles[i % len(linestyles)], 
#                 linewidth=2.5,  
#                 markersize=5)
# axs[2].set_title("Test Loss", fontsize=title_fontsize, fontweight='bold')
# axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[2].set_ylabel("Loss", fontsize=label_fontsize)
# axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axs[2].grid(True)

# Plot test accuracy
for i, test_acc in enumerate(test_accuracies):
    axs[1].plot(test_acc[0:5000], label=f"{labels[i]}", color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
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

# Plot sharpness (eigenvalues)
for i, sharpness in enumerate(sharpnesses):
    axs[2].scatter(np.arange(len(sharpness[0:100])) * 50, sharpness[0:100], 
                   label=f"{labels[i]}", 
                   color=colors[i % len(colors)], 
                   marker=markers[i % len(markers)], 
                   s=60, edgecolors='black', linewidths=1, alpha=0.5)
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

# Move legends outside the plots for all subplotss
for ax in axs:
    ax.legend(fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig(f"{arch}_{loss}_sharpness_alpha.png", bbox_inches='tight')
plt.show()


# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# arch = "fc-tanh"
# loss = "ce"

# # Updated paths to data with new alpha values
# base_paths = [
#     #f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_2.0_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_2.2_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_2.5_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_2.7_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_3.0_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_4.0_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_7.0_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_100.0_ivon"
# ]

# # Initialize lists to store the data
# train_losses = []
# train_accuracies = []
# test_losses = []
# test_accuracies = []
# sharpnesses = []  # Sharpness values

# # Define labels based on the varied alpha values
# labels = [
#     #r"$\alpha=2.0$", 
#     r"$\alpha=2.2$", 
#     # r"$\alpha=2.5$", 
#     # r"$\alpha=2.7$", 
#     # r"$\alpha=3.0$", 
#     # r"$\alpha=4.0$", 
#     # r"$\alpha=7.0$", 
#     r"$\alpha=100.0$"
# ]

# # Try to load the data for each file
# for path in base_paths:
#     try:
#         train_losses.append(torch.load(f"{path}/train_loss_final"))
#         train_accuracies.append(torch.load(f"{path}/train_acc_final"))
#         test_losses.append(torch.load(f"{path}/test_loss_final"))
#         test_accuracies.append(torch.load(f"{path}/test_acc_final"))
#         sharpnesses.append(torch.load(f"{path}/eigs_final")[:, 0])
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
# colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
# linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
# markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'h']

# # Update style for lighter gray x/y ticks, axes, and grids
# light_gray = '#d3d3d3'  # Light gray color for axis ticks and labels
# middle_gray = '#808080'
# gray_c = middle_gray

# # Set thicker plot lines
# linewidth = 3

# # Plot training accuracy
# for i, train_acc in enumerate(train_accuracies):
#     axs[0].plot(train_acc[0:5000], label=f"{labels[i]}", color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
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
#     axs[1].plot(test_acc[0:5000], label=f"{labels[i]}", color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
# axs[1].set_title("Test Accuracy", fontsize=title_fontsize, fontweight='bold')
# axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
# axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
# axs[1].set_ylim(0.35, 0.45)
# axs[1].spines['top'].set_color('none')  # Remove top spine
# axs[1].spines['right'].set_color('none')  # Remove right spine
# axs[1].spines['left'].set_color(gray_c)  # Make left spine light gray
# axs[1].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
# axs[1].grid(True)

# # Position the legend in the bottom right corner for the second plot
# axs[1].legend(fontsize=legend_fontsize, loc='upper right')

# # Plot sharpness (eigenvalues)
# for i, sharpness in enumerate(sharpnesses):
#     step = 50 if i == 0 else 50  
#     axs[2].scatter(np.arange(len(sharpness[0:100])) * step, sharpness[0:100], label=f"{labels[i]}", color=colors[i], s=100, edgecolor='black', linewidth=1, alpha=0.8)
# axs[2].axhline(2. / 0.05, linestyle='dotted', label=r"$2/\eta$", color='black')
# axs[2].set_title(r"Sharpness $\|\nabla^2 L\|_{2}$", fontsize=title_fontsize, fontweight='bold')
# axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[2].set_ylabel("Sharpness Value", fontsize=label_fontsize)
# axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
# axs[2].spines['top'].set_color('none')  # Remove top spine
# axs[2].spines['right'].set_color('none')  # Remove right spine
# axs[2].spines['left'].set_color(gray_c)  # Make left spine light gray
# axs[2].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
# axs[2].grid(True)

# # Add legend inside for the sharpness plot
# axs[2].legend(fontsize=legend_fontsize, loc='upper right')

# # Adjust layout
# plt.tight_layout()
# plt.savefig(f"{arch}_{loss}_sharpness_alpha_simplified.png", bbox_inches='tight')
# plt.show()
