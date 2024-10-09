# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# arch = "fc-tanh"
# loss = "mse"
# post = 10


# # Updated paths to data with different h0 and post values
# base_paths = [
#     f"RESULTS/cifar10-100/fc-tanh/seed_0/mse/gd/lr_0.01",
#     f"RESULTS/cifar10-200/fc-tanh/seed_0/mse/gd/lr_0.01",
#     #f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05/h0_1.0/ess_500000.0/post_{post}/beta2_1.0/beta_0.0_alpha_10.0_ivon",
#      f"RESULTS/cifar10-1k/fc-tanh/seed_0/mse/gd/lr_0.01"
# ]

# # Initialize lists to store the data
# train_accuracies = []
# test_accuracies = []
# sharpnesses = []

# # Define labels based on the varied parameters
# labels = [
#     r"$N=100$",
#     r"$N=200$",
#     r"$N=1000$"
# ]

# # Try to load the data for each file
# for path in base_paths:
#     try:
#         train_accuracies.append(torch.load(f"{path}/train_acc_final"))
#         test_accuracies.append(torch.load(f"{path}/test_acc_final"))
#         sharpnesses.append(torch.load(f"{path}/eigs_final")[:, 0])
#     except FileNotFoundError:
#         print(f"Missing data for path: {path}")
#         continue

# # Prepare for plotting with three vertical subplots (Train Accuracy, Test Accuracy, Sharpness)
# fig, axs = plt.subplots(1, 1, figsize=(10, 16), dpi=100)

# title_fontsize = 20
# label_fontsize = 24
# tick_fontsize = 24
# legend_fontsize = 28  # Larger font size for legends

# # Define distinct colors and markers for better differentiation
# colors = ['blue', 'orange','red']
# linestyles = ['-', '--',':']
# markers = ['o', 's','d']

# # Update style for lighter gray x/y ticks, axes, and grids
# light_gray = '#d3d3d3'  # Light gray color for axis ticks and labels
# dark_gray = '#a9a9a9'
# middle_gray = '#808080'
# gray_c = middle_gray
# # Set thicker plot lines
# linewidth = 6

# # # Plot training accuracy
# # for i, train_acc in enumerate(train_accuracies):
# #     axs[0].plot(train_acc, label=f"{labels[i]}", color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
# # axs[0].set_title("Train Accuracy", fontsize=title_fontsize, fontweight='bold')
# # axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
# # axs[0].set_ylabel("Accuracy", fontsize=label_fontsize)
# # axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors='black')
# # axs[0].spines['top'].set_color('none')  # Remove top spine
# # axs[0].spines['right'].set_color('none')  # Remove right spine
# # axs[0].spines['left'].set_color(gray_c)  # Make left spine light gray
# # axs[0].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
# # axs[0].grid(True)

# # # Position the legend in the bottom right corner for the first plot
# # axs[0].legend(fontsize=legend_fontsize, loc='lower right')

# # # Plot test accuracy
# # for i, test_acc in enumerate(test_accuracies):
# #     axs[1].plot(test_acc, label=f"{labels[i]}", color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
# # axs[1].set_title("Test Accuracy", fontsize=title_fontsize, fontweight='bold')
# # axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
# # axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
# # axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=gray_c)
# # #axs[1].set_ylim(0.25, 0.45)
# # axs[1].spines['top'].set_color('none')  # Remove top spine
# # axs[1].spines['right'].set_color('none')  # Remove right spine
# # axs[1].spines['left'].set_color(gray_c)  # Make left spine light gray
# # axs[1].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray
# # axs[1].grid(True)

# # # Position the legend in the bottom right corner for the second plot
# # axs[1].legend(fontsize=legend_fontsize, loc='lower right')

# # Plot sharpness (eigenvalues)

# ## step ratio is 4
# # Iterate over each sharpness array and plot the entire sharpness data
# for i, sharpness in enumerate(sharpnesses):
#     step = 50 if i == 0 else 50  # Keep the step size for consistency in plotting scale

#     # Plot the entire sharpness array without skipping any indices
#     axs[0].scatter(np.arange(len(sharpness)) * step, sharpness, 
#                    label=f"{labels[i]}", color=colors[i], 
#                    s=100, edgecolor='black', linewidth=1, alpha=0.8)

# # Plot the reference horizontal line for 2 / η
# axs[0].axhline(2. / 0.01, linestyle='dotted', label=r"$2/\eta$", color='black')

# # Set title and labels
# axs[0].set_title(r"Sharpness (Eigenvalues) $\|\nabla^2 L\|_{2}$", fontsize=title_fontsize, fontweight='bold')
# axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[0].set_ylabel("Sharpness", fontsize=label_fontsize)

# # Configure tick parameters and spine colors
# axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize, colors='black')
# axs[0].spines['top'].set_color('none')  # Remove top spine
# axs[0].spines['right'].set_color('none')  # Remove right spine
# axs[0].spines['left'].set_color(gray_c)  # Make left spine light gray
# axs[0].spines['bottom'].set_color(gray_c)  # Make bottom spine light gray

# # Enable the grid
# axs[0].grid(True)

# # Show the legend for the labels
# axs[0].legend(fontsize=legend_fontsize, loc='lower right')


# # Adjust layout
# plt.tight_layout()
# plt.savefig(f"iclr_sharpness_plot.png", bbox_inches='tight')
# plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt

arch = "fc-tanh"
loss = "mse"
post = 10

# Updated paths to data with different h0 and post values
base_paths = [
    f"RESULTS/cifar10-100/fc-tanh/seed_0/mse/gd/lr_0.01",
    f"RESULTS/cifar10-200/fc-tanh/seed_0/mse/gd/lr_0.01",
    f"RESULTS/cifar10-1k/fc-tanh/seed_0/mse/gd/lr_0.01"
]

# Initialize lists to store the data
sharpnesses = []

# Define labels based on the varied parameters
labels = [
    r"$N=100$",
    r"$N=200$",
    r"$N=1000$"
]

# Try to load the data for each file
for path in base_paths:
    try:
        sharpnesses.append(torch.load(f"{path}/eigs_final")[:, 0])
    except FileNotFoundError:
        print(f"Missing data for path: {path}")
        continue

# Create the figure and axis for sharpness plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

title_fontsize = 20
label_fontsize = 24
tick_fontsize = 24
legend_fontsize = 22  # Larger font size for legends

# Define distinct colors and markers for better differentiation
colors = ['blue', 'orange', 'red']

# Iterate over each sharpness array and plot the entire sharpness data
for i, sharpness in enumerate(sharpnesses):
    step = 50  # Consistent step size for the x-axis scale
    ax.scatter(np.arange(len(sharpness)) * step, sharpness, 
               label=f"{labels[i]}", color=colors[i], 
               s=100, edgecolor='black', linewidth=1, alpha=0.8)

# Plot the reference horizontal line for 2 / η
ax.axhline(2. / 0.01, linestyle='dotted', label=r"$2/\eta$", color='black')

# Set title and labels
ax.set_title(r"Sharpness (Eigenvalues) $\|\nabla^2 L\|_{2}$", fontsize=title_fontsize, fontweight='bold')
ax.set_xlabel("Epoch", fontsize=label_fontsize)
ax.set_ylabel("Sharpness", fontsize=label_fontsize)

# Configure tick parameters and spine colors
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, colors='black')
ax.spines['top'].set_color('none')  # Remove top spine
ax.spines['right'].set_color('none')  # Remove right spine
ax.spines['left'].set_color('#808080')  # Make left spine light gray
ax.spines['bottom'].set_color('#808080')  # Make bottom spine light gray

# Enable the grid
ax.grid(True)

# Show the legend for the labels
ax.legend(fontsize=legend_fontsize, loc='lower right')

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(f"iclr_sharpness_plot.png", bbox_inches='tight')
plt.show()

