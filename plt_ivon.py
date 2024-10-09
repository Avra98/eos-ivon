

### This part of code if for fixed covariance plots and sharpness
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# arch = "resnet32"
# loss = "ce"
# post=10

# # Updated paths to data with different h0 and post values based on the figure you provided
# base_paths = [
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.01_post_{post}_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.03_post_{post}_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.05_post_{post}_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.07_post_{post}_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.1_post_{post}_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_0.5_post_{post}_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_{post}_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_5.0_post_{post}_ivon",
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
#     r"$h_0=0.01$",
#     r"$h_0=0.03$",
#     r"$h_0=0.05$",
#     r"$h_0=0.07$",
#     r"$h_0=0.1$",
#     r"$h_0=0.5$",
#     r"$h_0=1.0$",
#     r"$h_0=5.0$",
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
# #axs[3].set_ylim(0.35,0.45)  
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
# plt.savefig(f"{arch}_{loss}_sharpness_post_{post}.png", bbox_inches='tight')
# plt.show()




# # alpha plots for student-t distribution
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# arch = "fc-tanh"
# loss = "ce"

# # Updated paths to data with new alpha values
# base_paths = [
#    # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_1.0_ivon",
#     f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_2.0_ivon",
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.05_h0_1.0_post_10_alpha_2.2_ivon",
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
# sharpnesses = []  # Change pre_sharpnesses to sharpnesses since we're reading eigs now

# # Define labels based on the varied alpha values
# labels = [
#    # r"$\alpha=1.0$", 
#     r"$\alpha=2.0$", 
#     # r"$\alpha=2.2$", 
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
#         sharpnesses.append(torch.load(f"{path}/eigs_final")[:, 0])  # Loading eigs instead of pre-eigs
#     except FileNotFoundError:
#         print(f"Missing data for path: {path}")
#         continue

# # Prepare for plotting with five vertical subplots (Train Loss, Train Accuracy, Test Loss, Test Accuracy, Sharpness)
# fig, axs = plt.subplots(5, 1, figsize=(18, 30), dpi=100)  # Adjust figsize to accommodate 5 plots

# title_fontsize = 20
# label_fontsize = 18
# legend_fontsize = 20  # Smaller font size for the legend
# tick_fontsize = 18  # Font size for tick labels

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

# # Plot training accuracy
# for i, train_acc in enumerate(train_accuracies):
#     axs[1].plot(train_acc[0:5000], label=f" {labels[i]}", 
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
#     axs[2].plot(test_loss[0:5000], label=f" {labels[i]}", 
#                 marker=markers[i % len(markers)], 
#                 color=colors[i % len(colors)], 
#                 linestyle=linestyles[i % len(linestyles)], 
#                 linewidth=2.5,  
#                 markersize=5)
# axs[2].set_title("Test Loss", fontsize=title_fontsize, fontweight='bold')
# axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[2].set_ylabel("Loss", fontsize=label_fontsize)
# #axs[2].set_ylim(1.5, 2.0) 
# axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axs[2].grid(True)

# # Plot test accuracy
# for i, test_acc in enumerate(test_accuracies):
#     axs[3].plot(test_acc[0:5000], label=f"{labels[i]}", 
#                 marker=markers[i % len(markers)], 
#                 color=colors[i % len(colors)], 
#                 linestyle=linestyles[i % len(linestyles)], 
#                 linewidth=2.5,  
#                 markersize=5)
# axs[3].set_title("Test Accuracy", fontsize=title_fontsize, fontweight='bold')
# axs[3].set_xlabel("Epoch", fontsize=label_fontsize)
# axs[3].set_ylabel("Accuracy", fontsize=label_fontsize)
# axs[3].set_ylim(0.40, 0.44) 
# axs[3].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axs[3].grid(True)

# # Plot sharpness (eigenvalues)
# for i, sharpness in enumerate(sharpnesses):
#     axs[4].scatter(np.arange(len(sharpness[0:100])) * 50, sharpness[0:100], 
#                    label=f"{labels[i]}", 
#                    color=colors[i % len(colors)], 
#                    marker=markers[i % len(markers)], 
#                    s=60, edgecolors='black', linewidths=1, alpha=0.5)
# axs[4].axhline(2. / 0.05, linestyle='dotted', label=r"$2/\eta$", color='black')
# axs[4].set_title(r"Sharpness $\| \nabla^2 L\|$", fontsize=title_fontsize, fontweight='bold')
# axs[4].set_xlabel("Iteration", fontsize=label_fontsize)
# axs[4].set_ylabel("Sharpness Value", fontsize=label_fontsize)
# axs[4].set_ylim(0, 50)  # Example y-axis limit
# axs[4].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# axs[4].grid(True)

# # Move legends outside the plots for all subplots
# for ax in axs:
#     ax.legend(fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

# plt.tight_layout()
# plt.savefig(f"{arch}_{loss}_sharpness_alpha_main.png", bbox_inches='tight')
# plt.show()








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
#     # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005_h0_1.0_post_{post}_alpha_100.0_beta2_0.999_beta_0.0_ess_1000000.0_ivon",
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
    # f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005/h0_1.0/ess_50000000.0/post_{post}/beta2_0.999/beta_0.0_alpha_100.0_ivon",
    f"RESULTS/cifar10-10k/{arch}/seed_0/{loss}/ivon/lr_0.0005/h0_1.0/ess_100000000.0/post_{post}/beta2_0.999/beta_0.0_alpha_100.0_ivon"
]

# Initialize lists to store the data
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
pre_sharpnesses = []

# Define labels based on the varied parameters (\lambda for LaTeX form)
labels = [
    "ADAM",
    r"IVON $\lambda=1e6$", 
    r"IVON $\lambda=2e6$",
    # r"IVON $\lambda=3e6$",
    # r"IVON $\lambda=5e6$",
    r"IVON $\lambda=1e7$", 
    # r"IVON $\lambda=2e7$", 
    # r"IVON $\lambda=3e7$", 
    # r"IVON $\lambda=5e7$", 
    r"IVON $\lambda=1e8$" 
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
fig, axs = plt.subplots(5, 1, figsize=(18, 30), dpi=100)

title_fontsize = 20
label_fontsize = 18
legend_fontsize = 20  # Smaller font size for the legend
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

# Plot training loss
for i, train_loss in enumerate(train_losses):
    axs[0].plot(train_loss[0:30000], label=f"{labels[i]}", marker=markers[i % len(markers)], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=4, markersize=1)
axs[0].set_title("Train Loss", fontsize=title_fontsize,fontweight='bold')
axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
axs[0].set_ylabel("Loss", fontsize=label_fontsize)
axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[0].grid(True)

# Plot training accuracy
for i, train_acc in enumerate(train_accuracies):
    axs[1].plot(train_acc[0:30000], label=f" {labels[i]}", marker=markers[i % len(markers)], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=4, markersize=1)
axs[1].set_title("Train Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[1].grid(True)

# Plot test loss
for i, test_loss in enumerate(test_losses):
    axs[2].plot(test_loss[0:30000], label=f"{labels[i]}", marker=markers[i % len(markers)], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=4, markersize=1)
axs[2].set_title("Test Loss", fontsize=title_fontsize, fontweight='bold')
axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
axs[2].set_ylabel("Loss", fontsize=label_fontsize)
axs[2].set_ylim(0.38, 0.6) 
axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[2].grid(True)

# Plot test accuracy
for i, test_acc in enumerate(test_accuracies):
    axs[3].plot(test_acc[0:30000], label=f"{labels[i]}", marker=markers[i % len(markers)], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=4, markersize=1)
axs[3].set_title("Test Accuracy", fontsize=title_fontsize, fontweight='bold')
axs[3].set_xlabel("Epoch", fontsize=label_fontsize)
axs[3].set_ylabel("Accuracy", fontsize=label_fontsize)
axs[3].set_ylim(0.3, 0.5) 
axs[3].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[3].grid(True)

# Plot pre-sharpness (pre-eigenvalues)
for i, pre_sharpness in enumerate(pre_sharpnesses):
    step = 200 if i == 0 else 50
    axs[4].scatter(np.arange(len(pre_sharpness[0:600])) * step, pre_sharpness[0:600], label=rf" {labels[i]}", color=colors[i % len(colors)], marker=markers[i % len(markers)], s=60, edgecolors='black', linewidths=1, alpha=0.8)
axs[4].axhline(2. / 0.0005, linestyle='dotted', label=r"$2/\eta$", color='black')
axs[4].set_title(r"Pre-Conditioned-Sharpness $\|\mathbf{P(t)}^{-1} \nabla^2 L (t)\|$", fontsize=title_fontsize, fontweight='bold')
axs[4].set_xlabel("Epoch", fontsize=label_fontsize)
axs[4].set_ylabel("Sharpness Value", fontsize=label_fontsize)
axs[4].set_ylim(0, 6e3)  # Example y-axis limit
axs[4].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[4].grid(True)

# Move legends outside the plots for all subplots
for ax in axs:
    ax.legend(fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig(f"{arch}_{loss}_post_{post}_pre-eigs_h1.0.png", bbox_inches='tight')
plt.show()
