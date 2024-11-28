# import re
# import matplotlib.pyplot as plt
# import numpy as np

# def parse_log_file(file_path):
#     with open(file_path, 'r', encoding="utf8") as file:
#         log_data = file.readlines()

#     pattern = re.compile(r"(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
#     eigen_pattern = re.compile(r"eigenvalues:\s*tensor\(\[([\d.]+)\]\)")

#     epochs = []
#     train_loss = []
#     train_acc = []
#     val_loss = []
#     val_acc = []
#     hessian_eigenvalues = []

#     for line in log_data:
#         match = pattern.match(line)
#         if match:
#             epoch = int(match.group(1))
#             t_loss = float(match.group(2))
#             t_acc = float(match.group(3))
#             v_loss = float(match.group(4))
#             v_acc = float(match.group(5))

#             epochs.append(epoch)
#             train_loss.append(t_loss)
#             train_acc.append(t_acc)
#             val_loss.append(v_loss)
#             val_acc.append(v_acc)

#         eigen_match = eigen_pattern.search(line)
#         if eigen_match:
#             eigenvalue = float(eigen_match.group(1))
#             hessian_eigenvalues.append(eigenvalue)

#     return epochs, train_loss, train_acc, val_loss, val_acc, hessian_eigenvalues


# def plot_logs(log_files, labels):
#     fig, axs = plt.subplots(3, 1, figsize=(14, 18), dpi=100)
    
#     title_fontsize = 20
#     label_fontsize = 18
#     legend_fontsize = 20
#     tick_fontsize = 16

#     colors = ['blue', 'orange']
#     markers = ['o', 's']
#     linestyles = ['-', '--']
    
#     for i, (log_file, label) in enumerate(zip(log_files, labels)):
#         epochs, train_loss, train_acc, val_loss, val_acc, hessian_eigenvalues = parse_log_file(log_file)
        
#         # Plot Train Loss
#         axs[0].plot(epochs, train_loss, label=f"{label} Train Loss", color=colors[i % len(colors)], marker=markers[i % len(markers)], linestyle=linestyles[0], linewidth=4, markersize=1)
#         axs[0].plot(epochs, val_loss, label=f"{label} Val Loss", color=colors[i % len(colors)], marker=markers[i % len(markers)], linestyle=linestyles[1], linewidth=4, markersize=1)
#         axs[0].set_title("Training and Validation Loss", fontsize=title_fontsize, fontweight='bold')
#         axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
#         axs[0].set_ylabel("Loss", fontsize=label_fontsize)
#         axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
#         axs[0].grid(True)

#         # Plot Train Accuracy
#         axs[1].plot(epochs, train_acc, label=f"{label} Train Accuracy", color=colors[i % len(colors)], marker=markers[i % len(markers)], linestyle=linestyles[0], linewidth=4, markersize=1)
#         axs[1].plot(epochs, val_acc, label=f"{label} Val Accuracy", color=colors[i % len(colors)], marker=markers[i % len(markers)], linestyle=linestyles[1], linewidth=4, markersize=1)
#         axs[1].set_title("Training and Validation Accuracy", fontsize=title_fontsize, fontweight='bold')
#         axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
#         axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
#         axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
#         axs[1].grid(True)

#         # Plot Sharpness (Hessian Eigenvalues)
#         if hessian_eigenvalues:
#             axs[2].plot(np.arange(len(hessian_eigenvalues)) * 100, hessian_eigenvalues, label=f"{label} Sharpness", color=colors[i % len(colors)], marker=markers[i % len(markers)], linestyle=linestyles[0], linewidth=4, markersize=1)
#         axs[2].axhline(y=400, color='black', linestyle='--', label='2/lr')
#         axs[2].set_title("Sharpness", fontsize=title_fontsize, fontweight='bold')
#         axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
#         axs[2].set_ylabel("Sharpness Value", fontsize=label_fontsize)
#         axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
#         axs[2].grid(True)
    
#     # Move legends outside the plots for all subplots
#     for ax in axs:
#         ax.legend(fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

#     plt.tight_layout()
#     plt.savefig('vit/train_val_loss_acc_eigen.png', bbox_inches='tight')
#     plt.show()

# # Example usage
# log_files = ['vit/gd-lr0.005-cifar50k.out', 'vit/ivon-lr0.005-cifar50k.out']
# labels = ['GD', 'IVON']
# plot_logs(log_files, labels)
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        log_data = file.readlines()

    pattern = re.compile(r"(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    eigen_pattern = re.compile(r"eigenvalues:\s*tensor\(\[([\d.]+)\]\)")

    epochs = []
    train_acc = []
    val_acc = []
    hessian_eigenvalues = []

    for line in log_data:
        match = pattern.match(line)
        if match:
            epoch = int(match.group(1))
            t_acc = float(match.group(3))
            v_acc = float(match.group(5))

            epochs.append(epoch)
            train_acc.append(t_acc)
            val_acc.append(v_acc)

        eigen_match = eigen_pattern.search(line)
        if eigen_match:
            eigenvalue = float(eigen_match.group(1))
            hessian_eigenvalues.append(eigenvalue)

    return epochs, train_acc, val_acc, hessian_eigenvalues


def plot_logs(log_files, labels):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), dpi=100)

    title_fontsize = 18
    label_fontsize = 16
    legend_fontsize = 20
    tick_fontsize = 16

    colors = ['blue', 'orange']
    markers = ['o', 's']
    linestyles = ['-', '--']

    for i, (log_file, label) in enumerate(zip(log_files, labels)):
        epochs, train_acc, val_acc, hessian_eigenvalues = parse_log_file(log_file)

        # Plot Training Accuracy
        axs[0].plot(epochs, train_acc, label=f"{label}", color=colors[i % len(colors)], marker=markers[i % len(markers)], linestyle=linestyles[0], linewidth=2, markersize=2)
        axs[0].set_title("Training Accuracy", fontsize=title_fontsize)
        axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
        axs[0].set_ylabel("Accuracy", fontsize=label_fontsize)
        axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[0].legend(fontsize=legend_fontsize, loc='lower right')
        axs[0].grid(True)

        # Plot Validation Accuracy
        axs[1].plot(epochs, val_acc, label=f"{label}", color=colors[i % len(colors)], marker=markers[i % len(markers)], linestyle=linestyles[0], linewidth=2, markersize=2)
        axs[1].set_title("Validation Accuracy", fontsize=title_fontsize)
        axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
        axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
        axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[1].legend(fontsize=legend_fontsize, loc='lower right')
        axs[1].grid(True)

        # Plot Sharpness (Hessian Eigenvalues)
        if hessian_eigenvalues:
            axs[2].plot(np.arange(len(hessian_eigenvalues)) * 100, hessian_eigenvalues, label=f"{label}", color=colors[i % len(colors)], marker=markers[i % len(markers)], linestyle=linestyles[0], linewidth=2, markersize=2)
        if i==0:    
            axs[2].axhline(y=400, color='black', linestyle='--', label='2/lr')
        axs[2].set_title("Sharpness", fontsize=title_fontsize)
        axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
        axs[2].set_ylabel("Sharpness Value", fontsize=label_fontsize)
        axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[2].legend(fontsize=legend_fontsize, loc='upper left')
        axs[2].grid(True)

    # Move legends outside the plots for all subplots
    # for ax in axs:
    #     ax.legend(fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

    # Add overall legend for GD and IVON outside the figure
    #fig.legend(['GD', 'IVON'], loc='upper right')

    plt.tight_layout()
    plt.savefig('vit/train_val_acc_eigen.png', bbox_inches='tight')
    plt.show()

# Example usage
log_files = ['vit/gd-lr0.005-cifar50k.out', 'vit/ivon-lr0.005-cifar50k.out']
labels = ['GD', 'IVON']
plot_logs(log_files, labels)