import re
import matplotlib.pyplot as plt
import numpy as np
import torch


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

    # Resample data to every 50 epochs (keep every 5th entry)
    epochs = epochs[::5]
    train_acc = train_acc[::5]
    val_acc = val_acc[::5]
    hessian_eigenvalues = hessian_eigenvalues[::5] if hessian_eigenvalues else []

    return epochs, train_acc, val_acc, hessian_eigenvalues


def parse_pt_files(base_paths):
    train_accuracies = []
    test_accuracies = []
    eigenvalues = []

    for path in base_paths:
        try:
            train_accuracies.append(np.array(torch.load(f"{path}/train_acc.pt", weights_only=True)))
            test_accuracies.append(np.array(torch.load(f"{path}/test_acc.pt", weights_only=True)))
            eigenvalues.append(np.array(torch.load(f"{path}/eigenvalues.pt", weights_only=True)))
        except FileNotFoundError as e:
            print(f"Error loading data from {path}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    return train_accuracies, test_accuracies, eigenvalues


def plot_combined(log_files, labels_log, base_paths, labels_pt):
    # Parse log files
    log_data = [parse_log_file(file) for file in log_files]

    # Parse .pt files (excluding h0=0.8)
    train_accuracies_pt, test_accuracies_pt, eigenvalues_pt = parse_pt_files(base_paths[:-1])  # Exclude h0=0.8

    fig, axs = plt.subplots(3, 1, figsize=(15,20 ), dpi=120)

    title_fontsize = 28
    label_fontsize = 24
    legend_fontsize = 24
    tick_fontsize = 24

    colors = ['blue', 'orange', 'green', 'red']
    markers = ['o', 's', '^', 'd']
    linestyles = ['-', '--', '-.', ':']

    # Combine data from log files and .pt files
    combined_labels = labels_log + labels_pt[:-1]  # Exclude h0=0.8 from labels
    combined_train_accuracies = [log[1] for log in log_data] + train_accuracies_pt
    combined_val_accuracies = [log[2] for log in log_data] + test_accuracies_pt
    combined_eigenvalues = [log[3] for log in log_data] + eigenvalues_pt
    combined_epochs = [log[0] for log in log_data] + [np.arange(len(acc)) * 50 for acc in train_accuracies_pt]

    # Plot Training Accuracy
    for i, (epochs, train_acc) in enumerate(zip(combined_epochs, combined_train_accuracies)):
        axs[0].plot(
            epochs, train_acc, label=combined_labels[i], color=colors[i % len(colors)],
            marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)],
            linewidth=2, markersize=6
        )
    axs[0].set_title("Training Accuracy", fontsize=title_fontsize)

    # Plot Validation Accuracy
    for i, (epochs, val_acc) in enumerate(zip(combined_epochs, combined_val_accuracies)):
        axs[1].plot(
            epochs, val_acc, label=combined_labels[i], color=colors[i % len(colors)],
            marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)],
            linewidth=2, markersize=6
        )
    axs[1].set_ylim(0.5, 0.63)
    axs[1].set_title("Validation Accuracy", fontsize=title_fontsize)

    # Plot Sharpness (Eigenvalues)
    for i, eigs in enumerate(combined_eigenvalues):
        if eigs is not None:
            if i > 1:
                axs[2].scatter(
                    np.arange(len(eigs)) * 200, eigs, label=combined_labels[i], color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], s=40, alpha=0.8
                )
            else:
                axs[2].scatter(
                    np.arange(len(eigs)) * 500, eigs, label=combined_labels[i], color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], s=40, alpha=0.8
                )
    axs[2].axhline(400, color='black', linestyle='dashed', linewidth=2, label=r"$\frac{2}{\rho}$")
    axs[2].set_title("Sharpness", fontsize=title_fontsize)

    # Labels, ticks, and legends
    for ax in axs:
        ax.set_xlabel("Epoch", fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.grid(True)
        ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig('vit/combined_train_val_sharpness.png', bbox_inches='tight')
    plt.show()


# File paths
log_files = ['vit/gd-lr0.005-cifar50k.out', 'vit/ivon-lr0.005-cifar50k.out']
labels_log = ['GD', 'IVON $h_0=0.1$']
base_paths = [
    "RESULTS/ivon/vit/0.2",
    "RESULTS/ivon/vit/0.5",
    "RESULTS/ivon/vit/0.8"
]
labels_pt = ['IVON $h_0=0.2$', 'IVON $h_0=0.5$', 'IVON $h_0=0.8$']

# Call the combined plotting function excluding h0=0.8
plot_combined(log_files, labels_log, base_paths, labels_pt)
