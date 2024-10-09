import numpy as np
import os
import matplotlib.pyplot as plt

def plot_multiple_results(settings):
    """
    Plots loss curves and maximum eigenvalue curves for multiple experiments.
    
    Parameters:
    - settings: list of dicts, each dict contains 'learning_rate', 'opt', and 'h0' settings.
    """
    # Define color scheme: darker color for 'gd', lighter colors for others
    color_scheme = {
        'gd': 'darkblue',
        'ivon_0.1': 'skyblue',
        'ivon_1.0': 'lightgreen'
    }

    plt.figure(figsize=(14, 6))  # Create a figure with 2 subplots (side by side)
    
    # Sort settings to ensure 'gd' is plotted last
    settings = sorted(settings, key=lambda x: x['opt'] == 'gd', reverse=True)

    # Loop through the provided settings and load the loss/eigs
    for setting in settings:
        lr = setting['learning_rate']
        opt = setting['opt']
        h0 = setting['h0']
        exp_dir = f'regression_results/lr_{lr}_opt_{opt}_h0_{h0}_wd'
        
        # Load loss and eigs arrays
        losses_path = os.path.join(exp_dir, 'losses_wd.npy')
        eigs_path = os.path.join(exp_dir, 'eigs_wd.npy')
        
        if not os.path.exists(losses_path) or not os.path.exists(eigs_path):
            print(f"Skipping {exp_dir}, files not found.")
            continue

        losses = np.load(losses_path)
        eigs = np.load(eigs_path)

        # Determine the color for the current plot based on optimizer and h0
        if opt == "gd":
            color = color_scheme['gd']
            label = f'LR={lr}, Opt={opt}'  # Omit h0 for 'gd'
            line_width = 5  # Make the gd line bolder
            z_order = 10  # Plot 'gd' on top of others
        elif opt == "ivon" and h0 == 0.1:
            color = color_scheme['ivon_0.1']
            label = f'LR={lr}, Opt={opt}, h0={h0}'
            line_width = 2
            z_order = 5  # Default order
        elif opt == "ivon" and h0 == 1.0:
            color = color_scheme['ivon_1.0']
            label = f'LR={lr}, Opt={opt}, h0={h0}'
            line_width = 2
            z_order = 5  # Default order
        else:
            color = 'gray'
            label = f'LR={lr}, Opt={opt}, h0={h0}'
            line_width = 2
            z_order = 5  # Default order
        
        # Calculate indices for plotting eigs (eigs are computed every 10 epochs)
        eigs_x = np.arange(0, len(losses), len(losses) // len(eigs))

        # Plot the loss curve
        plt.subplot(1, 2, 1)
        plt.plot(range(len(losses)), losses, label=label, color=color, linewidth=line_width, alpha=0.8, zorder=z_order)
        
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Log Loss', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Loss', fontsize=16)
        plt.legend(fontsize=12)

        # Plot the max eigenvalue curve
        plt.subplot(1, 2, 2)
        plt.plot(eigs_x, eigs, label=label, color=color, marker='o', markersize=6, linewidth=line_width, alpha=0.8, zorder=z_order)
        
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Log Sharpness', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Sharpness', fontsize=16)
        plt.legend(fontsize=12)

    # Adjust the layout and save/show the plots
    plt.tight_layout()

    plt.savefig(f'regression_results/comparison_plot_wd.png', bbox_inches='tight')
    plt.show()

# Example settings for different experiments
settings = [
    {'learning_rate': 0.1, 'opt': 'gd', 'h0': 0.1},  # 'h0' won't be shown for 'gd'
    {'learning_rate': 0.1, 'opt': 'ivon', 'h0': 0.1},
    {'learning_rate': 0.1, 'opt': 'ivon', 'h0': 1.0}
]

# Call the function to plot the results
plot_multiple_results(settings)
