import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import math
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ivon._ivon import IVON


def quadratic_function(x):
    return 0.5 * (x - 3) ** 2


def check_status(x, loss):
    if x is None or loss is None:
        return 1  # Divergence
    if 2.5 <= x <= 3.5:
        return 0  # Convergence
    if isinstance(loss, torch.Tensor):
        if torch.isnan(loss) or torch.isinf(loss):
            return 1  # Divergence
    else:
        if math.isnan(loss) or math.isinf(loss):
            return 1  # Divergence
    return 2  # Oscillation


def run_experiment(lr, max_steps, seed, h0, post_samples, opt, device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    x = torch.tensor([0.0], requires_grad=True, device=device)

    if opt == 'gd':
        optimizer = torch.optim.SGD([x], lr=lr)
    elif opt == 'ivon':
        optimizer = IVON([x], lr=lr, ess=1e1, weight_decay=0.0, mc_samples=post_samples, beta1=0.0, beta2=1.0, hess_init=h0)

    for step in range(max_steps):
        optimizer.zero_grad()
        if opt == 'gd':
            loss = quadratic_function(x)
            loss.backward()
        elif opt == 'ivon':
            for _ in range(post_samples):
                optimizer.zero_grad()
                with optimizer.sampled_params(train=True):
                    loss = quadratic_function(x)
                    loss.backward()
        optimizer.step()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN encountered at step {step}. Terminating early.")
            return None, None
        
        if step % 10 == 0:
            print(f"Step {step}: x = {x.item():.4f}, loss = {loss.item():.4f}")
            print(optimizer.state_dict()['param_groups'][0]['hess'])

    final_x = x.item()
    final_loss = loss.item()
    return final_x, final_loss


def main(lr, max_steps, seed, opt, device_id):
    h0_values = np.logspace(-3, 0, 10)  # Dense grid for h0
    post_samples_values = np.logspace(0, 2, 10, dtype=int)  # Dense grid for post_samples as integers


    status_matrix = np.zeros((len(h0_values), len(post_samples_values)))

    for i, h0 in enumerate(h0_values):
        for j, post_samples in enumerate(post_samples_values):
            print(f"Running experiment with h0={h0}, post_samples={post_samples}")
            final_x, final_loss = run_experiment(lr, max_steps, seed, h0, post_samples, opt, device_id)
            status = check_status(final_x, final_loss)
            status_matrix[i, j] = status
            print(f"Result: h0={h0}, post_samples={post_samples}, status={status}\n")

    # Plotting the results
    X, Y = np.meshgrid(post_samples_values, h0_values)
    plt.figure(figsize=(8, 6))

    cmap = mcolors.ListedColormap(['blue', 'red', 'orange'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.pcolormesh(X, Y, status_matrix, cmap=cmap, norm=norm, shading='auto')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim([post_samples_values.min(), post_samples_values.max()])
    plt.ylim([h0_values.min(), h0_values.max()])

    plt.title(f"Convergence, Divergence, and Oscillation Regions (Log Scale)\nLearning Rate: {lr:.6f}")

    plt.xlabel("post_samples (log scale)")
    plt.ylabel("h0 (log scale)")
    plt.xticks(post_samples_values)
    plt.yticks(h0_values)

    convergence_patch = mpatches.Patch(color='blue', label='Convergence')
    divergence_patch = mpatches.Patch(color='red', label='Divergence')
    oscillation_patch = mpatches.Patch(color='orange', label='Oscillation')
    plt.legend(handles=[convergence_patch, divergence_patch, oscillation_patch], loc='upper right')
    
    filename = f"convergence_divergence_oscillation_lr_{lr:.6f}.png"
    plt.savefig(filename)
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimization experiments with different h0 and post_samples values.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for the optimizer.")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of optimization steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--opt", type=str, default="ivon", help="Optimizer type: 'gd' or 'ivon'.")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the GPU to use (if available).")

    args = parser.parse_args()

    main(lr=args.lr, max_steps=args.max_steps, seed=args.seed, opt=args.opt, device_id=args.device_id)
