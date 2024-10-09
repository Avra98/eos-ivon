import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ivon._ivon import IVON



# Define the Ackley function
def ackley_function(x, a=20, b=0.2, c=2 * torch.pi):
    d = x.shape[0]
    sum1 = torch.sum(x ** 2)
    sum2 = torch.sum(torch.cos(c * x))
    term1 = -a * torch.exp(-b * torch.sqrt(sum1 / d))
    term2 = -torch.exp(sum2 / d)
    return term1 + term2 + a + torch.exp(torch.tensor(1.0))

# Define the Holder Table function
def holder_table_function(x):
    term1 = torch.sin(x[0]) * torch.cos(x[1])
    term2 = torch.exp(torch.abs(1 - torch.sqrt(x[0]**2 + x[1]**2) / torch.pi))
    return -torch.abs(term1 * term2)

# Define the Levy function
def levy_function(x):
    term1 = torch.sin(3 * torch.pi * x[0]) ** 2
    term2 = (x[0] - 1) ** 2 * (1 + torch.sin(3 * torch.pi * x[1]) ** 2)
    term3 = (x[1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * x[1]) ** 2)
    return term1 + term2 + term3

# Calculate the Hessian matrix
def hessian(f, x):
    x = x.clone().detach().requires_grad_(True)
    hess = torch.zeros((x.shape[0], x.shape[0]))
    grad = torch.autograd.grad(f(x), x, create_graph=True)[0]
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            hess[i, j] = torch.autograd.grad(grad[i], x, retain_graph=True)[0][j]
    return hess


def get_learning_rate(lr,iteration, num_iterations):
    if iteration < num_iterations // 3:
        return lr * 0.1  # Small constant learning rate
    elif iteration < 2 * num_iterations // 3:
        return lr * 0.2  # Medium constant learning rate
    else:
        return lr  # High constant learning rate
        

def main(args):
    # Select the function to optimize
    if args.function == 'ackley':
        func = ackley_function
        plot_title = 'Ackley Function'
        file_prefix = 'ackley'
        grid_range = 10
        initial_x = torch.tensor(args.initial_x, requires_grad=True)
    elif args.function == 'holder':
        func = holder_table_function
        plot_title = 'Holder Table Function'
        file_prefix = 'holder_table'
        grid_range = 10
        initial_x = torch.tensor(args.initial_x, requires_grad=True)
    elif args.function == 'levy':
        func = levy_function
        plot_title = 'Levy Function N. 13'
        file_prefix = 'levy'
        grid_range = 10
        initial_x = torch.tensor(args.initial_x, requires_grad=True)
    else:
        raise ValueError("Unsupported function. Choose from 'ackley', 'holder', or 'levy'.")

    # Gradient descent parameters
    init_learning_rate = args.init_learning_rate
    num_iterations = args.num_iterations
    post_samples=args.post_samples
    opt=args.opt
    h0=args.h0

    # Prepare 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a grid of points
    x1 = np.linspace(-grid_range, grid_range, 100)
    x2 = np.linspace(-grid_range, grid_range, 100)
    x1, x2 = np.meshgrid(x1, x2)
    x1x2 = np.stack([x1, x2], axis=-1)

    # Compute the function value for the grid
    z = np.array([[func(torch.tensor([x, y])).item() for x, y in row] for row in x1x2])

    # Plot the function surface with transparency
    ax.plot_surface(x1, x2, z, cmap='viridis', alpha=0.4, rstride=50, cstride=50, linewidth=0.5, edgecolor='k')

    # Set the view angle to top view
    ax.view_init(elev=40, azim=-40)

    # Gradient descent loop
    x = initial_x.clone().detach().requires_grad_(True)
    sharpness_values = []
    lr_reciprocal_values = []
    optimizer = IVON([x], lr=init_learning_rate, ess= 1.0, weight_decay=0.0,mc_samples=post_samples, beta1=0.0,beta2=1.0, hess_init=h0)
    for i in range(num_iterations):

        if opt=="gd":
            # Zero the gradients
            if x.grad is not None:
                x.grad.zero_()

            # Compute the loss (function value)
            loss = func(x)

            # Compute gradients
            loss.backward()

            learning_rate = get_learning_rate(init_learning_rate,i,num_iterations)
            # Update x using gradient descent
            with torch.no_grad():
                x -= learning_rate * x.grad

        elif opt=="ivon":
            optimizer.zero_grad()
            for _ in range(post_samples):
                optimizer.zero_grad()
                with optimizer.sampled_params(train=True): 
                    loss =func(x)
                    loss.backward()
                    
            optimizer.step()  
            #print("x is",x)     

        # Calculate the Hessian and its maximum eigenvalue
        hess = hessian(func, x)
        #print("hess is", hess)
        max_eigenvalue = torch.max(torch.linalg.eigvals(hess).real)
        #print('max_eigenvalue is',max_eigenvalue )
        sharpness_values.append(max_eigenvalue.item())
        if opt=="ivon":
            learning_rate = init_learning_rate
            lr_reciprocal_values.append(2 / learning_rate)
        else:    
            lr_reciprocal_values.append(2 / learning_rate)

        # Plot the current position
        if i % 10 == 0:
            ax.scatter(x[0].item(), x[1].item(), func(x).item(), color='r', s=10)
            plt.draw()
            plt.pause(0.01)

        # Print the loss and sharpness every 100 iterations
        if i % 10 == 0:
            print(f"Iteration {i}: loss = {loss.item()}, iterates: {x},sharpness = {max_eigenvalue.item()}, lr = {learning_rate}")

    print(f"Final x: {x}")

    # Save the final plot
    plt.title(plot_title)
    plt.savefig(f'{file_prefix}_optimization_top_view_{opt}_h0{h0}_post{post_samples}.png')
    plt.show()

    plt.figure()
    plt.plot(sharpness_values, label='Sharpness (Max Eigenvalue of Hessian)')
    plt.plot(lr_reciprocal_values, label='2 / Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.ylim(0,100)
    plt.legend()
    plt.savefig(f'sharpness_plot_{file_prefix}_{opt}_h0{h0}_post{post_samples}.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Function Optimization using Gradient Descent')
    parser.add_argument('--function', type=str, default = 'holder', choices=['ackley', 'holder', 'levy'], help='Function to optimize')
    parser.add_argument('--initial_x', type=float, nargs=2, default=[7.5, 5.0], help='Initial value of x')
    parser.add_argument('--init_learning_rate', type=float, default=0.1, help='Learning rate for gradient descent')
    parser.add_argument('--num_iterations', type=int, default=1000, help='Number of iterations for gradient descent')
    parser.add_argument("--opt", type=str, default="gd", help="gd or ivon")
    parser.add_argument("--h0", type=float, default=0.1, help="Posterior variance init (not used in this example)")
    parser.add_argument("--post_samples", type=int, default=10, help="Number of posterior samples (not used in this example)")
    args = parser.parse_args()
    main(args)
