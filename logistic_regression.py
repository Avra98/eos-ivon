import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.sparse.linalg import LinearOperator, eigsh
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ivon._ivon import IVON

# Function to compute Hessian-vector product
def compute_hvp(model, loss, vector):
    # First pass to compute the gradients
    grads = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
    
    # Compute dot product with the vector
    dot = parameters_to_vector(grads).mul(vector).sum()
    
    # Second backward pass to get the Hessian-vector product
    hvp= [g.contiguous() for g in torch.autograd.grad(dot, model.parameters(), retain_graph=True)]
    
    # Flatten and return the Hessian-vector product
    return parameters_to_vector(hvp)


def l2_regularization(model, lambda_reg):
    """Compute the L2 regularization term for all parameters in the model."""
    l2_norm = sum(param.norm(2) ** 2 for param in model.parameters())
    return lambda_reg * l2_norm


# Lanczos algorithm to compute largest eigenvalue
def lanczos(hvp_func, dim, neigs=1):
    def mv(vec):
        vec_tensor = torch.tensor(vec, dtype=torch.float32)
        hvp_result = hvp_func(vec_tensor)
        return hvp_result.cpu().detach().numpy()

    # Define the linear operator that represents the Hessian-vector product function
    operator = LinearOperator((dim, dim), matvec=mv)

    # Compute eigenvalues using Lanczos (SciPy eigsh function)
    evals, _ = eigsh(operator, k=neigs, which='LM')  # 'LM' -> Largest Magnitude
    return evals[-1]  # Return the largest eigenvalue


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)  # bias=True by default
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def generate_separable_data(num_points, seed=0):
    torch.manual_seed(seed)
    
    # Generate two clusters of points
    X_class1 = torch.randn(num_points//2 - 1, 2) - 2  # -2 shift for class 1
    X_class2 = torch.randn(num_points//2 - 1, 2) + 2  # +2 shift for class 2
    
    # Add two points close to each other but belonging to different classes
    X_class1_close = torch.tensor([[-0.01, 0.01]])  # Point close to origin for class 1
    X_class2_close = torch.tensor([[0.01, -0.01]])  # Point close to origin for class 2
    
    # Combine points
    X = torch.cat([X_class1, X_class1_close, X_class2, X_class2_close], dim=0)
    y = torch.cat([torch.ones(num_points//2), -torch.ones(num_points//2)])
    
    return X, y

def plot_decision_boundary(X, y, model, learning_rate,opt,h0):
    # Create a grid to plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Flatten grid to pass through the model
    grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    
    # Get model predictions
    with torch.no_grad():
        Z = model(grid).reshape(xx.shape)
    
    # Plot contour and data points
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if opt=="gd":
        plt.title(f'{opt} with lr:{learning_rate}')
    else:
        plt.title(f'{opt} with lr:{learning_rate} and h0:{h0}')  
    
    # Save the plot as a PNG file with learning rate in the file name
    plt.savefig(f'decision_boundary_lr_{learning_rate}_opt_{opt}_h0_{h0}_wd.png')
    
    # Show the plot
    plt.show()
    plt.close()

def plot_loss_curve(losses, eigs,learning_rate,opt,h0):
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    
    # Save the plot as a PNG file with learning rate in the file name
    plt.savefig(f'loss_curve_lr_{learning_rate}_opt_{opt}_h0_{h0}.png')
    
    # Show the plot
    plt.show()

def main(num_points, learning_rate, seed, opt, h0, post_samples):
    save_dir = f'regression_results/lr_{learning_rate}_opt_{opt}_h0_{h0}_wd'
    os.makedirs(save_dir, exist_ok=True)
    # Generate data
    X, y = generate_separable_data(num_points, seed)
    
    # Initialize the model (no manual bias term added)
    model = LogisticRegressionModel(input_dim=X.shape[1])
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    
    if opt == "gd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:    
        optimizer = IVON(model.parameters(), lr=learning_rate, ess=1.0, weight_decay=0.0, mc_samples=post_samples, beta1=0.0, beta2=1.0, hess_init=h0)
    
    # List to store loss values
    losses = []
    eigs=[]
    
    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass
        if opt=="gd":
            outputs = model(X).squeeze()
            loss = criterion(outputs, (y + 1) / 2)  # Adjust labels from {-1, 1} to {0, 1}            
            # Store the loss value
            losses.append(loss.item())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else: 
            optimizer.zero_grad()    
            for _ in range(post_samples):
                with optimizer.sampled_params(train=True):   
                    optimizer.zero_grad()
                    outputs = model(X).squeeze()
                    loss = criterion(outputs, (y + 1) / 2) # Adjust labels from {-1, 1} to {0, 1}
                    loss.backward()
            losses.append(loss.item())     
            optimizer.step()

 
            # Optionally print loss every 100 iterations
        if epoch % 10 == 0:
            outputs = model(X).squeeze()
            loss = criterion(outputs, (y + 1) / 2) # Recreate loss for Hessian
            vector = torch.randn(parameters_to_vector(model.parameters()).shape)  # Random vector
            hvp_func = lambda v: compute_hvp(model, loss, v)  # Use fresh loss

            # Compute max eigenvalue using Lanczos
            max_eigenvalue = lanczos(hvp_func, vector.numel())
            eigs.append(max_eigenvalue)
            print(f'Epoch {epoch}: Loss = {loss.item()},sharpness = {max_eigenvalue.item()}')
    
    # Output final weights
    print(f'Final weights: {model.linear.weight.data.numpy()}')
    print(f'Final bias: {model.linear.bias.data.numpy()}')


    np.save(os.path.join(save_dir, 'losses_wd.npy'), np.array(np.log(losses)))
    np.save(os.path.join(save_dir, 'eigs_wd.npy'), np.array(np.log(eigs)))
    # Plot and save decision boundary
    plot_decision_boundary(X, y, model, learning_rate,opt,h0)
    
    # Plot and save loss curve
    plot_loss_curve(losses, eigs,learning_rate,opt,h0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for gradient descent')
    parser.add_argument('--num_points', type=int, default=100, help='Number of data points to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data generation')
    parser.add_argument('--opt', type=str, default="gd", help='Optimizer choice (gd for gradient descent, ivon for IVON optimizer)')
    parser.add_argument('--h0', type=float, default=0.1, help='Posterior variance initialization for IVON optimizer')
    parser.add_argument('--post_samples', type=int, default=10, help='Number of posterior samples for IVON optimizer')
    
    args = parser.parse_args()
    
    main(args.num_points, args.learning_rate, args.seed, args.opt, args.h0, args.post_samples)
