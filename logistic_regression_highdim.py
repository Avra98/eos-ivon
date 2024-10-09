import torch
import torch.nn as nn
import torch.optim as optim
import argparse

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def generate_separable_data(num_points, num_features, seed=0):
    torch.manual_seed(seed)
    
    # Generate two clusters of points in high-dimensional space
    X = torch.cat([torch.randn(num_points//2, num_features) - 2, torch.randn(num_points//2, num_features) + 2], dim=0)
    y = torch.cat([torch.ones(num_points//2), -torch.ones(num_points//2)])
    
    return X, y

def main(num_points, num_features, learning_rate, seed=0):
    # Generate data
    X, y = generate_separable_data(num_points, num_features, seed)
    
    # Add bias term to X
    X_with_bias = torch.cat((torch.ones(num_points, 1), X), dim=1)
    
    # Initialize the model
    model = LogisticRegressionModel(X_with_bias.shape[1])
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # List to store loss values
    losses = []
    
    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_with_bias).squeeze()
        loss = criterion(outputs, (y + 1) / 2)  # Adjust labels from {-1, 1} to {0, 1}
        
        # Store the loss value
        losses.append(loss.item())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Optionally print loss every 100 iterations
        if epoch % 1 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')
    
    # Output final weights
    print(f'Final weights: {model.linear.weight.data.numpy()}')
    print(f'Final bias: {model.linear.bias.data.numpy()}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for gradient descent')
    parser.add_argument('--num_points', type=int, default=100, help='Number of data points to generate')
    parser.add_argument('--num_features', type=int, default=10, help='Number of features (dimensions) in the data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data generation')
    
    args = parser.parse_args()
    
    main(args.num_points, args.num_features, args.learning_rate, args.seed)
