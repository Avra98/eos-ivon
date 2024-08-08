from typing import List, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD, Adam
import ivon
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt

# the default value for "physical batch size", which is the largest batch size that we try to put on the GPU
DEFAULT_PHYS_BS = 1000


def get_gd_directory(dataset: str, lr: float, arch_id: str, seed: int, opt: str, loss: str, beta: float = None):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/{opt}/"
    if opt == "gd":
        return f"{directory}/lr_{lr}"
    elif opt == "polyak" or opt == "nesterov":
        return f"{directory}/lr_{lr}_beta_{beta}"
    elif opt == "adam":
        return f"{directory}/lr_{lr}_adam"
    elif opt == "ivon":
        return f"{directory}/lr_{lr}_ivon"


def get_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/flow/tick_{tick}"


def get_modified_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, gd_lr: float, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/modified_flow_lr_{gd_lr}/tick_{tick}"


def get_gd_optimizer(parameters, opt: str, lr: float, momentum: float, train_samp: float = None, beta2: float = None, h0: float =None) -> Optimizer:
    if opt == "gd":
        return SGD(parameters, lr=lr)
    elif opt == "polyak":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=False)
    elif opt == "nesterov":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=True)
    elif opt == "adam":
        return Adam(parameters, lr=lr)
    elif opt == "ivon":
        print(h0,beta2)
        return ivon.IVON(parameters, lr=lr, ess= train_samp, weight_decay=1e-4, beta1=momentum,beta2=beta2, hess_init=h0)


def save_files(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}")


def save_files_final(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}_final")


def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X.cuda(), batch_y.cuda()


def compute_losses(network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset,
                   batch_size: int = DEFAULT_PHYS_BS, device: torch.device = torch.device('cpu')):
    """Compute loss over a dataset, with data and model moved to the specified device."""
    # Move the model to the specified device
    network.to(device)

    L = len(loss_functions)
    losses = [0. for _ in range(L)]
    with torch.no_grad():  # No gradients needed for loss computation
        for (X, y) in iterate_dataset(dataset, batch_size):
            # Move data to the specified device
            X, y = X.to(device), y.to(device)            
            preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                # Accumulate the loss, normalized by the dataset size
                losses[l] += loss_fn(preds, y).item() / len(dataset)
    return losses

# def compute_losses(network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset,
#                    batch_size: int = DEFAULT_PHYS_BS):
#     """Compute loss over a dataset."""
#     L = len(loss_functions)
#     losses = [0. for l in range(L)]
#     with torch.no_grad():
#         for (X, y) in iterate_dataset(dataset, batch_size):
#             preds = network(X)
#             for l, loss_fn in enumerate(loss_functions):
#                 losses[l] += loss_fn(preds, y) / len(dataset)
#     return losses

def get_loss_and_acc(loss: str):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()
    raise NotImplementedError(f"no such loss function: {loss}")


def compute_hvp(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, 
                vector: Tensor, device: str, physical_batch_size: int = DEFAULT_PHYS_BS):
    """Compute a Hessian-vector product with specified device."""
    n = len(dataset)
    hvp = torch.zeros_like(vector, device=device)
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        X, y = X.to(device), y.to(device)
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    return hvp

def lanczos(matrix_vector, dim: int, neigs: int, device: str):
    """Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors."""
    def mv(vec: np.ndarray):
        # Create a tensor on the specified device
        gpu_vec = torch.tensor(vec, dtype=torch.float, device=device)
        # Perform the matrix-vector multiplication on the device
        result = matrix_vector(gpu_vec)
        # Ensure the result is moved to CPU and then converted to NumPy
        return result.cpu().numpy() 
    # Define the linear operator with the correct dimensions and the matvec function
    operator = LinearOperator((dim, dim), matvec=mv)
    # Compute the eigenvalues and eigenvectors using SciPy
    evals, evecs = eigsh(operator, neigs)
    # Convert results back to torch tensors on the specified device
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).to(device).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).to(device).float()


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                            neigs=6, physical_batch_size=1000, device='cuda'):
    """Compute the leading Hessian eigenvalues using a specified device."""
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset, 
                                          delta, device, physical_batch_size).detach()
    nparams = len(parameters_to_vector(network.parameters()).to(device))
    evals, evecs = lanczos(hvp_delta, nparams, neigs, device)
    return evals



def compute_gradient(network: nn.Module, loss_fn: nn.Module,
                     dataset: Dataset, physical_batch_size: int = DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at the current network parameters. """
    p = len(parameters_to_vector(network.parameters()))
    average_gradient = torch.zeros(p, device='cuda')
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        batch_loss = loss_fn(network(X), y) / len(dataset)
        batch_gradient = parameters_to_vector(torch.autograd.grad(batch_loss, inputs=network.parameters()))
        average_gradient += batch_gradient
    return average_gradient


class AtParams(object):
    """ Within a with block, install a new set of parameters into a network.

    Usage:

        # suppose the network has parameter vector old_params
        with AtParams(network, new_params):
            # now network has parameter vector new_params
            do_stuff()
        # now the network once again has parameter vector new_params
    """

    def __init__(self, network: nn.Module, new_params: Tensor):
        self.network = network
        self.new_params = new_params

    def __enter__(self):
        self.stash = parameters_to_vector(self.network.parameters())
        vector_to_parameters(self.new_params, self.network.parameters())

    def __exit__(self, type, value, traceback):
        vector_to_parameters(self.stash, self.network.parameters())


def compute_gradient_at_theta(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                              theta: torch.Tensor, batch_size=DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at arbitrary network parameters "theta".  """
    with AtParams(network, theta):
        return compute_gradient(network, loss_fn, dataset, physical_batch_size=batch_size)


class SquaredLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return 0.5 * ((input - target) ** 2).sum()


class SquaredAccuracy(nn.Module):
    def __init__(self):
        super(SquaredAccuracy, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target.argmax(1)).float().sum()


class AccuracyCE(nn.Module):
    def __init__(self):
        super(AccuracyCE, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target).float().sum()


class VoidLoss(nn.Module):
    def forward(self, X, Y):
        return 0


def initialize_storage(network):
    storage = []
    for layer in network.modules():
        if isinstance(layer, torch.nn.Linear):
            storage.append({'singular_values': [], 'U_matrices': []})
    return storage

# Function to initialize subspace distances storage
def initialize_subspace_distances(network):
    subspace_distances = []
    for layer in network.modules():
        if isinstance(layer, torch.nn.Linear):
            subspace_distances.append([])
    return subspace_distances

# def store_singular_values(network, storage):
#     for (i, layer), store in zip(enumerate(network.modules()), storage):
#         if isinstance(layer, torch.nn.Linear):
#             u, s, v = torch.svd(layer.weight.data)
#             store.append(s.cpu().numpy())  # Store the singular values as NumPy arrays
#             print(f"Layer {i+1} Singular Values: {s.cpu().numpy()}")

def store_singular_values(network, storage):
    linear_layer_index = 0  # Initialize an index to track linear layers specifically

    # Iterate through all modules in the network
    for layer in network.modules():
        if isinstance(layer, torch.nn.Linear):
            # Compute SVD on the weight data of the linear layer
            u, s, v = torch.svd(layer.weight.data)

            # Ensure the storage is appropriately sized for the number of linear layers encountered
            if linear_layer_index >= len(storage):
                storage.append([])  # Append a new list if needed

            # Store the singular values in the corresponding sub-list in storage
            storage[linear_layer_index].append(s.cpu().numpy())
            #print(f"Layer {linear_layer_index + 1} Singular Values: {s.cpu().numpy()}")

            linear_layer_index += 1  # Increment only for linear layers


def store_singular_values_and_subspace_distance(network, storage, subspace_distances, r=10):
    linear_layer_index = 0  # Initialize an index to track linear layers specifically

    # Iterate through all modules in the network
    for layer in network.modules():
        if isinstance(layer, torch.nn.Linear):
            # Compute SVD on the weight data of the linear layer
            u, s, v = torch.svd(layer.weight.data)
            
            # Store the singular values in the corresponding sub-list in storage
            storage[linear_layer_index]['singular_values'].append(s.cpu().numpy())
            # Store the U matrix in the corresponding sub-list in storage
            storage[linear_layer_index]['U_matrices'].append(u.cpu().numpy())

            # If this is not the first time step, compute the subspace distance
            if len(storage[linear_layer_index]['U_matrices']) > 1:
                U_t = storage[linear_layer_index]['U_matrices'][-1][:, :r]  # Use first r columns
                U_t1 = storage[linear_layer_index]['U_matrices'][-2][:, :r]  # Get the previous U matrix and use first r columns
                subspace_distance = np.abs(r - np.linalg.norm(np.dot(U_t.T, U_t1), ord='fro')**2)
                #print(f"Layer {linear_layer_index + 1} Subspace Distance: {subspace_distance}")
                subspace_distances[linear_layer_index].append(subspace_distance)
            else:
                subspace_distances[linear_layer_index].append(None)  # No previous step to compare

            linear_layer_index += 1  # Increment only for linear layers

def plot_singular_values(storage, output_directory):
    for i, singular_values in enumerate(storage):
        if singular_values:  # Check if the list is non-empty
            all_singular_values = np.stack(singular_values)  # Stack for plotting

            plt.figure(figsize=(12, 6))
            num_singular_values = all_singular_values.shape[1]  # Total number of singular values

            # Determine indices for top 5 and last 5 singular values
            indices_top_5 = range(5)
            indices_last_5 = range(max(0, num_singular_values - 5), num_singular_values)

            # Plot top 5 singular values
            for idx in indices_top_5:
                plt.plot(all_singular_values[:, idx], label=f'Singular Value {idx+1}')

            # Plot last 5 singular values
            for idx in indices_last_5:
                plt.plot(all_singular_values[:, idx], label=f'Singular Value {idx+1}')

            plt.title(f'Layer {i+1} Singular Value Evolution')
            plt.xlabel('Iteration')
            plt.ylabel('Singular Value')
            plt.legend()
            plt.savefig(f"{output_directory}/Layer_{i+1}_Singular_Values_Evolution.png")
            plt.close()
        else:
            print(f"Skipping Layer {i+1}: No singular values recorded.")