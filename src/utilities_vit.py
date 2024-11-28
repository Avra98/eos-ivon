import math
import os
import sys

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities import DEFAULT_PHYS_BS, get_loss_and_acc


def get_param_with_grad(net):
    return [param for param in net.parameters() if param.requires_grad]


def compute_hvp(
    network, loss_fn, data, target, vector, device, trainable_only, physical_batch_size
):
    """Compute a Hessian-vector product with specified device."""
    n = len(data)
    hvp = torch.zeros_like(vector, device=device)
    chunk_num = math.ceil(len(data) / physical_batch_size)
    for X, y in zip(data.chunk(chunk_num), target.chunk(chunk_num)):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(
            loss,
            inputs=get_param_with_grad(network) if trainable_only else network.parameters(),
            create_graph=True,
            allow_unused=True,
        )
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [
            g.contiguous()
            for g in torch.autograd.grad(
                dot,
                get_param_with_grad(network) if trainable_only else network.parameters(),
                retain_graph=True,
            )
        ]
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
    return (
        torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).to(device).float(),
        torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).to(device).float(),
    )


def get_hessian_eigenvalues(
    network, loss_fn, data, target, trainable_only, neigs, physical_batch_size=1000, device="cuda"
):
    """Compute the leading Hessian eigenvalues using a specified device."""
    hvp_delta = lambda delta: compute_hvp(
        network, loss_fn, data, target, delta, device, trainable_only, physical_batch_size
    ).detach()
    params = get_param_with_grad(network) if trainable_only else network.parameters()
    nparams = len(parameters_to_vector(params).to(device))
    evals, evecs = lanczos(hvp_delta, nparams, neigs, device)
    return evals
