import torch

def _welford_mean(avg, newval, count):
    return newval if avg is None else avg + (newval - avg) / count

# Parameters for the Gaussian distribution
mean = 0.0
stddev = 2.0
n_samples = 100

# Initialize variables
avg = None
count = 0

# Sample from the Gaussian distribution and compute the Welford mean
for _ in range(n_samples):
    newval = torch.randn(1) * stddev + mean  # Sample from Gaussian
    count += 1
    avg = _welford_mean(avg, newval, count)

print(f"Final Welford mean after {n_samples} samples: {avg.item()}")
