# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gamma, t

# # Parameters
# n = int(1e5)
# alpha = 2.5 

# # Sampling tau from a gamma distribution
# tau_sample = gamma.rvs(a=alpha/2, scale=2/alpha, size=n)

# # Sampling x from a normal distribution with standard deviation depending on tau_sample
# x_sample = np.random.normal(loc=0, scale=np.sqrt(1/tau_sample), size=n)

# print(tau_sample)
# # Plotting the histogram of x_sample
# plt.hist(x_sample, bins=100, density=True, alpha=0.75, color='blue')

# # Generating values for the t-distribution line plot
# x_interval = np.linspace(min(x_sample), max(x_sample), 1000)
# t_values = t.pdf(x_interval, df=alpha)

# # Adding a line plot for the t-distribution
# plt.plot(x_interval, t_values, 'r-', linewidth=2)
# plt.title(f"distribution for alpha={alpha}")
# plt.savefig('t_student.png')

# # Show plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, t

# Parameters
n = int(1e5)
alpha_values = (1,2,3,10,1000) # Varying alpha from 1 to 10

# Prepare the plot with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

for alpha in alpha_values:
    # Sampling tau from a gamma distribution
    tau_sample = gamma.rvs(a=alpha/2, scale=2/alpha, size=n)

    # Sampling x from a normal distribution with standard deviation depending on tau_sample
    x_sample = np.random.normal(loc=0, scale=np.sqrt(1/tau_sample), size=n)

    # Generating values for the t-distribution line plot
    x_interval = np.linspace(-10, 10, 1000)  # Narrowing the range for clarity
    t_values = t.pdf(x_interval, df=alpha)

    # Plotting the histogram of x_sample with alpha normalization
    axs[0].hist(x_sample, bins=100, density=True, alpha=0.3, label=f'alpha={alpha}')

    # Adding a line plot for the t-distribution
    axs[1].plot(x_interval, t_values, '-', linewidth=2, label=f't-distribution alpha={alpha}')

# Customize the histogram subplot
axs[0].set_title("Sampled histograms for alphas", fontsize=16)
axs[0].set_xlabel('x', fontsize=14)
axs[0].set_ylabel('Density', fontsize=14)
axs[0].set_xlim([-10, 10])  # Limiting the x-axis for better clarity
axs[0].grid(True)
axs[0].legend(loc='upper right')

# Customize the t-distribution subplot
axs[1].set_title("pdf for alphas", fontsize=16)
axs[1].set_xlabel('x', fontsize=14)
axs[1].set_ylabel('Density', fontsize=14)
axs[1].set_xlim([-10, 10])  # Limiting the x-axis for better clarity
axs[1].grid(True)
axs[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('t_student.png')

# Show plot
plt.show()

