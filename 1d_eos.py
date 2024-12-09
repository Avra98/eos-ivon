import numpy as np
import matplotlib.pyplot as plt

# Define the function H_crit(z)
def H_crit(z, rho):
    return 2 * np.sqrt(z / 3) * np.sinh((1/3) * np.arcsinh((3 / rho) * np.sqrt(3 / z)))

# Parameters
rho = 1.5  # Example value for rho; adjust as needed
z = np.linspace(0.1, 50, 500)  # Define z range to avoid division by zero

# Compute H_crit(z)
H_values = H_crit(z, rho)

# Plot the function
plt.figure(figsize=(10, 8))
plt.plot(z, H_values, label=r'$H_{\mathrm{crit}}(z)$', color='blue')

# Plot the horizontal line
horizontal_line = 2 / rho
plt.axhline(y=horizontal_line, color='red', linestyle='--', label=r'$\frac{2}{\rho}$')

# Add labels and legend
plt.xlabel(r'$z= \frac{N_s \|\mathbf{g}\|^2}{\mathrm{Tr}(\Sigma)}$', fontsize=24)
plt.ylabel(r'$H_{\mathrm{crit}}(z)$', fontsize=24)
plt.legend(fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)



# Add grid
plt.grid(True)

# Save the figure
plt.savefig('H_crit_plot.png', dpi=300, bbox_inches='tight')  # Save with high resolution

# Show the plot
plt.show()
