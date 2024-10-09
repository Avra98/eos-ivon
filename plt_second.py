import torch
import matplotlib.pyplot as plt
from os import environ
import numpy as np

# New dataset and file path
dataset = "cifar10-10k"
arch = "fc-tanh"
loss = "ce"
gd_lr = 0.05
gd_eig_freq = 20
h0 = 0.05

# Updated directory
gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0_diff/{loss}/ivon/lr_{gd_lr}_h0_{h0}_post_10_ivon"

# Loading the required data
gd_train_loss = torch.load(f"{gd_directory}/train_loss_final")
gd_train_acc = torch.load(f"{gd_directory}/train_acc_final")
gd_sharpness = torch.load(f"{gd_directory}/eigs_final")[:, 0]
rhs_values = torch.load(f"{gd_directory}/rhs_values_final")
scaled_losses = torch.load(f"{gd_directory}/scaled_losses_final")

print(rhs_values)
print(scaled_losses)

# Creating the plots
plt.figure(figsize=(6, 12), dpi=500)

# Plot 1: Train Loss
plt.subplot(4, 1, 1)
plt.plot(gd_train_loss)
plt.ylabel("Loss")
plt.title("Train Loss")

# Plot 2: Train Accuracy
plt.subplot(4, 1, 2)
plt.plot(gd_train_acc)
plt.ylabel("Accuracy")
plt.title("Train Accuracy")

# Plot 3: Sharpness
plt.subplot(4, 1, 3)
plt.scatter(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5)
plt.axhline(2. / gd_lr, linestyle='dotted')
plt.axvline(x=320, color='black', linestyle='dotted') 
plt.title("Sharpness")
plt.xlabel("Iteration")

# Plot 4: Transformed RHS Values vs Scaled Losses (combined scatter plot)
plt.subplot(4, 1, 4)
plt.scatter(torch.arange(len(gd_sharpness)) * gd_eig_freq, scaled_losses.cpu() - rhs_values.cpu() , s=5)
#plt.ylim(-0.2,0.5)
#plt.plot([min(scaled_losses.cpu()), max(scaled_losses.cpu())], [min(scaled_losses.cpu()), max(scaled_losses.cpu())], 'r--')  # x=y line
plt.axvline(x=320, color='black', linestyle='dotted') 
plt.ylim(-0.5,0.5)
plt.xlabel("Iteration")
plt.ylabel(r'$\frac{L(t+1) - L(t)}{\nabla L(t)^{\top} \mathbb{E}[g(t)]} - \frac{\eta^{2}}{2} \left[ \frac{ \mathbb{E}\left[ H(t) g(t) g(t)^{\top} \right]}{\nabla L(t)^{\top} \mathbb{E}[g(t)]} - \frac{2}{\eta} \right]$')
plt.title('Second order approximation ')


plt.tight_layout()

# Save the figure
plt.savefig(f"{gd_directory}/dynamics_second_h0{h0}.png",dpi=500)

# Display the figure
plt.show()
