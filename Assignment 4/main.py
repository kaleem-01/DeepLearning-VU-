import torchvision.transforms as transforms
import os
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

print(f'The available device is {device}')
PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-5



# Class Encoder
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)    # Output: (32, 14, 14)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)   # Output: (64, 7, 7)
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(64 * 7 * 7, z_dim)
        self.fc_log_var = nn.Linear(64 * 7 * 7, z_dim)

    def encode(self, x):
        h = F.relu(self.conv1(x))    # Shape: (batch_size, 32, 14, 14)
        h = F.relu(self.conv2(h))    # Shape: (batch_size, 64, 7, 7)
        h = h.view(h.size(0), -1)    # Flatten: (batch_size, 64*7*7)
        mu = self.fc_mu(h)           # Shape: (batch_size, z_dim)
        log_var = self.fc_log_var(h) # Shape: (batch_size, z_dim)
        return mu, log_var

    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

# Class Decoder
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        # Fully connected layer to expand z to a suitable size
        self.fc = nn.Linear(z_dim, 64 * 7 * 7)
        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Output: (32, 14, 14)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)   # Output: (1, 28, 28)

    def decode(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)         # Shape: (batch_size, 64, 7, 7)
        h = F.relu(self.deconv1(h))      # Shape: (batch_size, 32, 14, 14)
        h = self.deconv2(h)              # Shape: (batch_size, 1, 28, 28)
        return h

    def forward(self, z, x):
        x_logits = self.decode(z)
        # Flatten tensors for loss computation
        x_flat = x.view(x.size(0), -1)                # Shape: (batch_size, 784)
        x_logits_flat = x_logits.view(x.size(0), -1)  # Shape: (batch_size, 784)
        # Compute binary cross-entropy loss
        bce = F.binary_cross_entropy_with_logits(x_logits_flat, x_flat, reduction='none')
        log_px_given_z = -torch.sum(bce, dim=1)
        return log_px_given_z

    def sample(self, z):
        x_logits = self.decode(z)
        x_probs = torch.sigmoid(x_logits)
        x_sample = torch.bernoulli(x_probs)
        return x_sample

# Class Prior
class Prior(nn.Module):
    def __init__(self, z_dim, K=10, device='cuda'):
        super(Prior, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.num_components = K

        # Mixture weights (unnormalized). We'll apply softmax to get valid probabilities.
        self.pi_logits = nn.Parameter(torch.randn(K))

        # Means of the Gaussian components
        self.mu = nn.Parameter(torch.randn(K, z_dim))

        # Log variances of the Gaussian components
        self.log_var = nn.Parameter(torch.randn(K, z_dim))

    def sample(self, batch_size):
        """
        Sample z ~ p(z), where p(z) is a mixture of Gaussians.

        batch_size: int, number of samples to generate

        return:
        z: torch.tensor, with dimensionality (batch_size, z_dim)
        """
        # Compute mixture weights
        pi = F.softmax(self.pi_logits, dim=0)  # Shape: (num_components,)

        # Sample component indices according to the mixture weights
        component_indices = torch.multinomial(pi, batch_size, replacement=True)  # Shape: (batch_size,)

        # Gather the parameters for the selected components
        mu = self.mu[component_indices]           # Shape: (batch_size, z_dim)
        log_var = self.log_var[component_indices] # Shape: (batch_size, z_dim)
        std = torch.exp(0.5 * log_var)            # Shape: (batch_size, z_dim)

        # Sample from the selected Gaussian components
        epsilon = torch.randn(batch_size, self.z_dim).to(self.device)
        z = mu + std * epsilon                    # Shape: (batch_size, z_dim)
        return z

    def forward(self, z):
        """
        Compute the log-probability log p(z), where p(z) is a mixture of Gaussians.

        z: torch.tensor, with dimensionality (batch_size, z_dim)

        return:
        log_p_z: torch.tensor, with dimensionality (batch_size,)
        """

        # Expand z to compute log probabilities under each component
        z = z.unsqueeze(1)  # Shape: (batch_size, 1, z_dim)

        # Compute mixture weights
        pi = F.softmax(self.pi_logits, dim=0)  # Shape: (num_components,)

        # Expand parameters for vectorized computation
        mu = self.mu.unsqueeze(0)              # Shape: (1, num_components, z_dim)
        log_var = self.log_var.unsqueeze(0)    # Shape: (1, num_components, z_dim)
        var = torch.exp(log_var)               # Shape: (1, num_components, z_dim)

        # Compute log probability under each Gaussian component
        # log N(z; mu_k, var_k) = -0.5 * [log(2π) + log var_k + (z - mu_k)^2 / var_k]
        log_prob = -0.5 * (log_var + ((z - mu) ** 2) / var + torch.log(torch.tensor(2.0 * math.pi, device=z.device)))        # Sum over z_dim to get log_prob for each component
        log_prob = torch.sum(log_prob, dim=2)  # Shape: (batch_size, num_components)

        # Add log mixture weights
        log_pi = torch.log(pi + 1e-8)  # Add small value to prevent log(0)
        log_prob += log_pi.unsqueeze(0)  # Shape: (batch_size, num_components)

        # Compute log p(z) = log ∑_k [π_k * N(z; μ_k, Σ_k)] using log-sum-exp trick
        log_p_z = torch.logsumexp(log_prob, dim=1)  # Shape: (batch_size,)

        return log_p_z

# Class PlanarFlow
class PlanarFlow(nn.Module):
    def __init__(self, z_dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(z_dim))
        self.w = nn.Parameter(torch.randn(z_dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        # Compute the linear term
        linear = torch.matmul(z, self.w) + self.b
        # Apply the transformation
        f_z = z + self.u * torch.tanh(linear)
        return f_z

    def inverse(self, z):
        # Inverting planar flow analytically is not straightforward
        # Numerical methods may be required
        # For simplicity, assume inverse is approximated
        raise NotImplementedError("Inverse of PlanarFlow is not implemented.")

    def log_abs_det_jacobian(self, z):
        linear = torch.matmul(z, self.w) + self.b
        psi = (1 - torch.tanh(linear) ** 2) * self.w
        det_jacobian = 1 + torch.matmul(self.u, psi)
        return torch.log(torch.abs(det_jacobian) + 1e-8)

# Class NormalizingFlowPrior
class NormalizingFlowPrior(nn.Module):
    def __init__(self, z_dim, K):
        super(NormalizingFlowPrior, self).__init__()
        self.z_dim = z_dim
        self.K = K
        self.flows = nn.ModuleList([PlanarFlow(z_dim) for _ in range(K)])

    def sample(self, batch_size=32):
        # Sample from base distribution
        z = torch.randn(batch_size, self.z_dim)
        # Apply flows sequentially
        for flow in self.flows:
            z = flow(z)
        return z

    def forward(self, z):
        z_k = z
        # Compute log(2*pi) and move to the correct device
        log_2pi_value = math.log(2 * math.pi)
        log_2pi = torch.tensor(log_2pi_value).to(z_k.device)
        # Compute log probability of the base distribution
        log_p0 = -0.5 * torch.sum(z_k ** 2 + log_2pi, dim=1)
        sum_log_abs_det_jacobians = 0
        # Apply inverse flows and accumulate log determinants
        for flow in reversed(self.flows):
            # Assuming invertibility, apply inverse flow
            z_k = flow(z_k)
            sum_log_abs_det_jacobians += flow.log_abs_det_jacobian(z_k)
        log_p = log_p0 - sum_log_abs_det_jacobians
        return log_p


# Class VAE
class VAE(nn.Module):
    def __init__(self, z_dim, device='cpu'):
        super(VAE, self).__init__()
        self.device = device
        self.z_dim = z_dim
        # Initialize Encoder, Decoder, and Prior
        self.encoder = Encoder(z_dim).to(device)
        self.decoder = Decoder(z_dim).to(device)
        self.prior = Prior(z_dim, K=10).to(device)  # If using normalizing flows

    def sample(self, batch_size):
        z = self.prior.sample(batch_size).to(self.device)
        x_samples = self.decoder.sample(z)
        return x_samples

    def forward(self, x, reduction='mean'):
        batch_size = x.size(0)
        # Encode x to obtain the parameters of q(z|x)
        mu, log_var = self.encoder.encode(x)
        # Sample z from q(z|x)
        z = self.encoder.reparameterization(mu, log_var)
        # Compute log q(z|x)
        log_2pi = torch.log(torch.tensor(2 * torch.pi).to(mu.device))
        log_q_z_given_x = -0.5 * (log_var + (z - mu) ** 2 / torch.exp(log_var) + log_2pi)
        log_q_z_given_x = torch.sum(log_q_z_given_x, dim=1)
        # Compute log p(z)
        log_p_z = self.prior.forward(z)
        # Compute log p(x|z)
        log_p_x_given_z = self.decoder.forward(z, x)
        # Compute Negative ELBO
        NELBO = (log_q_z_given_x - log_p_z) - log_p_x_given_z
        # Apply reduction
        if reduction == 'sum':
            return NELBO.sum()
        else:
            return NELBO.mean()



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the latent dimension
    z_dim = 20

    # Instantiate the model components
    encoder = Encoder(z_dim=z_dim).to(device)
    decoder = Decoder(z_dim=z_dim).to(device)

    # Create a dummy input
    batch_size = 32
    dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)

    # Test the encoder
    mu, log_var = encoder.encode(dummy_input)
    z = encoder.reparameterization(mu, log_var)

    # Test the decoder
    reconstructed_x = decoder.decode(z)

    # Print shapes to verify
    print("Input shape:", dummy_input.shape)
    print("Encoded mu shape:", mu.shape)
    print("Encoded log_var shape:", log_var.shape)
    print("Latent vector z shape:", z.shape)
    print("Reconstructed x shape:", reconstructed_x.shape)





    # Define transformations for training and testing datasets
    transforms_train = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = MNIST('./files/', train=True, download=True,
                        transform=transforms_train
                    )

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(14))

    test_dataset = MNIST('./files/', train=False, download=True,
                        transform=transforms_test
                        )
    #-dataloaders
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #-creating a dir for saving results
    name = 'vae'
    result_dir = "." + '/results/' + name + '/'
    if not(os.path.exists(result_dir)):
        os.mkdir(result_dir)

    #-hyperparams (please do not modify them for the final report)RuntimeError: The size of tensor a (32) must match the size of tensor b (16) at non-singleton dimension 0

    num_epochs = 1000 # max. number of epochs
    max_patience = 20 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped

    model = VAE(z_dim=32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    nll_val = training(name=result_dir + name, max_patience=max_patience, 
                    num_epochs=num_epochs, model=model, optimizer=optimizer,
                    training_loader=train_loader, val_loader=val_loader,
                    shape=(28,28))

    test_loss = evaluation(name=result_dir + name, test_loader=test_loader)
    f = open(result_dir + name + '_test_loss_lr_3e-3.txt', "w")
    f.write(str(test_loss))
    f.close()

    samples_real(result_dir + name, test_loader)
    samples_generated(result_dir + name, test_loader, extra_name='_lr_3e-3')

    plot_curve(result_dir + name, nll_val)



if __name__ == '__main__':
    main()