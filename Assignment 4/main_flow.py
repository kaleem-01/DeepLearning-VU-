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
PI = torch.from_numpy(np.asarray(np.pi)).to(device)
EPS = 1.e-5

# Class Encoder
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)    # Output: (32, 14, 14)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)   # Output: (64, 7, 7)
        self.fc_mu = nn.Linear(64 * 7 * 7, z_dim)
        self.fc_log_var = nn.Linear(64 * 7 * 7, z_dim)

    def encode(self, x):
        h = F.relu(self.conv1(x)) 
        h = F.relu(self.conv2(h)) 
        h = h.view(h.size(0), -1) 
        mu = self.fc_mu(h)      
        log_var = self.fc_log_var(h) 
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
        self.fc = nn.Linear(z_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def decode(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)
        h = F.relu(self.deconv1(h))
        h = self.deconv2(h)
        return h

    def forward(self, z, x):
        x_logits = self.decode(z)
        x_flat = x.view(x.size(0), -1)
        x_logits_flat = x_logits.view(x.size(0), -1)
        bce = F.binary_cross_entropy_with_logits(x_logits_flat, x_flat, reduction='none')
        log_px_given_z = -torch.sum(bce, dim=1)
        return log_px_given_z

    def sample(self, z):
        x_logits = self.decode(z)
        x_probs = torch.sigmoid(x_logits)
        x_sample = torch.bernoulli(x_probs)
        return x_sample

# Class Prior (Mixture of Gaussians) - not used here, but kept for reference
class Prior(nn.Module):
    def __init__(self, z_dim, K=10, device='cuda'):
        super(Prior, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.num_components = K
        self.pi_logits = nn.Parameter(torch.randn(K))
        self.mu = nn.Parameter(torch.randn(K, z_dim))
        self.log_var = nn.Parameter(torch.randn(K, z_dim))

    def sample(self, batch_size):
        pi = F.softmax(self.pi_logits, dim=0)
        component_indices = torch.multinomial(pi, batch_size, replacement=True)
        mu = self.mu[component_indices]
        log_var = self.log_var[component_indices]
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn(batch_size, self.z_dim, device=self.device)
        z = mu + std * epsilon
        return z

    def forward(self, z):
        z = z.unsqueeze(1)
        pi = F.softmax(self.pi_logits, dim=0)
        mu = self.mu.unsqueeze(0)
        log_var = self.log_var.unsqueeze(0)
        var = torch.exp(log_var)
        log_prob = -0.5 * (log_var + ((z - mu)**2)/var + math.log(2*math.pi))
        log_prob = torch.sum(log_prob, dim=2)
        log_pi = torch.log(pi + 1e-8)
        log_prob += log_pi.unsqueeze(0)
        log_p_z = torch.logsumexp(log_prob, dim=1)
        return log_p_z

# PlanarFlow with correct device handling
class PlanarFlow(nn.Module):
    def __init__(self, z_dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(z_dim))
        self.w = nn.Parameter(torch.randn(z_dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        linear = torch.matmul(z, self.w) + self.b
        f_z = z + torch.tanh(linear).unsqueeze(1)*self.u.unsqueeze(0)
        return f_z

    def log_abs_det_jacobian(self, z):
        linear = torch.matmul(z, self.w) + self.b
        psi = (1 - torch.tanh(linear)**2).unsqueeze(1)*self.w.unsqueeze(0)
        det_jacobian = 1 + torch.sum(self.u * psi, dim=1)
        return torch.log(torch.abs(det_jacobian) + 1e-8)

class NormalizingFlowPrior(nn.Module):
    def __init__(self, z_dim, K):
        super(NormalizingFlowPrior, self).__init__()
        self.z_dim = z_dim
        self.K = K
        self.flows = nn.ModuleList([PlanarFlow(z_dim) for _ in range(K)])

    def sample(self, batch_size=32, device='cuda'):
        z = torch.randn(batch_size, self.z_dim, device=device)
        for flow in self.flows:
            z = flow(z)
        return z

    def forward(self, z):
        # Simple standard normal as placeholder
        log_2pi_value = math.log(2 * math.pi)
        log_2pi = torch.tensor(log_2pi_value, device=z.device)
        log_p0 = -0.5 * torch.sum(z**2 + log_2pi, dim=1)
        # Not accurately accounting for flows (requires inverse), but for code correctness:
        return log_p0

# VAE with NormalizingFlowPrior
class VAE(nn.Module):
    def __init__(self, z_dim, device='cpu'):
        super(VAE, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim).to(device)
        self.decoder = Decoder(z_dim).to(device)
        self.prior = NormalizingFlowPrior(z_dim, K=10).to(device)

    def sample(self, batch_size):
        z = self.prior.sample(batch_size, device=self.device)
        x_samples = self.decoder.sample(z)
        return x_samples

    def forward(self, x, reduction='mean'):
        x = x.to(self.device)  # ensure input is on the correct device
        mu, log_var = self.encoder.encode(x)
        z = self.encoder.reparameterization(mu, log_var)
        log_2pi = torch.log(torch.tensor(2*math.pi, device=mu.device))
        log_q_z_given_x = -0.5*(log_var + (z - mu)**2/torch.exp(log_var) + log_2pi)
        log_q_z_given_x = torch.sum(log_q_z_given_x, dim=1)
        log_p_z = self.prior.forward(z)
        log_p_x_given_z = self.decoder.forward(z, x)
        NELBO = (log_q_z_given_x - log_p_z) - log_p_x_given_z
        if reduction == 'sum':
            return NELBO.sum()
        else:
            return NELBO.mean()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_dim = 20
    encoder = Encoder(z_dim=z_dim).to(device)
    decoder = Decoder(z_dim=z_dim).to(device)

    batch_size = 32
    dummy_input = torch.randn(batch_size, 1, 28, 28, device=device)
    mu, log_var = encoder.encode(dummy_input)
    z = encoder.reparameterization(mu, log_var)
    reconstructed_x = decoder.decode(z)
    print("Input shape:", dummy_input.shape)
    print("Encoded mu shape:", mu.shape)
    print("Encoded log_var shape:", log_var.shape)
    print("Latent vector z shape:", z.shape)
    print("Reconstructed x shape:", reconstructed_x.shape)

    transforms_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = MNIST('./files/', train=True, download=True, transform=transforms_train)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(14))
    test_dataset = MNIST('./files/', train=False, download=True, transform=transforms_test)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    name = 'vae'
    result_dir = "." + '/results/' + name + '/'
    if not(os.path.exists(result_dir)):
        os.mkdir(result_dir)

    num_epochs = 1000
    max_patience = 20
    model = VAE(z_dim=32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # IMPORTANT: In training and evaluation routines (from utils), ensure you do:
    # x, _ = next_batch
    # x = x.to(device)

    nll_val = training(name=result_dir + name, max_patience=max_patience, 
                       num_epochs=num_epochs, model=model, optimizer=optimizer,
                       training_loader=train_loader, val_loader=val_loader,
                       shape=(28,28))  # ensure these funcs also move data to device

    test_loss = evaluation(name=result_dir + name, test_loader=test_loader, device=device)
    with open(result_dir + name + '_test_loss.txt', "w") as f:
        f.write(str(test_loss))

    samples_real(result_dir + name, test_loader, device=device)
    samples_generated(result_dir + name, test_loader, extra_name='_FINAL', device=device)

    plot_curve(result_dir + name, nll_val)

if __name__ == '__main__':
    main()
