import numpy as np
import torch
from torch import nn

from riffly.models.general import ModelInterface


class VAE(ModelInterface):
    def __init__(self, columns, rows, hidden_layers: list[int] | None = None, latent_dim=32, dropout=0.0) -> None:
        super().__init__(columns, rows)

        # self.fc1 = nn.Linear(columns * rows, hidden_dim)
        # self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # self.fc2 = nn.Linear(latent_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, columns * rows)

        if hidden_layers is None or len(hidden_layers) == 0:
            hidden_layers = [256]

        self.dropout = nn.Dropout(p=dropout) 

        self.encode_fcs = nn.ModuleList([nn.Linear(columns * rows, hidden_layers[0])])
        self.decode_fcs = nn.ModuleList([nn.Linear(hidden_layers[0], columns * rows)])
        for i in range(1, len(hidden_layers)):
            self.encode_fcs.append(
                nn.Linear(in_features=hidden_layers[i - 1], out_features=hidden_layers[i]),
            )
            self.decode_fcs.insert(
                0,
                nn.Linear(in_features=hidden_layers[i], out_features=hidden_layers[i - 1]),
            )
        # mu and logvar handles conversion to latent_dim
        # self.encode_fcs.append(nn.Linear(hidden_layers[-1], latent_dim))

        # Latent -> first decoder layer
        self.decode_fcs.insert(0, nn.Linear(latent_dim, hidden_layers[-1]))

        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[-1], latent_dim)

        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim

    def encode(self, x):
        for layer in self.encode_fcs:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        for layer in self.decode_fcs[:-1]:
            z = torch.relu(layer(z))
            z = self.dropout(z)
        z = self.decode_fcs[-1](z)  # no relu
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def _generate(self) -> np.ndarray:
        device = next(self.parameters()).device
        z = torch.randn(1, self.latent_dim, device=device)
        matrix = self.decode(z).view(self.rows, self.columns)
        return matrix.detach().cpu().numpy()


class ConvVAE(ModelInterface):
    def __init__(self, columns, rows, latent_dim) -> None:
        super().__init__(columns, rows)

        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.mu = nn.Linear(16 * columns * rows, latent_dim)
        self.logvar = nn.Linear(16 * columns * rows, latent_dim)
        self.fc = nn.Linear(latent_dim, 16 * columns * rows)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def encode(self, x):
        x = x.reshape(-1, 1, self.columns, self.rows)
        h = torch.relu(self.conv1(x))
        h = h.view(-1, 16 * self.columns * self.rows)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc(z))
        h = h.view(-1, 16, self.columns, self.rows)
        return torch.sigmoid(self.deconv1(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class SimpleVAE(ModelInterface):
    def __init__(self, columns, rows, hidden_dim=128, latent_dim=32) -> None:
        super().__init__(columns, rows)

        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(columns * rows, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, columns * rows)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
