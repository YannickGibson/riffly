"""Transformer-based VAE for piano-roll inputs.

Treats the piano roll as a time sequence: each of the ``columns`` time-steps
is a ``rows``-dim binary vector, projected to ``d_model`` and passed through
a pre-norm Transformer encoder stack. A linear head projects the flattened
encoder output to ``mu``/``logvar``. The decoder expands ``z`` back into
``(columns, d_model)`` tokens, runs another Transformer stack, and projects
each token to ``rows`` binary probabilities.

Forward I/O shape matches ``VAE`` / ``ConvVAE`` (flattened
``(B, rows*columns)`` in and out) so existing train/eval loops apply
unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from riffly.models.general import ModelInterface


class TransformerVAE(ModelInterface):
    def __init__(
        self,
        columns: int,
        rows: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int | None = None,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(columns, rows)

        if d_model % nhead != 0:
            msg = f"d_model={d_model} must be divisible by nhead={nhead}"
            raise ValueError(msg)
        ff = dim_feedforward if dim_feedforward is not None else d_model * 4

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        # Published so ``train.train()`` and Neptune param logging don't need special-casing.
        self.hidden_layers = [d_model] * num_layers

        # Time-axis sequence: embed each column's ``rows``-dim activity into ``d_model``.
        self.in_proj = nn.Linear(rows, d_model)
        self.pos_emb = nn.Parameter(torch.randn(columns, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.enc_norm = nn.LayerNorm(d_model)

        flat = d_model * columns
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_up = nn.Linear(latent_dim, flat)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.dec_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, rows)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, rows*cols) -> (B, cols, rows)
        x = x.view(-1, self.rows, self.columns).transpose(1, 2).float()
        h = self.in_proj(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        h = self.enc_norm(h).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_up(z).view(-1, self.columns, self.d_model)
        h = h + self.pos_emb.unsqueeze(0)
        h = self.decoder(h)
        h = self.dec_norm(h)
        out = self.out_proj(h)  # (B, cols, rows)
        out = out.transpose(1, 2)  # (B, rows, cols)
        return torch.sigmoid(out).reshape(-1, self.rows * self.columns)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def _generate(self) -> np.ndarray:
        device = next(self.parameters()).device
        z = torch.randn(1, self.latent_dim, device=device)
        matrix = self.decode(z).view(self.rows, self.columns)
        return matrix.detach().cpu().numpy()
