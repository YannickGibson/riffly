"""2D-convolutional VAE for piano-roll inputs.

Encoder: N stride-2 Conv2d blocks (Conv -> BN -> ReLU -> Dropout2d) followed
by FC heads for mu/logvar. Decoder mirrors with ConvTranspose2d blocks.
Forward I/O shape matches ``riffly.models.vae.VAE`` — flattened
``(B, rows*columns)`` — so existing train/eval loops apply unchanged.

Depth, channel width, latent size and dropout are all configurable. Default
is a 3-stage network (32 -> 64 -> 128) with a 32-dim latent, sized for
22x32 or 15x32 piano rolls.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from riffly.models.general import ModelInterface


def _down(d: int) -> int:
    return (d + 1) // 2


class ConvVAE(ModelInterface):
    def __init__(
        self,
        columns: int,
        rows: int,
        latent_dim: int = 32,
        base_channels: int = 32,
        num_stages: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(columns, rows)

        channels = [base_channels * (2 ** i) for i in range(num_stages)]
        shapes: list[tuple[int, int]] = [(rows, columns)]
        for _ in range(num_stages):
            h, w = shapes[-1]
            shapes.append((_down(h), _down(w)))

        enc_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            enc_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        bottleneck_h, bottleneck_w = shapes[-1]
        flat = channels[-1] * bottleneck_h * bottleneck_w
        self._bottleneck_ch = channels[-1]
        self._bottleneck_h = bottleneck_h
        self._bottleneck_w = bottleneck_w

        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_up = nn.Linear(latent_dim, flat)

        dec_layers: list[nn.Module] = []
        for stage in range(num_stages):
            src_h, src_w = shapes[num_stages - stage]
            tgt_h, tgt_w = shapes[num_stages - stage - 1]
            op = (tgt_h - (src_h * 2 - 1), tgt_w - (src_w * 2 - 1))
            in_ch = channels[num_stages - stage - 1]
            out_ch = channels[num_stages - stage - 2] if stage < num_stages - 1 else 1
            dec_layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=op),
            )
            if stage < num_stages - 1:
                dec_layers += [
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout),
                ]
        self.decoder = nn.Sequential(*dec_layers)

        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.hidden_layers = channels

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, 1, self.rows, self.columns)
        h = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_up(z).view(-1, self._bottleneck_ch, self._bottleneck_h, self._bottleneck_w)
        out = self.decoder(h)
        return torch.sigmoid(out).view(-1, self.rows * self.columns)

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
