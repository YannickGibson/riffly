"""Useful loss functions for training models."""
from __future__ import annotations

import torch
from torch import nn


def vae_loss(recon, target, mu, logvar):
    recon = recon.float().clamp(1e-7, 1 - 1e-7)
    target = target.float()
    mu = mu.float()
    logvar = logvar.float()
    recon_loss = nn.functional.binary_cross_entropy(recon, target, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    alpha = 0.9  # 0.9 is a good value
    return alpha * recon_loss + kl_loss


def focal_loss_probs(probs, target, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss on sigmoid probabilities (sum-reduced to match BCE scale).

    Accepts probabilities in [0, 1] (the VAE decoder already applies sigmoid),
    so we avoid needing to change the model output path.
    """
    probs = probs.clamp(1e-7, 1 - 1e-7).float()
    target = target.float()
    p_t = target * probs + (1 - target) * (1 - probs)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    fl = -alpha_t * (1 - p_t) ** gamma * torch.log(p_t)
    return fl.sum()


def focal_vae_loss(recon, target, mu, logvar, alpha: float = 0.25, gamma: float = 2.0):
    recon_loss = focal_loss_probs(recon, target, alpha=alpha, gamma=gamma)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


class BetaAnnealedVAELoss:
    """β-annealed VAE loss: recon_w · BCE + β(epoch) · KL.

    β ramps linearly from 0 to ``beta_max`` over ``warmup_epochs`` epochs,
    then stays at ``beta_max``. Train loop calls ``set_epoch(e)`` before
    each epoch; loss is called with the usual ``(recon, target, mu, logvar)``.
    """

    def __init__(
        self,
        beta_max: float = 1.0,
        warmup_epochs: int = 20,
        recon_weight: float = 0.9,
        recon_loss: str = "bce",  # "bce" | "focal"
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        self.beta_max = beta_max
        self.warmup_epochs = max(1, warmup_epochs)
        self.recon_weight = recon_weight
        self.recon_loss = recon_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @property
    def current_beta(self) -> float:
        frac = min(1.0, (self.epoch + 1) / self.warmup_epochs)
        return self.beta_max * frac

    def __call__(self, recon, target, mu, logvar):
        recon = recon.float().clamp(1e-7, 1 - 1e-7)
        target = target.float()
        if self.recon_loss == "focal":
            r = focal_loss_probs(recon, target, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            r = nn.functional.binary_cross_entropy(recon, target, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.recon_weight * r + self.current_beta * kl

    def __repr__(self) -> str:
        return (
            f"BetaAnnealedVAELoss(beta_max={self.beta_max}, warmup_epochs={self.warmup_epochs}, "
            f"recon_weight={self.recon_weight}, recon_loss={self.recon_loss})"
        )
