"""Load the two trained ConvVAE checkpoints from gpu_runs/!1_noice/ and render
generated-melody grids using ``riffly.plots.generate_and_plot``.

If either checkpoint fails to load (e.g. shape mismatch, missing keys), the
script aborts immediately rather than silently rendering a half-broken result.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

import riffly as rfl

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

RUNS_ROOT = "/home/user/me/code/melody_generator/gpu_runs/!1_noice"


@dataclass(frozen=True)
class Checkpoint:
    label: str
    weights_path: str
    rows: int
    columns: int
    latent_dim: int
    base_channels: int
    num_stages: int
    dropout: float
    threshold: float


CHECKPOINTS = [
    Checkpoint(
        label="gridsearch_run16",
        weights_path=f"{RUNS_ROOT}/gridsearch_to_080__run16/weights/final_f10.764.pth",
        rows=22, columns=32,
        latent_dim=128, base_channels=48, num_stages=3, dropout=0.0,
        threshold=0.34,
    ),
    Checkpoint(
        label="ike_loose_e1000",
        weights_path=f"{RUNS_ROOT}/ike_loose_v1_lr1e5_s0_e1000/weights/final_f10.607.pth",
        rows=32, columns=32,
        latent_dim=128, base_channels=48, num_stages=3, dropout=0.1,
        threshold=0.36,
    ),
]


def load(ckpt: Checkpoint) -> rfl.models.ConvVAE:
    if not os.path.isfile(ckpt.weights_path):
        raise FileNotFoundError(f"weights not found: {ckpt.weights_path}")
    model = rfl.models.ConvVAE(
        columns=ckpt.columns, rows=ckpt.rows,
        latent_dim=ckpt.latent_dim, base_channels=ckpt.base_channels,
        num_stages=ckpt.num_stages, dropout=ckpt.dropout,
    )
    state = torch.load(ckpt.weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch for {ckpt.label}: "
            f"missing={list(missing)}, unexpected={list(unexpected)}",
        )
    model.eval()
    return model


def main() -> int:
    torch.manual_seed(1)
    np.random.seed(1)

    loaded: list[tuple[Checkpoint, rfl.models.ConvVAE]] = []
    for ckpt in CHECKPOINTS:
        print(f"loading {ckpt.label}  ({ckpt.rows}x{ckpt.columns}, tau={ckpt.threshold})")
        try:
            model = load(ckpt)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  loaded OK   params={n_params:,}")
        loaded.append((ckpt, model))

    for ckpt, model in loaded:
        out_dir = os.path.join(ASSETS_DIR, ckpt.label)
        rfl.plots.generate_and_plot(
            model=model,
            images_to_show=10,
            seed=1,
            threshold=ckpt.threshold,
            save_folder=out_dir,
        )
        print(f"rendered {out_dir}/generated_melodies.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
