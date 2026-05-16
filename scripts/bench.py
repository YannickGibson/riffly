"""CPU inference benchmark for Riffly's three model classes.

Measures, per model:
- Parameter+buffer footprint in MB.
- Mean wallclock per ``model.generate(threshold=0.5)`` call on CPU after warmup.
- Throughput (samples per second).

Also renders a 10-up grid of generated piano rolls per model to ``scripts/out/``
using ``riffly.plots.generate_and_plot``.
"""
from __future__ import annotations

import os
import platform
import sys
import time
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import torch

import riffly as rfl

ROWS = 22
COLUMNS = 32
WARMUP = 5
ITERS = 100
SEED = 42

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)


def model_size_mb(model: torch.nn.Module) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 * 1024)


def time_generate(model: torch.nn.Module, iters: int = ITERS, warmup: int = WARMUP) -> tuple[float, float]:
    for _ in range(warmup):
        model.generate(threshold=0.5)
    t0 = time.perf_counter()
    for _ in range(iters):
        model.generate(threshold=0.5)
    elapsed = time.perf_counter() - t0
    mean_ms = (elapsed / iters) * 1000.0
    samples_per_s = iters / elapsed
    return mean_ms, samples_per_s


def build_models() -> list[tuple[str, Callable[[], torch.nn.Module]]]:
    return [
        (
            "VAE (MLP, hidden=[480], latent=32)",
            lambda: rfl.models.VAE(columns=COLUMNS, rows=ROWS, hidden_layers=[480], latent_dim=32, dropout=0.0),
        ),
        (
            "ConvVAE (3 stages, base=32, latent=32)",
            lambda: rfl.models.ConvVAE(columns=COLUMNS, rows=ROWS, latent_dim=32, base_channels=32, num_stages=3, dropout=0.0),
        ),
        (
            "TransformerVAE (d_model=128, nhead=4, layers=3, latent=32)",
            lambda: rfl.models.TransformerVAE(columns=COLUMNS, rows=ROWS, d_model=128, nhead=4, num_layers=3, latent_dim=32, dropout=0.0),
        ),
    ]


def slug(name: str) -> str:
    return name.split()[0].lower().rstrip(",")


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    print("=" * 78)
    print(f"Riffly CPU benchmark   shape={ROWS}x{COLUMNS}   warmup={WARMUP}   iters={ITERS}")
    print(f"Python {sys.version.split()[0]}   torch {torch.__version__}   threads={torch.get_num_threads()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor() or 'unknown'}")
    print("=" * 78)

    header = f"{'Model':<58}  {'Size (MB)':>10}  {'ms/sample':>10}  {'samples/s':>10}"
    print(header)
    print("-" * len(header))

    rows: list[tuple[str, float, float, float]] = []
    for name, ctor in build_models():
        model = ctor()
        model.eval()
        with torch.inference_mode():
            size = model_size_mb(model)
            ms, sps = time_generate(model)
        rows.append((name, size, ms, sps))
        print(f"{name:<58}  {size:>10.3f}  {ms:>10.2f}  {sps:>10.1f}")

        # Render and save a grid of generated melodies for this model.
        out_path_dir = os.path.join(OUT_DIR, slug(name))
        rfl.plots.generate_and_plot(
            model=model,
            images_to_show=10,
            seed=SEED,
            threshold=0.5,
            save_folder=out_path_dir,
        )
        print(f"    -> samples saved to {out_path_dir}/generated_melodies.png")

    print("-" * len(header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
