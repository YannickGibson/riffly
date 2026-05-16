"""Train the v0 publishable ConvVAE checkpoint on ADL Piano MIDI / Pop.

Architecture is the 32x32 winner recipe from research/training/runs
(`gridsearch_focused_32x32/run_000`). Run names map to filter-loosening
tiers (see ``FILTER_PRESETS``); v1 (default filters) collapsed because the
Pop corpus only yielded 57 training segments. v2 / v3 widen the gates.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from riffly import Riffly


CELLS = 32 * 32  # 32x32 piano roll

FILTER_PRESETS = {
    "default": dict(few_notes_count_threshold=int(CELLS * 0.15),
                    many_notes_threshold=int(CELLS * 0.7),
                    min_unique_pitches=12),
    "loose1":  dict(few_notes_count_threshold=int(CELLS * 0.05),
                    many_notes_threshold=int(CELLS * 0.85),
                    min_unique_pitches=8),
    "loose2":  dict(few_notes_count_threshold=int(CELLS * 0.03),
                    many_notes_threshold=int(CELLS * 0.85),
                    min_unique_pitches=6),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="pop_v2", help="run name (folder under runs/)")
    p.add_argument("--filters", default="loose1", choices=list(FILTER_PRESETS))
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()

    data = os.path.expanduser("~/datasets/adl-piano-midi/Pop")
    save = os.path.abspath(f"runs/{args.run}")
    os.makedirs(save, exist_ok=True)

    model = Riffly(
        "convvae",
        rows=32,
        columns=32,
        latent_dim=128,
        base_channels=48,
        num_stages=3,
        dropout=0.1,
    )

    # Filter-specific preprocessing cache — the loader keys its cache on the
    # data path, not on filter settings, so different presets must not share
    # a cache dir or the second run silently reuses the first's segments.
    cache_dir = os.path.expanduser(f"~/.cache/riffly_pop_{args.filters}")

    t0 = time.time()
    results = model.train(
        data=data,
        epochs=args.epochs,
        batch_size=64,
        lr=1e-3,
        ema_decay=0.999,
        half=True,
        weight_decay=0.0,
        val_split=0.3,
        seed=42,
        save=save,
        cache_dir=cache_dir,
        **FILTER_PRESETS[args.filters],
    )
    elapsed = time.time() - t0

    ckpt = os.path.join(save, "model.pt")
    size_mb = os.path.getsize(ckpt) / (1024 * 1024)
    summary = {
        "best_score": results["best_score"],
        "best_epoch": results["best_epoch"],
        "last_score": results["last_score"],
        "train_segments": results["train_segments"],
        "val_segments": results["val_segments"],
        "epochs": results["epochs"],
        "elapsed_sec": round(elapsed, 1),
        "checkpoint_mb": round(size_mb, 2),
        "checkpoint_path": ckpt,
    }
    with open(os.path.join(save, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
