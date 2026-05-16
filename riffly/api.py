"""High-level facade for Riffly.

``Riffly`` wraps a model, its training pipeline, and its export paths behind a
small Ultralytics-style API::

    from riffly import Riffly

    model = Riffly("convvae")
    model.train(data="datasets/adl-piano-midi", epochs=100)
    model.generate(n=8, save="out/", plot=True)
    model.save("riffly.pt")

    Riffly("riffly.pt").generate(n=4, wav=True, save="more/")

The low-level modules (``riffly.models``, ``riffly.train``, ``riffly.plots``,
``riffly.processes``) remain available for full control.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from riffly import train as _train
from riffly import weights as _weights
from riffly.constants import Wave
from riffly.datasets import MIDIDataset
from riffly.models.conv_vae import ConvVAE
from riffly.models.transformer_vae import TransformerVAE
from riffly.models.vae import VAE
from riffly.processes import InteractivePostprocess, MIDIPostprocess, MultiTrackPostprocess
from riffly.processes.preprocess import OctaveShift
from riffly.utils.losses import vae_loss
from riffly.utils.metrics import pixel_iou

_ARCHITECTURES = {
    "vae": VAE,
    "convvae": ConvVAE,
    "transformer": TransformerVAE,
}

_RIFFLY_VERSION = "0.1.0"


def _f1(y_true, y_pred) -> float:
    """Binary F1 — named so ``riffly.train`` can key validation scores by it."""
    return f1_score(y_true, y_pred, average="binary")


def _default_cache_dir() -> str:
    """Standard per-user location for the preprocessed-MIDI cache.

    Never the current working directory — that would litter the user's project.
    Override with ``$RIFFLY_CACHE``; otherwise ``$XDG_CACHE_HOME/riffly`` or
    ``~/.cache/riffly``.
    """
    override = os.environ.get("RIFFLY_CACHE")
    if override:
        return override
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "riffly")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class Riffly:
    """A trainable, generative melody model with a one-line interface."""

    def __init__(
        self,
        model: str = "convvae",
        *,
        rows: int = 22,
        columns: int = 32,
        **model_kwargs,
    ) -> None:
        if isinstance(model, str) and model in _ARCHITECTURES and not os.path.isfile(model):
            self._build(arch=model, rows=rows, columns=columns, model_kwargs=model_kwargs)
        elif isinstance(model, str) and _weights.is_alias(model) and not os.path.isfile(model):
            self._load(_weights.resolve(model))
        else:
            self._load(str(model))

    # -- construction -----------------------------------------------------

    def _validate_shape(self, rows: int, columns: int) -> None:
        if not _is_power_of_two(columns) or columns > 64:
            msg = (
                f"columns={columns} is invalid: it must be a power of two "
                f"(16, 32, 64) and at most 64, so generated loops repeat cleanly."
            )
            raise ValueError(msg)
        if rows <= 0:
            raise ValueError(f"rows={rows} must be positive.")

    def _build(self, arch: str, rows: int, columns: int, model_kwargs: dict) -> None:
        self._validate_shape(rows, columns)
        self.arch = arch
        self.rows = rows
        self.columns = columns
        self.model_kwargs = dict(model_kwargs)
        self.threshold = 0.5
        self.results: dict | None = None
        self.model = _ARCHITECTURES[arch](columns=columns, rows=rows, **self.model_kwargs)
        self.model.eval()

    def _load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no such file: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "arch" not in ckpt:
            msg = (
                f"{path} is not a Riffly checkpoint. Riffly() loads bundles written "
                f"by Riffly.save(); to load a bare state_dict, use the low-level "
                f"riffly.models API directly."
            )
            raise ValueError(msg)
        self._build(
            arch=ckpt["arch"],
            rows=ckpt["rows"],
            columns=ckpt["columns"],
            model_kwargs=ckpt.get("model_kwargs", {}),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.threshold = ckpt.get("threshold", 0.5)

    # -- training ---------------------------------------------------------

    def train(
        self,
        data: str,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_split: float = 0.3,
        save: str | None = None,
        seed: int = 42,
        device: str | None = None,
        cache_dir: str | None = None,
        ema_decay: float = 0.0,
        weight_decay: float = 0.0,
        half: bool = False,
        **filter_kwargs,
    ) -> dict:
        """Train the model on a folder of ``.mid`` files.

        ``ema_decay`` (e.g. 0.999) enables an exponential moving average of
        the weights; validation runs and the saved checkpoint use the EMA
        weights. ``half`` toggles AMP fp16 training on CUDA. Both are no-ops
        at their defaults so existing callers are unaffected.

        ``filter_kwargs`` overrides dataset quality filters
        (``few_notes_count_threshold``, ``many_notes_threshold``,
        ``min_unique_pitches``). The preprocessed-MIDI cache is written under
        ``cache_dir`` (default: ``~/.cache/riffly``). Returns a dict of
        training results.
        """
        if not os.path.isdir(data):
            raise FileNotFoundError(f"data folder not found: {data}")

        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cells = self.rows * self.columns
        dataset_kwargs = {
            "few_notes_count_threshold": int(cells * 0.15),
            "many_notes_threshold": int(cells * 0.7),
            "min_unique_pitches": 12,
            **filter_kwargs,
        }
        dataset = MIDIDataset(
            path=data,
            columns=self.columns,
            rows=self.rows,
            parent_cache_folder=cache_dir or _default_cache_dir(),
            octave_shift=OctaveShift.DOWN,
            **dataset_kwargs,
        )
        if len(dataset) < 2:
            msg = (
                f"only {len(dataset)} usable segment(s) were extracted from {data!r}. "
                f"Riffly needs quantized, constant-tempo MIDI; expressive performance "
                f"MIDI (with tempo rubato) is filtered out. Try a different corpus or "
                f"loosen the filters via train(..., min_unique_pitches=..., "
                f"few_notes_count_threshold=...)."
            )
            raise ValueError(msg)
        train_ds, val_ds = train_test_split(dataset, test_size=val_split, random_state=seed)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        self.model.to(torch_device)
        losses, val_losses, val_scores, best_score, best_model, best_epoch, last_score = _train.train(
            model=self.model,
            train_dl=train_dl,
            val_dl=val_dl,
            loss_fn=vae_loss,
            optimizer_name="adamw",
            lr=lr,
            epochs=epochs,
            device=torch_device,
            val_metrics=[_f1, pixel_iou],
            val_score_metric=_f1,
            vae=True,
            verbose=False,
            save_folder=save,
            ema_decay=ema_decay,
            weight_decay=weight_decay,
            half=half,
        )
        if best_model is not None:
            self.model = best_model
        self.model.to("cpu").eval()

        self.results = {
            "best_score": best_score,
            "best_epoch": best_epoch,
            "last_score": last_score,
            "train_losses": losses,
            "val_losses": val_losses,
            "val_scores": val_scores,
            "epochs": epochs,
            "train_segments": len(train_ds),
            "val_segments": len(val_ds),
        }
        if save is not None:
            os.makedirs(save, exist_ok=True)
            self.save(os.path.join(save, "model.pt"))
        return self.results

    def val(
        self,
        data: str,
        *,
        batch_size: int = 64,
        seed: int = 42,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> dict:
        """Run validation on a folder of ``.mid`` files; return the score dict."""
        if not os.path.isdir(data):
            raise FileNotFoundError(f"data folder not found: {data}")
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cells = self.rows * self.columns
        dataset = MIDIDataset(
            path=data,
            columns=self.columns,
            rows=self.rows,
            parent_cache_folder=cache_dir or _default_cache_dir(),
            octave_shift=OctaveShift.DOWN,
            few_notes_count_threshold=int(cells * 0.15),
            many_notes_threshold=int(cells * 0.7),
            min_unique_pitches=12,
        )
        val_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.to(torch_device)
        scores = _train.validation_loop(
            model=self.model,
            val_dl=val_dl,
            val_metrics=[_f1, pixel_iou],
            loss_fn=vae_loss,
            device=torch_device,
            vae=True,
            verbose=False,
        )[0]
        self.model.to("cpu")
        return scores

    # -- generation -------------------------------------------------------

    def generate(
        self,
        n: int = 8,
        *,
        threshold: float | None = None,
        seed: int | None = None,
        save: str | None = None,
        midi: bool = True,
        wav: bool = True,
        png: bool = True,
        multi_track: bool = False,
        max_empty_retries: int = 10,
    ) -> list[np.ndarray]:
        """Sample ``n`` new melodies.

        Returns the list of generated piano-roll arrays. When ``save`` is a
        folder, every melody is written there in each enabled format:
        ``melody_i.mid``, ``melody_i.wav`` and ``melody_i.png``. Opt out of a
        format with ``midi=False``, ``wav=False`` or ``png=False``.

        With ``multi_track=True`` the ``.mid`` and ``.wav`` are split into
        three voices (melody / chord / 808 bass) instead of a single track.

        To display a melody on screen instead of saving it, use
        ``riffly.plot``.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        tau = self.threshold if threshold is None else threshold

        self.model.eval()
        rolls: list[np.ndarray] = []
        for _ in range(n):
            roll = np.asarray(self.model.generate(threshold=tau))
            attempts = 0
            while roll.sum() == 0 and attempts < max_empty_retries:
                roll = np.asarray(self.model.generate(threshold=tau))
                attempts += 1
            rolls.append(roll)

        if save is not None:
            os.makedirs(save, exist_ok=True)
            for i, roll in enumerate(rolls):
                mat = roll.astype(float)
                mid_path = os.path.join(save, f"melody_{i}.mid")
                wav_path = os.path.join(save, f"melody_{i}.wav")
                mt = MultiTrackPostprocess(mat, connect_notes=True) if multi_track else None

                if midi:
                    if mt is not None:
                        mt.export_midi(mid_path)
                    else:
                        MIDIPostprocess(mat, connect_notes=True).midi.write(mid_path)
                if wav:
                    try:
                        if mt is not None:
                            mt.export_wav(wav_path)
                        else:
                            InteractivePostprocess(mat, connect_notes=True).export_beat(
                                path=wav_path,
                                wave=Wave.SQUARE,
                                repeat=2,
                                time_multiplier=1,
                                clap_interval=None,
                                hat_interval=None,
                            )
                    except ImportError as e:
                        raise ImportError(
                            "wav export needs the 'interactive' extra: "
                            "pip install 'riffly[interactive]' (or pass wav=False).",
                        ) from e
                if png:
                    try:
                        self._save_png(roll, os.path.join(save, f"melody_{i}.png"))
                    except ImportError as e:
                        raise ImportError(
                            "png export needs the 'interactive' extra: "
                            "pip install 'riffly[interactive]' (or pass png=False).",
                        ) from e

        return rolls

    @staticmethod
    def _save_png(roll: np.ndarray, path: str) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(roll, interpolation="none", cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title("Generated Melody")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> str:
        """Save a self-contained checkpoint (config + weights) to ``path``."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "arch": self.arch,
                "rows": self.rows,
                "columns": self.columns,
                "model_kwargs": self.model_kwargs,
                "state_dict": self.model.state_dict(),
                "threshold": self.threshold,
                "riffly_version": _RIFFLY_VERSION,
            },
            path,
        )
        return path

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters())
        return (
            f"Riffly(arch={self.arch!r}, shape={self.rows}x{self.columns}, "
            f"params={n_params:,}, threshold={self.threshold})"
        )
