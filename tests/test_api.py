"""Smoke tests for the high-level ``Riffly`` facade.

No training here (training needs a MIDI corpus); these cover construction,
generation, file export, and checkpoint round-trips.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from riffly import Riffly


def test_construct_each_arch():
    for arch in ("vae", "convvae", "transformer"):
        model = Riffly(arch, rows=22, columns=32)
        assert model.arch == arch
        assert (model.rows, model.columns) == (22, 32)


def test_generate_returns_arrays():
    model = Riffly("convvae", rows=22, columns=32)
    rolls = model.generate(n=3, seed=0)
    assert len(rolls) == 3
    for roll in rolls:
        assert isinstance(roll, np.ndarray)
        assert roll.shape == (22, 32)


def test_generate_save_writes_all_formats(tmp_path):
    model = Riffly("convvae", rows=22, columns=32)
    model.generate(n=3, save=str(tmp_path), seed=0)
    assert len(list(tmp_path.glob("*.mid"))) == 3
    assert len(list(tmp_path.glob("*.wav"))) == 3
    assert len(list(tmp_path.glob("*.png"))) == 3


def test_generate_can_opt_out_of_formats(tmp_path):
    model = Riffly("convvae", rows=22, columns=32)
    model.generate(n=2, save=str(tmp_path), wav=False, png=False, seed=0)
    assert len(list(tmp_path.glob("*.mid"))) == 2
    assert len(list(tmp_path.glob("*.wav"))) == 0
    assert len(list(tmp_path.glob("*.png"))) == 0


def test_generate_multi_track_writes_three_voices(tmp_path):
    import pretty_midi

    model = Riffly("convvae", rows=22, columns=32)
    model.generate(n=1, save=str(tmp_path), multi_track=True, png=False, seed=0)
    mid_file = next(tmp_path.glob("*.mid"))
    midi = pretty_midi.PrettyMIDI(str(mid_file))
    assert len(midi.instruments) == 3
    assert {inst.name for inst in midi.instruments} == {"top", "mid", "bass"}


def test_save_load_roundtrip(tmp_path):
    model = Riffly("convvae", rows=22, columns=32, latent_dim=16, base_channels=16)
    model.threshold = 0.42
    path = model.save(str(tmp_path / "model.pt"))

    reloaded = Riffly(path)
    assert reloaded.arch == model.arch
    assert (reloaded.rows, reloaded.columns) == (model.rows, model.columns)
    assert reloaded.threshold == 0.42

    sd_a = model.model.state_dict()
    sd_b = reloaded.model.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"weight mismatch on {k}"


def test_invalid_columns_rejected():
    with pytest.raises(ValueError):
        Riffly("convvae", rows=22, columns=96)  # not a power of two, and > 64
    with pytest.raises(ValueError):
        Riffly("convvae", rows=22, columns=128)  # power of two but > 64


def test_loading_non_checkpoint_raises(tmp_path):
    bad = tmp_path / "bare.pt"
    torch.save({"just": "a dict"}, bad)
    with pytest.raises(ValueError):
        Riffly(str(bad))
