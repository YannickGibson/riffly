"""Minimal smoke tests for the public model and processing surface.

Kept fast (<5s) so it can run after every change. Confirms:
- Each model class constructs, forwards, and generates on CPU.
- Multi-track decomposition obeys its row-disjoint rule.
- ConvVAE state_dict roundtrips on CPU.
"""
from __future__ import annotations

import numpy as np
import torch

from riffly import models
from riffly.processes import MultiTrackPostprocess, PianoSampler, decompose_three_voices


def test_vae_loads_on_cpu_and_generates():
    model = models.VAE(columns=32, rows=22, hidden_layers=[128], latent_dim=16, dropout=0.0)
    model.eval()
    out = model.generate(threshold=0.5)
    assert isinstance(out, np.ndarray)
    assert out.shape == (22, 32)


def test_convvae_loads_on_cpu_and_generates():
    model = models.ConvVAE(columns=32, rows=22, latent_dim=16, base_channels=16, num_stages=3, dropout=0.0)
    model.eval()
    out = model.generate(threshold=0.5)
    assert out.shape == (22, 32)


def test_transformer_vae_loads_on_cpu_and_generates():
    model = models.TransformerVAE(columns=32, rows=22, d_model=32, nhead=4, num_layers=2, latent_dim=16, dropout=0.0)
    model.eval()
    out = model.generate(threshold=0.5)
    assert out.shape == (22, 32)


def test_decompose_three_voices_default_no_carve():
    m = np.zeros((8, 4), dtype=np.float32)
    for r in (1, 3, 5, 7):
        m[r, 0] = 1.0
    m[4, 1] = 1.0
    m[0, 2] = m[2, 2] = 1.0
    top, mid, bass = decompose_three_voices(m)
    assert bass.sum() == 0
    assert top[7, 0] == 1.0 and top[:, 0].sum() == 1.0
    assert {int(r) for r in np.flatnonzero(mid[:, 0])} == {1, 3, 5}
    assert top[4, 1] == 1.0 and mid[:, 1].sum() == 0
    assert top[2, 2] == 1.0 and mid[0, 2] == 1.0 and mid[:, 2].sum() == 1
    assert top[:, 3].sum() == 0 and mid[:, 3].sum() == 0


def test_decompose_three_voices_carve_bass():
    m = np.zeros((8, 1), dtype=np.float32)
    for r in (1, 3, 5, 7):
        m[r, 0] = 1.0
    top, mid, bass = decompose_three_voices(m, carve_bass=True)
    overlap = (top > 0).astype(int) + (mid > 0).astype(int) + (bass > 0).astype(int)
    assert overlap.max() <= 1
    assert top[7, 0] == 1.0
    assert bass[1, 0] == 1.0
    assert {int(r) for r in np.flatnonzero(mid[:, 0])} == {3, 5}


def test_multi_track_default_808_bass():
    rng = np.random.default_rng(0)
    m = (rng.random((22, 32)) > 0.7).astype(np.float32)
    mt = MultiTrackPostprocess(m, connect_notes=True)
    assert len(mt.midi.instruments) == 3
    progs = sorted(inst.program for inst in mt.midi.instruments)
    assert progs == sorted([80, 0, 33])
    assert len(mt.bass_instrument.notes) > 0
    assert all(n.pitch < 60 for n in mt.bass_instrument.notes)


def test_multi_track_bass_off():
    rng = np.random.default_rng(0)
    m = (rng.random((22, 32)) > 0.7).astype(np.float32)
    mt = MultiTrackPostprocess(m, connect_notes=True, bass_mode="off")
    assert len(mt.bass_instrument.notes) == 0


def test_multi_track_bass_carved_octave_down():
    rng = np.random.default_rng(0)
    m = (rng.random((22, 32)) > 0.7).astype(np.float32)
    mt_raw = MultiTrackPostprocess(m, connect_notes=True, bass_mode="lowest_carved")
    mt_oct = MultiTrackPostprocess(m, connect_notes=True, bass_mode="lowest_carved_octave_down")
    if mt_raw.bass_instrument.notes:
        mins_raw = min(n.pitch for n in mt_raw.bass_instrument.notes)
        mins_oct = min(n.pitch for n in mt_oct.bass_instrument.notes)
        assert mins_oct == mins_raw - 12


def test_piano_sampler_absent_folder_is_safe():
    sampler = PianoSampler("/tmp/__definitely_does_not_exist__")
    assert sampler.available is False


def test_random_note_octave_shift():
    from riffly.transforms import RandomNoteOctaveShift
    torch.manual_seed(0)
    x = torch.zeros(32, 64)
    for r in (3, 9, 15, 21):
        x[r, r] = 1.0
    x_before_count = (x > 0.5).sum().item()
    aug = RandomNoteOctaveShift(max_notes=3, rows_per_octave=7)
    for _ in range(20):
        y = aug(x)
        assert (y > 0.5).sum().item() <= x_before_count
        new_active = ((y > 0.5) & ~(x > 0.5)).nonzero(as_tuple=False)
        for r, c in new_active.tolist():
            orig_rows = (x[:, c] > 0.5).nonzero(as_tuple=True)[0].tolist()
            assert any(abs(r - o) % 7 == 0 for o in orig_rows)


def test_state_dict_roundtrip_cpu():
    rows, cols, latent = 22, 32, 16
    a = models.ConvVAE(columns=cols, rows=rows, latent_dim=latent, base_channels=16, num_stages=3, dropout=0.0)
    sd = a.state_dict()
    b = models.ConvVAE(columns=cols, rows=rows, latent_dim=latent, base_channels=16, num_stages=3, dropout=0.0)
    b.load_state_dict(sd)
    sd_b = b.state_dict()
    assert sd.keys() == sd_b.keys()
    for k in sd:
        assert torch.equal(sd[k], sd_b[k]), f"state_dict mismatch on key {k}"
