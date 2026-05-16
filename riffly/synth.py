"""Procedural drum synthesizers — no audio samples shipped.

The TR-808 is analog synthesis: a pitched sine with exponential pitch and
amplitude envelopes. A hi-hat is filtered noise with a very short decay. A
clap is a sequence of short noise bursts through a bandpass. All output here
is generated from math — there are no bundled ``.wav`` files, which keeps the
whole package MIT-clean.
"""
from __future__ import annotations

import numpy as np
import scipy.signal

from riffly.constants import Sounds808


def _midi_to_hz(midi_pitch: float) -> float:
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))


def synth_808(
    midi_pitch: int = 36,
    duration_sec: float = 0.5,
    sound: Sounds808 = Sounds808.PUNCHY,
    sr: int = 44100,
) -> np.ndarray:
    """Synthesize one 808-style bass note at ``midi_pitch`` for ``duration_sec``.

    PUNCHY adds a brief noise click at the attack and soft-clips the body for
    a more aggressive timbre. SMOOTH stays clean with a longer tail.
    """
    n = int(sr * duration_sec)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / sr
    f0 = _midi_to_hz(midi_pitch)

    if sound == Sounds808.PUNCHY:
        pitch_mod, pitch_tau, amp_tau, drive, click_amp = 4.0, 0.025, max(duration_sec * 0.6, 0.15), 1.8, 0.4
    elif sound == Sounds808.SMOOTH:
        pitch_mod, pitch_tau, amp_tau, drive, click_amp = 1.5, 0.06, max(duration_sec * 0.9, 0.30), 1.0, 0.0
    else:
        raise ValueError(f"Unknown 808 sound: {sound!r}")

    freq = f0 * (1.0 + (pitch_mod - 1.0) * np.exp(-t / pitch_tau))
    phase = 2 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase)
    if drive != 1.0:
        body = np.tanh(body * drive)

    amp_env = np.exp(-t / amp_tau)
    attack_n = min(n, max(1, int(0.002 * sr)))
    amp_env[:attack_n] *= np.linspace(0.0, 1.0, attack_n)
    audio = body * amp_env

    if click_amp > 0.0:
        click_n = min(n, int(0.003 * sr))
        if click_n > 0:
            click = np.random.default_rng(0).standard_normal(click_n)
            click *= np.exp(-np.arange(click_n) / max(click_n / 3, 1))
            audio[:click_n] += click * click_amp

    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / peak * 0.9
    return audio.astype(np.float32)


def synth_hat(duration_sec: float = 0.04, sr: int = 44100) -> np.ndarray:
    """Synthesize a closed hi-hat: high-passed noise with a very short decay."""
    n = int(sr * duration_sec)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    noise = np.random.default_rng(1).standard_normal(n)
    sos = scipy.signal.butter(4, 7000.0, btype="highpass", fs=sr, output="sos")
    filtered = scipy.signal.sosfilt(sos, noise)
    env = np.exp(-(np.arange(n) / sr) / 0.012)
    audio = filtered * env
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / peak * 0.7
    return audio.astype(np.float32)


def synth_clap(sr: int = 44100) -> np.ndarray:
    """Synthesize a clap: four short bandpassed noise bursts ~10 ms apart."""
    rng = np.random.default_rng(2)
    burst_n = max(1, int(0.005 * sr))
    gap_n = max(1, int(0.010 * sr))
    tail_n = int(0.18 * sr)
    total_n = 3 * gap_n + burst_n + tail_n
    out = np.zeros(total_n, dtype=np.float64)
    for i in range(4):
        start = i * gap_n
        amp = 1.0 if i == 3 else 0.7
        b = rng.standard_normal(burst_n) * amp
        b *= np.exp(-np.arange(burst_n) / max(burst_n / 3, 1))
        out[start : start + burst_n] += b
    sos = scipy.signal.butter(4, [1000.0, 3000.0], btype="bandpass", fs=sr, output="sos")
    out = scipy.signal.sosfilt(sos, out)
    out *= np.exp(-(np.arange(total_n) / sr) / 0.08)
    peak = float(np.abs(out).max())
    if peak > 0:
        out = out / peak * 0.8
    return out.astype(np.float32)
