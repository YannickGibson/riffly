"""Sample-based piano renderer for the mid (chord) voice.

Loads a folder of piano samples named ``piano_<MIDI>.ogg`` (or ``.wav``) —
one sample per recorded pitch — and renders a list of ``pretty_midi.Note``s
by pitch-shifting the nearest sampled pitch via scipy resample and placing
each at its start time. Same trick the existing 808 path uses
(`MIDIPostprocess._add_808`) so the code is familiar.

Falls back to nothing (returns empty audio) if the sample folder doesn't
exist or is empty — the caller should then use a synth wave instead.
"""
from __future__ import annotations

import os
import re

import numpy as np


class PianoSampler:
    """Loads piano samples once, then renders arbitrary note lists."""

    # piano_<midi>.ogg | .wav
    _FILE_RE = re.compile(r"piano_(\d+)\.(ogg|wav)$")

    def __init__(self, sample_dir: str, sr: int = 44100) -> None:
        self.sample_dir = sample_dir
        self.sr = sr
        self.samples: dict[int, np.ndarray] = {}
        if not os.path.isdir(sample_dir):
            return
        import librosa  # deferred import keeps this module cheap when unused
        for fname in os.listdir(sample_dir):
            m = self._FILE_RE.match(fname)
            if not m:
                continue
            pitch = int(m.group(1))
            data, _ = librosa.load(os.path.join(sample_dir, fname), sr=sr, mono=True)
            self.samples[pitch] = data.astype(np.float64)

    @property
    def available(self) -> bool:
        return bool(self.samples)

    def _nearest_pitch(self, pitch: int) -> int:
        """Return the sampled pitch closest to ``pitch``. Assumes ``self.samples`` is non-empty."""
        keys = list(self.samples.keys())
        return min(keys, key=lambda k: abs(k - pitch))

    def _shift_sample(self, pitch: int) -> np.ndarray:
        """Get the nearest sample and pitch-shift by resampling (same trick as 808 path)."""
        src_pitch = self._nearest_pitch(pitch)
        src = self.samples[src_pitch]
        semitones = pitch - src_pitch
        if semitones == 0:
            return src
        from scipy.signal import resample
        ratio = 2 ** (-semitones / 12)  # lower pitch → stretch longer
        new_len = max(1, int(len(src) * ratio))
        return resample(src, new_len).astype(np.float64)

    def render(self, notes, total_seconds: float | None = None) -> np.ndarray:
        """Render a list of ``pretty_midi.Note`` into a mono waveform.

        Velocity is applied as ``v/127`` gain. If two notes overlap, they sum
        (they decay naturally from the sample's own envelope).
        """
        if not self.available or not notes:
            n = int((total_seconds or 0) * self.sr)
            return np.zeros(n, dtype=np.float64)
        max_end = max(n.end for n in notes)
        if total_seconds is not None:
            max_end = max(max_end, total_seconds)
        out = np.zeros(int(max_end * self.sr) + 1, dtype=np.float64)
        for note in notes:
            shifted = self._shift_sample(int(note.pitch))
            # Truncate the sample to the note's duration so adjacent notes
            # don't bleed indefinitely — the piano sample itself has a
            # natural decay, so a hard cut is fine for chord voicings.
            dur_samples = max(1, int((note.end - note.start) * self.sr))
            shifted = shifted[:dur_samples]
            # Short fade-out to avoid clicks at the cut point.
            fade = min(int(0.02 * self.sr), len(shifted) // 4)
            if fade > 0:
                env = np.linspace(1.0, 0.0, fade)
                shifted[-fade:] *= env
            # Linear velocity scaling: squared was perceptually too quiet for
            # piano-sample chords because the recorded samples already have
            # a natural pp/ff dynamic baked in.
            gain = note.velocity / 127.0
            start = int(note.start * self.sr)
            end = min(start + len(shifted), len(out))
            out[start:end] += shifted[: end - start] * gain
        return out
