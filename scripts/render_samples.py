"""Render the README "Sample output" preview assets.

For each sample MIDI under ``assets/preview/`` this writes a matching ``.png``
piano roll and ``.wav`` rendering. Both are produced *directly from the MIDI*
(via ``pretty_midi``) so they faithfully reflect the file — no key
normalisation, octave shifting or grid quantisation.

Run once::

    python scripts/render_samples.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
from scipy.io import wavfile

AUDIO_FS = 44100   # wav sample rate
ROLL_FS = 100      # piano-roll columns per second
PEAK = 0.4         # wav loudness — normalised peak amplitude (0..1)
LOOPS = 4          # how many times the loop repeats in the wav
PREVIEW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "assets", "preview")

SAMPLES = ("sample_left", "sample_right")


def render_wav(pm: pretty_midi.PrettyMIDI, path: str) -> None:
    """Synthesize the MIDI to a WAV, faithful to its notes and timing.

    The synthesized audio is trimmed to the loop's musical length
    (``get_end_time()``, dropping the note-release tail) and tiled ``LOOPS``
    times so the riff repeats seamlessly.
    """
    audio = pm.synthesize(fs=AUDIO_FS)
    loop_len = int(round(pm.get_end_time() * AUDIO_FS))
    audio = np.tile(audio[:loop_len], LOOPS)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * PEAK
    wavfile.write(path, AUDIO_FS, audio.astype(np.float32))


def render_png(pm: pretty_midi.PrettyMIDI, path: str) -> None:
    """Plot the MIDI's true piano roll (128 pitches), cropped to active range."""
    roll = pm.get_piano_roll(fs=ROLL_FS)  # (128, time)
    active = np.where(roll.sum(axis=1) > 0)[0]
    lo = max(active.min() - 2, 0)
    hi = min(active.max() + 3, 128)
    crop = roll[lo:hi] > 0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(crop, interpolation="none", cmap="gray", origin="lower", aspect="auto")
    ax.set_title("Generated Melody")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for name in SAMPLES:
        pm = pretty_midi.PrettyMIDI(os.path.join(PREVIEW_DIR, f"{name}.mid"))
        png_path = os.path.join(PREVIEW_DIR, f"{name}.png")
        wav_path = os.path.join(PREVIEW_DIR, f"{name}.wav")
        render_png(pm, png_path)
        render_wav(pm, wav_path)
        print(f"{name}: wrote {png_path} and {wav_path}")


if __name__ == "__main__":
    main()
