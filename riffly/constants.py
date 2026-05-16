"""Constants used across the project."""

from enum import Enum

import numpy as np
import scipy

# Constants
MAJOR_SCALE: list[int] = [0, 2, 4, 5, 7, 9, 11]
MAJOR_SCALE_DIFF: list[int] = [2, 2, 1, 2, 2, 2, 1]
MINOR_SCALE: list[int] = [0, 2, 3, 5, 7, 8, 10]
MINOR_SCALE_DIFF: list[int] = [2, 1, 2, 2, 1, 2, 2]
KEYS: list[int] = list(range(12))


def __get_scales() -> dict[int, list[int]]:
    """Returns a dict where key is scale root and value is list of all numerical pitches of notes in that scale.

    For example:
    {
        0: [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, ...],  # C minor scale
        1: [1, 3, 4, 6, 8, 9, 11, 13, 15, 16, ...],  # C# minor scale
        ...
    }
    """
    scales: dict[int, list[int]] = {i: [] for i in range(12)}
    for key in KEYS:
        pitch = key
        note_index = 0
        # Figure first pitch >= 0
        while pitch - MINOR_SCALE_DIFF[note_index - 1] >= 0:
            note_index -= 1
            pitch -= MINOR_SCALE_DIFF[note_index]

        # Add all notes to scale
        while pitch < 128:
            scales[key].append(pitch)
            pitch += MINOR_SCALE_DIFF[note_index % len(MINOR_SCALE_DIFF)]
            note_index += 1
        scales[key].append(pitch)
    return scales


SCALES: dict[int, list[int]] = __get_scales()
KEY_TO_NAME = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}
NAME_TO_KEY = {v: k for k, v in KEY_TO_NAME.items()}

# For example: "C minor": ["A#", "C", "D", "D#"", "F", "G", "G#"]
NOTES_OF_A_SCALE = {
    KEY_TO_NAME[(k) % 12] + " minor": sorted({KEY_TO_NAME[x % 12] for x in v}) for k, v in SCALES.items()
}


class Wave(Enum):
    """Defines different wave types for the play function."""

    SINE = 1
    SAWTOOTH = 2
    SQUARE = 3
    E_PIANO = 4
    BRASS = 5


WAVE_FUNCTIONS = {
    Wave.SINE: lambda x: np.sin(x),
    Wave.SAWTOOTH: lambda x: scipy.signal.sawtooth(x),
    Wave.SQUARE: lambda x: scipy.signal.square(x),
    # WARNING: E_PIANO sounds broken in practice — the `% 2` zeroes half the
    # waveform, producing a noisy/gated tone rather than a piano-like timbre.
    # Keep the entry so existing checkpoints/notebooks referencing it still
    # load, but do NOT pick it as a default for new code paths.
    Wave.E_PIANO: lambda x: np.sin(x * (np.int32(x) % 2)),
    Wave.BRASS: lambda x: np.sin(x * (np.int32(x) % 4)),
}


class Sounds808(Enum):
    """Timbre variants for the procedurally synthesized 808 (see ``riffly.synth``)."""

    # Fetched by frontend — careful when changing.
    PUNCHY = "punchy"
    SMOOTH = "smooth"
