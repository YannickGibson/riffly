"""Interactive playback functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from riffly.constants import Wave
from riffly.processes import InteractivePostprocess

if TYPE_CHECKING:
    import torch


def play(
    data: np.ndarray | torch.Tensor,
    columns: int | None = None,
    rows: int | None = None,
    connect_notes: bool = True,
    wave: Wave = Wave.SAWTOOTH,
    repeat: int = 2,
    speed: int = 1,
    hat_interval: int | None = None,
    show_image: bool = True,
) -> None:
    """Postprocess and play a single matrix (tensor or numpy array).

    Args:
        data: A 2D tensor or numpy array representing a piano roll.
        connect_notes: Whether to connect consecutive notes.
        wave: The wave type to use for playback.
        repeat: Number of times to repeat playback.
        speed: Playback speed multiplier.
        hat_interval: Hi-hat interval in seconds, or None to disable.
        show_image: Whether to display the piano roll image.

    """
    if not isinstance(data, np.ndarray):
        data = data.detach().cpu().numpy()

    if columns is not None and rows is not None:
        data = data.reshape(rows, columns)

    mpost = InteractivePostprocess(data, connect_notes=connect_notes)
    play_postprocesses(
        [mpost],
        wave=wave,
        repeat=repeat,
        speed=speed,
        hat_interval=hat_interval,
        show_image=show_image,
    )


def play_predictions(
    preds: np.ndarray,
    indexes: list,
    rows: int,
    columns: int,
    midis_to_play: int = 2,
    connect_notes: bool = True,
    wave: Wave = Wave.SAWTOOTH,
    repeat: int = 2,
    time_multiplier: int = 1,
    hat_interval: int | None = None,
    show_image: bool = True,
    titles: list[str] | None = None,
    files: list[str] | None = None,
) -> None:
    """Build InteractivePostprocess objects from predictions and play them.

    Args:
        files: List of file paths. If provided, filenames will be extracted and used as titles.
        titles: List of titles. Used only if files is not provided.

    """
    midis_to_play = min(midis_to_play, len(preds))

    # Extract filenames from files if provided
    if files is not None:
        titles = [files[indexes[i]].replace("\\", "/").split("/")[-1] for i in range(midis_to_play)]

    # Build InteractivePostprocess objects
    mposts: list[InteractivePostprocess] = []
    labels: list[str] = []
    for i in range(midis_to_play):
        example = preds[indexes[i]].reshape(rows, columns)
        mpost = InteractivePostprocess(example, connect_notes=connect_notes)
        mposts.append(mpost)
        labels.append(titles[i] if titles else f"Prediction {i}")

    play_postprocesses(
        mposts,
        wave=wave,
        repeat=repeat,
        time_multiplier=time_multiplier,
        hat_interval=hat_interval,
        show_image=show_image,
        titles=labels,
    )


def play_postprocesses(
    mposts: list[InteractivePostprocess],
    wave: Wave = Wave.SAWTOOTH,
    repeat: int = 2,
    time_multiplier: int = 1,
    hat_interval: int | None = None,
    show_image: bool = True,
    titles: list[str] | None = None,
    octave_shift: int = 0,
) -> None:
    """Display and play a list of InteractivePostprocess objects."""
    import matplotlib.pyplot as plt

    count = len(mposts)

    # Show images
    if show_image and count > 0:
        cols = min(count, 5)
        fig_rows = (count - 1) // 5 + 1
        fig, axes = plt.subplots(nrows=fig_rows, ncols=cols, figsize=(4 * cols, 4 * fig_rows), squeeze=False)
        for ax in axes.flatten():
            ax.axis("off")
        for i, mpost in enumerate(mposts):
            row, col = divmod(i, cols)
            axes[row, col].imshow(mpost.matrix, interpolation="none", cmap="gray", origin="lower", vmin=0, vmax=1)
            title = titles[i] if titles else f"Melody {i}"
            axes[row, col].set_title(title)
            axes[row, col].axis("on")
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        fig.tight_layout()
        plt.show()

    # Play the melodies
    for mpost in tqdm(mposts):
        mpost.play(wave=wave, repeat=repeat, time_multiplier=time_multiplier, hat_interval=hat_interval, octave_shift=octave_shift)
