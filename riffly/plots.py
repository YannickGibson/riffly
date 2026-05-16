"""Plotting utilities for visualizing piano rolls and training metrics."""

from __future__ import annotations

import os

import numpy as np
import pretty_midi
import torch
from typing import TYPE_CHECKING

from riffly.constants import SCALES
from riffly.models.general import ModelInterface
from riffly.utils.general import lowest_highest_note

if TYPE_CHECKING:
    from neptune.metadata_containers import Run

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def plot(piano_roll: np.ndarray, title: str = "Generated Melody", figsize: tuple[int, int] = (8, 4)) -> None:
    """Plot a piano roll numpy array."""
    _fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(piano_roll, interpolation="none", cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("on")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def first_last_column(piano_roll: np.ndarray) -> tuple[int, int]:
    # Returns first and last column from a piano roll
    piano_roll_sum = piano_roll.sum(axis=0)
    first_column = 0
    # iterate from left
    for i in range(len(piano_roll_sum)):
        if piano_roll_sum[i] > 0:
            first_column = i
            break
    # iterate from right
    last_column = 0
    for i in range(len(piano_roll_sum))[::-1]:
        if piano_roll_sum[i] > 0:
            last_column = i
            break
    return first_column, last_column


def plot_piano_roll(
    piano_roll: np.ndarray,
    stretch: bool = False,
    figsize: tuple[int, int] = (8, 4),
    preprocessed=False,
    quantize_interval=None,
    max_time=None,
    length_perc=1,
    title=None,
    ax=None,
) -> None:
    if ax is None:
        ax_predefined = False
        _fig, ax = plt.subplots(figsize=figsize)
    else:
        ax_predefined = True

    # Trim length
    piano_roll = piano_roll[:, : int(piano_roll.shape[1] * length_perc)]

    # Plot the piano roll matrix
    ax.imshow(
        piano_roll,
        aspect="auto",
        cmap="gray",
        interpolation="none",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    lowest_note, highest_note = lowest_highest_note(piano_roll)
    first_column, last_column = first_last_column(piano_roll)

    A_MINOR_LETTERS = [pretty_midi.utilities.note_number_to_name(note_number) for note_number in SCALES[9]]
    if preprocessed:
        y_height = piano_roll.shape[0]
        ax.set_yticks(np.arange(len(SCALES[9][:y_height])) - 0.5)
        ax.set_yticklabels(A_MINOR_LETTERS[:y_height], fontsize=6)
    if not preprocessed:
        ax.set_yticks(np.array(SCALES[9]) - 0.5)
        ax.set_yticklabels(A_MINOR_LETTERS, fontsize=6)
    if stretch:
        ax.set_ylim(lowest_note - 1, highest_note + 1)
        ax.set_xlim(first_column, last_column)
    if quantize_interval:
        max_time *= length_perc
        columns_interval = (quantize_interval / max_time) * piano_roll.shape[1]
        # Draw vertical lines for quantize interval
        rng = np.arange(first_column, piano_roll.shape[1], columns_interval)  # float range
        for i in rng:
            ax.axvline((i) - 0.5, color="red", linestyle="-", linewidth=1)

    # set minor ticks every 4 points
    ax.set_xticks(np.arange(first_column, piano_roll.shape[1] + 1, 8) - 0.5)
    ax.set_xticklabels(np.arange(first_column, piano_roll.shape[1] + 1, 8), fontsize=6)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(base=1, offset=0.5))
    ax.set_xlabel("Time")
    ax.set_ylabel("Note Number")
    if title is None:
        ax.set_title(f"Piano Roll (shape: {piano_roll.shape})")
    else:
        ax.set_title(title)
    ax.grid(which="both", color="black", linestyle="-", linewidth=0.3)
    if not ax_predefined:
        plt.show()


def plot_dataset_samples(
    dataset,
    n: int = 5,
    cols: int = 5,
    figsize: tuple[int, int] | None = None,
    preprocessed: bool = True,
) -> None:
    """Plot the first n images from a dataset.

    Args:
        dataset: A MIDIDataset or similar dataset that returns (path, tensor) tuples.
        n: Number of samples to plot.
        cols: Number of columns in the grid.
        figsize: Figure size. If None, auto-calculated based on grid size.
        preprocessed: Whether the data is preprocessed (affects y-axis labels).

    """
    n = min(n, len(dataset))
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (cols * 4, rows * 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i in range(n):
        row_idx, col_idx = divmod(i, cols)
        ax = axes[row_idx, col_idx]

        path, tensor = dataset[i]
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()

        # Reshape if flattened
        if tensor.ndim == 1:
            piano_roll = tensor.reshape(dataset.rows, dataset.columns)
        else:
            piano_roll = tensor

        ax.imshow(
            piano_roll,
            aspect="auto",
            cmap="gray",
            interpolation="none",
            origin="lower",
            vmin=0,
            vmax=1,
        )
        # Extract short name from path
        short_name = path.replace("\\", "/").split("/")[-1]
        short_name = short_name[:10] + "..." if len(short_name) > 13 else short_name
        ax.set_title(short_name, fontsize=8)
        ax.set_xlabel("Time", fontsize=6)
        ax.set_ylabel("Note", fontsize=6)
        ax.tick_params(axis="both", labelsize=5)

    # Hide empty subplots
    for i in range(n, rows * cols):
        row_idx, col_idx = divmod(i, cols)
        axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.show()


def show_val_score_and_loss(
    val_scores_dict: dict[str, list[float]],
    val_score_metric_name: str,
    val_losses: list[float],
    losses: list[float],
    batch_count: int,
    batch_size: int,
    epochs: int,
    best_epoch: int | None = None,
    neptune_run: Run | None = None,
    neptune_only: bool = False,
    save_folder: str | None = None,
) -> None:
    if not neptune_run and neptune_only:
        msg = "neptune_only is True but neptune_run is False."
        raise ValueError(msg)

    import pandas as pd

    # Plot: normalize steps to epochs
    _fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    # Validation score
    for val_name, val_scores in val_scores_dict.items():
        tmp = pd.DataFrame({val_name: val_scores})
        tmp.index = range(1, len(tmp) + 1)  # shift axis by one to the right
        tmp.plot(ax=ax1, marker="o", label=val_name)
    ax1.set_title("Validation Score")

    # Validation loss
    tmp = pd.DataFrame({"val_loss": val_losses})
    tmp.index = range(1, len(tmp) + 1)  # shift axis by one to the right
    tmp.plot(ax=ax2, marker="o")
    ax2.set_title("Validation Loss")

    # Loss
    tmp = pd.DataFrame({"loss": losses})
    tmp.index = range(1, len(tmp) + 1)
    tmp.plot(ax=ax3)
    ax3.set_title("Training Loss")

    # Scale x axis to epochs for loss
    total_steps = batch_count * epochs
    ax3.set_xticks([*list(range(0, total_steps, batch_count)), total_steps])
    ax3.set_xticklabels([i // batch_count for i in range(0, total_steps, batch_count)] + [epochs])

    # Draw vertical lines for best epoch
    best_epoch_postfix = "st" if best_epoch == 0 else "nd" if best_epoch == 1 else "rd" if best_epoch == 2 else "th"
    if best_epoch is not None:
        ax1.axvline(
            x=(best_epoch + 1),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"{val_scores_dict[val_score_metric_name][best_epoch]:.3f} acc",
        )
        ax2.axvline(
            x=(best_epoch + 1),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"{val_losses[best_epoch]:.3f} loss",
        )
        ax3.axvline(
            x=(best_epoch + 1) * batch_count,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"{best_epoch + 1}{best_epoch_postfix} epoch",
        )

    # Other settings
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(ax.get_xticks().astype(int))  # set integers to x axis
        ax.legend()
    ax1.set_ylim(0, 1)

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        plot_path = os.path.join(save_folder, "val_score_and_loss.png")
    else:
        plot_path = "tmp_plot.png"
    plt.savefig(plot_path)
    if not neptune_only:
        plt.show()
    if neptune_run:
        # Log the figure to neptune
        neptune_run["val_score_and_loss_plot"].log(plot_path)


def plot_preds_labels(
    preds,
    labels,
    image_rows,
    image_cols,
    scoring_fn=None,
    images_to_show=10,
    random_sample=True,
    save_folder: str | None = None,
):
    """Plot predictions and labels. Odd rows are labels, even rows are predictions."""
    images_to_show = min(images_to_show, len(preds))
    # If images to show are over 5 make 2 more rows
    cols = min(images_to_show, 5)
    rows = ((images_to_show - 1) // 5 + 1) * 2

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 3.0, rows * 2.0))

    # 10 random indexes
    if random_sample:
        import random

        indexes = random.sample(range(len(preds)), images_to_show)
    else:
        indexes = range(images_to_show)

    # Settings
    for ax in axes.flatten():
        ax.axis("off")  # non-displayed axes will be invisible
        ax.set_aspect("equal")

    # Iterate batches
    for i in range(images_to_show):
        # Variables
        pred = preds[indexes[i]].reshape(image_rows, image_cols)
        label = labels[indexes[i]].reshape(image_rows, image_cols)
        row = (i // cols) * 2
        col = i % cols

        # Plot Label and Prediction
        axes[row, col].imshow(label, interpolation="none", cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row + 1, col].imshow(pred, interpolation="none", cmap="gray", origin="lower", vmin=0, vmax=1)

        # Title and labels
        axes[row, col].set_title(f"Label {i}")
        axes[row + 1, col].set_title(f"Pred {i} ({scoring_fn(pred.flatten(), label.flatten()):.2f})")
        if col == 0:
            axes[row, col].set_ylabel("Label")
            axes[row + 1, col].set_ylabel("Prediction")

        # Add border
        for ax in [axes[row, col], axes[row + 1, col]]:
            ax.axis("on")  # Hide axes
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1)
            ax.set_xticks([])
            ax.set_yticks([])

    # Show
    fig.tight_layout()
    if save_folder is not None:
        import os

        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, "preds_vs_labels.png"), dpi=160, bbox_inches="tight")
    plt.show()

    return indexes


def generate_and_plot(
    model: ModelInterface,
    images_to_show: int = 10,
    seed: int | None = None,
    threshold: float | None = None,
    save_folder: str | None = None,
    max_empty_retries: int = 10,
) -> list[np.ndarray]:
    """Generate images and plot them in a grid using model.generate().

    ``seed=None`` (the default) draws from the current global RNG state so
    each call produces a different set of melodies. If a sampled z decodes to
    an all-zero piano roll, it's resampled up to ``max_empty_retries`` times
    before giving up — this avoids silent WAVs/empty MIDIs downstream.
    """
    # Grid: up to 5 columns, wrap to extra rows as needed.
    cols = min(images_to_show, 5)
    rows = (images_to_show - 1) // cols + 1
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 3.2, rows * 2.2), squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
        ax.set_aspect("equal")

    # Set seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # iterate batches
    generated = []
    for i in range(images_to_show):
        # Variables
        row = i // cols
        col = i % cols

        # Generate — resample z if the thresholded roll is empty
        pred = model.generate(threshold=threshold)
        attempts = 0
        while bool(np.asarray(pred).sum() == 0) and attempts < max_empty_retries:
            pred = model.generate(threshold=threshold)
            attempts += 1
        generated.append(pred)

        # Plot
        axes[row, col].imshow(pred, interpolation="none", cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row, col].set_title(f"Generated {i}")

        # Add black border
        ax = axes[row, col]
        ax.axis("on")  # Hide axes
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    if save_folder is not None:
        import os

        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, "generated_melodies.png"), dpi=160, bbox_inches="tight")
    plt.show()

    return generated
