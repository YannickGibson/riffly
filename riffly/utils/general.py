"""General utility functions used across the project."""

import random

import numpy as np
import pretty_midi
import torch
from pretty_midi import note_name_to_number


def get_threshold(x: np.ndarray) -> float:
    return 0.5  # np.median(x)  # + 2 * np.std(x)


def lowest_highest_note(piano_roll: np.ndarray) -> tuple[int, int]:
    # Returns lowest and highest note from a piano roll
    piano_roll_sum = piano_roll.sum(axis=1)
    lowest_note = 0
    # iterate from bottom
    for i in range(len(piano_roll_sum)):
        if piano_roll_sum[i] > 0:
            lowest_note = i
            break
    # iterate from top
    highest_note = 0
    for i in range(len(piano_roll_sum))[::-1]:
        if piano_roll_sum[i] > 0:
            highest_note = i
            break
    return lowest_note, highest_note


def get_minimum_duration(instrument: pretty_midi.Instrument) -> float:
    """Get minimum duration of notes in the instrument."""
    if len(instrument.notes) == 0:
        raise ValueError("Instrument has no notes, cannot calculate minimum duration.")

    # Iterate
    min_duration = float("inf")
    for note in instrument.notes:
        curr_duration = note.end - note.start
        min_duration = min(min_duration, curr_duration)
    return min_duration


def set_seeds(seed: int = 33, verbose: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if verbose:
        pass


def get_all_octaves(notes: list[str] | set[str]) -> set[str]:
    """Given a list of notes, gives all octaves of those notes in one set."""
    no_octaves = []
    for note in notes:
        no_octaves.append(note[0])

    all_octaves = set()
    for reduced_note in no_octaves:
        for octave in range(10):
            all_octaves.add(f"{reduced_note}{octave}")
    return all_octaves


def trim_notes_vertically(notes: list[str] | set[str], start_note, end_note) -> set[str]:
    """Trims notes to be within start_note and end_note (inclusive) in respect to their pitch."""
    reduced = set()
    start_number = note_name_to_number(start_note)
    end_number = note_name_to_number(end_note)
    for note in notes:
        note_number = note_name_to_number(note)
        if start_number <= note_number <= end_number:
            reduced.add(note)
    return reduced
