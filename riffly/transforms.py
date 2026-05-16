"""Piano-roll augmentation transforms.

All transforms act on a 2D piano-roll ``torch.Tensor`` of shape ``(rows, columns)``
and return the same shape. Compose them and pass the result as the ``transform``
argument to ``MIDIDataset`` (applied in ``__getitem__`` before flattening).

Transforms are stochastic and intended for *training only* — always build
separate train/val datasets so val data stays un-augmented.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import Dataset


class Compose:
    def __init__(self, transforms: list[Callable]) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        inner = ", ".join(type(t).__name__ for t in self.transforms)
        return f"Compose([{inner}])"


class RandomErase:
    """Zero a random rectangular area to teach the VAE to in-paint."""

    def __init__(self, p: float = 0.5, area_range: tuple[float, float] = (0.02, 0.15)) -> None:
        self.p = p
        self.area_range = area_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        rows, cols = x.shape[-2], x.shape[-1]
        total = rows * cols
        frac = torch.empty(1).uniform_(*self.area_range).item()
        erase = int(total * frac)
        if erase <= 0:
            return x
        # rectangle with random aspect
        aspect = torch.empty(1).uniform_(0.3, 3.0).item()
        h = max(1, min(rows, int((erase * aspect) ** 0.5)))
        w = max(1, min(cols, int(erase / h)))
        top = int(torch.randint(0, rows - h + 1, (1,)).item())
        left = int(torch.randint(0, cols - w + 1, (1,)).item())
        out = x.clone()
        out[..., top : top + h, left : left + w] = 0
        return out


class RandomTimeShift:
    """Beat-aligned cyclic shift along the time axis.

    Shifts by a random multiple of ``quantum`` columns, wrapping around, up
    to ``±max_steps * quantum``. Quantum=4 matches a beat at 16th-note
    resolution (4 cols per beat) which is the common piano-roll grid; the
    wrap preserves every note as long as the segment is close to periodic.
    """

    def __init__(self, quantum: int = 4, max_steps: int = 2, p: float = 0.5) -> None:
        self.quantum = max(1, quantum)
        self.max_steps = max_steps
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p or self.max_steps <= 0:
            return x
        steps = int(torch.randint(-self.max_steps, self.max_steps + 1, (1,)).item())
        if steps == 0:
            return x
        return torch.roll(x, shifts=steps * self.quantum, dims=-1)


class RandomSmallNoteShift:
    """Per-active-cell scale-degree shift of ±1 or ±2 rows.

    Because the dataset is already transposed to C major (one row per scale
    step), moving a cell up or down a row is an in-scale re-harmonisation —
    no accidentals introduced. Applied independently per active cell with
    small probability so note durations aren't destroyed wholesale.
    """

    def __init__(self, p_per_cell: float = 0.05, max_shift: int = 2) -> None:
        self.p = p_per_cell
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0 or self.p <= 0:
            return x
        mask = x > 0.5
        if not mask.any():
            return x
        rand = torch.rand_like(x)
        to_shift = mask & (rand < self.p)
        if not to_shift.any():
            return x
        out = x.clone()
        coords = to_shift.nonzero(as_tuple=False)
        rows = x.shape[-2]
        for rc in coords.tolist():
            r, c = rc[-2], rc[-1]
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
            if shift == 0:
                continue
            new_r = r + shift
            if 0 <= new_r < rows:
                out[..., r, c] = 0.0
                out[..., new_r, c] = 1.0
        return out


class RandomNoteOctaveShift:
    """Octave-shift a tiny handful of individual notes (0 to ``max_notes``).

    Unlike ``RandomOctaveShift`` which moves the entire piano roll as a block,
    this picks at most ``max_notes`` active cells and moves each by ±1 octave
    (``±rows_per_octave`` rows). Often does nothing (the count is sampled
    from 0). In-scale: rows encode scale degrees, so an octave is exactly
    ``rows_per_octave=7`` rows.
    """

    def __init__(self, max_notes: int = 3, rows_per_octave: int = 7) -> None:
        self.max_notes = max(0, max_notes)
        self.rows_per_octave = rows_per_octave

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_notes <= 0:
            return x
        mask = x > 0.5
        if not mask.any():
            return x
        coords = mask.nonzero(as_tuple=False)
        if coords.numel() == 0:
            return x
        k = int(torch.randint(0, self.max_notes + 1, (1,)).item())
        if k == 0:
            return x
        k = min(k, coords.shape[0])
        perm = torch.randperm(coords.shape[0])[:k]
        rows = x.shape[-2]
        out = x.clone()
        for idx in perm.tolist():
            rc = coords[idx].tolist()
            r, c = rc[-2], rc[-1]
            direction = 1 if torch.rand(1).item() < 0.5 else -1
            new_r = r + direction * self.rows_per_octave
            if 0 <= new_r < rows:
                out[..., r, c] = 0.0
                out[..., new_r, c] = 1.0
        return out


class RandomOctaveShift:
    """Cyclic shift along the row (pitch) axis by +/- a few rows.

    Acts as a cheap within-scale transposition without leaving the matrix.
    """

    def __init__(self, max_shift: int = 2, p: float = 0.5) -> None:
        self.max_shift = max_shift
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p or self.max_shift <= 0:
            return x
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-2)


_AUG_REGISTRY: dict[str, Callable[[], Callable]] = {
    "erase": lambda: RandomErase(p=0.5, area_range=(0.02, 0.15)),
    "time_shift": lambda: RandomTimeShift(quantum=4, max_steps=2, p=0.5),
    "octave_shift": lambda: RandomOctaveShift(max_shift=2, p=0.5),
    "small_note_shift": lambda: RandomSmallNoteShift(p_per_cell=0.05, max_shift=2),
    "note_octave_shift": lambda: RandomNoteOctaveShift(max_notes=3, rows_per_octave=7),
}


class AugmentedSubset(Dataset):
    """Wrap a ``(path, flat_tensor)`` dataset and apply a 2D transform on each item.

    Useful to attach training-only augmentations to a ``torch.utils.data.Subset``
    produced by ``train_test_split``, while leaving the val subset untouched.
    """

    def __init__(self, subset, transform: Callable, rows: int, columns: int, flatten: bool = True) -> None:
        self.subset = subset
        self.transform = transform
        self.rows = rows
        self.columns = columns
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        path, tensor = self.subset[idx]
        if tensor.ndim == 1:
            tensor = tensor.view(self.rows, self.columns)
        tensor = self.transform(tensor)
        if self.flatten:
            tensor = tensor.flatten()
        return path, tensor


def build_transform(names: list[str] | None) -> Callable | None:
    """Build a Compose from a list of short augmentation names.

    Accepted names: ``flip``, ``erase``, ``time_shift``, ``octave_shift``.
    Returns ``None`` when ``names`` is empty/None so the dataset skips the
    transform branch entirely.
    """
    if not names:
        return None
    transforms = []
    for name in names:
        if name not in _AUG_REGISTRY:
            msg = f"Unknown augmentation {name!r}. Expected one of: {sorted(_AUG_REGISTRY)}"
            raise ValueError(msg)
        transforms.append(_AUG_REGISTRY[name]())
    return Compose(transforms)
