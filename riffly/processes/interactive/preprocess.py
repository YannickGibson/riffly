"""Includes a class that is a preprocessing extension with interactive functionality for plotting."""

import pretty_midi

from riffly.constants import Wave
from riffly.plots import plot_piano_roll
from riffly.processes.interactive.postprocess import InteractivePostprocess
from riffly.processes.preprocess import MIDIPreprocess


class InteractivePreprocess(MIDIPreprocess):
    """Includes preprocessing version with interactive functionality for plotting."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def plot(
        self,
        fs: int | None = None,
        stretch=False,
        figsize=(4, 2),
        quantize_interval=None,
        length_perc=1,
        from_time=None,
        to_time=None,
        ax=None,
        instrument_idx: int = 0,
    ) -> None:
        """Plots the piano roll of a specific instrument."""
        if from_time is not None and to_time is not None:
            notes = self.get_notes(instrument_idx=instrument_idx)
            reduced_notes = []
            for note in notes:
                if from_time <= note.start <= to_time and from_time <= note.end <= to_time:
                    reduced_notes.append(note)
            reduced_midi = pretty_midi.PrettyMIDI()
            reduced_instrument = pretty_midi.Instrument(program=self.instruments[instrument_idx].program)
            for note in reduced_notes:
                reduced_instrument.notes.append(note)
            reduced_midi.instruments.append(reduced_instrument)
            max_time = round(max([note.end for note in reduced_notes]))
            piano_roll = reduced_midi.get_piano_roll(self.fs, pedal_threshold=0)
        else:
            max_time = round(max([note.end for note in self.get_notes(instrument_idx=instrument_idx)]))
            piano_roll = self.get_roll(fs, instrument_idx=instrument_idx)
        plot_piano_roll(
            piano_roll,
            stretch=stretch,
            figsize=figsize,
            preprocessed=self.preprocessed,
            quantize_interval=quantize_interval,
            max_time=max_time,
            length_perc=length_perc,
            ax=ax,
        )

    def play(
        self,
        columns: int,
        rows: int,
        connect_notes: bool = True,
        wave: Wave = Wave.SAWTOOTH,
        repeat: int = 1,
        speed: float = 8,
        hat_interval: float | None = None,
        instrument_idx: int = 0,
    ) -> None:
        """Preprocess into a matrix and play it via InteractivePostprocess."""
        matrix = self.preprocess(columns=columns, rows=rows, instrument_idx=instrument_idx)

        mpost = InteractivePostprocess(matrix, connect_notes=connect_notes)
        mpost.play(wave=wave, repeat=repeat, speed=speed, hat_interval=hat_interval)
