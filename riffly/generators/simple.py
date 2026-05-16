"""Simple MIDI Generator Module providing basic functions to create basic MIDI files."""

import random
from copy import copy

import pretty_midi
from pretty_midi import note_name_to_number
from pretty_midi.constants import INSTRUMENT_MAP

# Local modules
from riffly.processes.postprocess import MIDIPostprocess
from riffly.utils.general import get_all_octaves, trim_notes_vertically

MINOR_SCALE_SEMITONES = [0, 2, 3, 5, 7, 8, 10]
MINOR_SCALE_TO_REPRESENTATION = {
    "A": "A minor/C major",
    "A#": "A# minor/C# major",
    "B": "B minor/D major",
    "C": "C minor/D# major",
    "C#": "C# minor/E major",
    "D": "D minor/F major",
    "D#": "D# minor/F# major",
    "E": "E minor/G major",
    "F": "F minor/G# major",
    "F#": "F# minor/A major",
    "G": "G minor/A# major",
    "G#": "G# minor/B major",
}


class SimpleMIDIGenerator:
    def __init__(
        self,
        minor_scale: str = "A",
        bpm: int = 120,
        beats_per_bar: int = 4,
        units_per_beat: int = 1,
        instrument_name: str = INSTRUMENT_MAP[0],
    ) -> None:
        self.midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        self.MINOR_SCALE = minor_scale
        self.SCALE_NOTES: set = set()
        self.SCALE_NOTES_STR: set[str] = set()

        for octave in range(10):
            scale_note_name = f"{minor_scale}{octave}"
            scale_note_number = pretty_midi.note_name_to_number(scale_note_name)
            self.SCALE_NOTES = self.SCALE_NOTES.union(
                {scale_note_number + semitones for semitones in MINOR_SCALE_SEMITONES},
            )
            self.SCALE_NOTES_STR = self.SCALE_NOTES_STR.union(
                {pretty_midi.note_number_to_name(scale_note_number + semitones) for semitones in MINOR_SCALE_SEMITONES},
            )
        self.BPM = bpm
        self.BEATS_PER_BAR = beats_per_bar
        self.UNITS_PER_BEAT = units_per_beat
        self.instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))
        self.midi.instruments.append(self.instrument)

    def play(self) -> None:
        import sounddevice as sd

        audio_data = self.midi.synthesize()
        sd.play(audio_data, 44100)
        sd.wait()

    def add_chord(self, notes: list[str], start_beat: int, end_beat: int) -> None:
        start = self.beats_to_seconds(start_beat)
        end = self.beats_to_seconds(end_beat)
        for note_name in notes:
            note_number = pretty_midi.note_name_to_number(note_name)
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=start, end=end)
            self.instrument.notes.append(note)

    def add_arpeggio(self, chord: list[str], start_beat: int, end_beat: int) -> None:
        start = self.beats_to_seconds(start_beat)
        end = self.beats_to_seconds(end_beat)
        note_duration = (end - start) / len(chord)
        for i, note_name in enumerate(chord):
            note_number = pretty_midi.note_name_to_number(note_name)
            note = pretty_midi.Note(
                velocity=100,
                pitch=note_number,
                start=start + i * note_duration,
                end=start + (i + 1) * note_duration,
            )
            self.instrument.notes.append(note)

    def export(self, filename: str) -> None:
        self.midi.write(filename)

    @property
    def note_duration(self) -> float:
        return min(note.end - note.start for note in self.instrument.notes)

    @property
    def max_time(self) -> float:
        max_end = max(note.end for note in self.instrument.notes)
        end_times = [4 * i for i in range(1, 8 + 1)]
        # choose end interval that is the smallest but greater than max_end
        return min(interval for interval in end_times if interval >= max_end)

    @property
    def columns(self) -> int:
        return int(self.max_time / self.note_duration)

    def toJSON(self, **kwargs) -> dict:
        return MIDIPostprocess.toJSON(self, **kwargs)

    def random_in_chord(self, chord: list[str], start_note: str, end_note: str, start_beat: int, end_beat: int) -> None:
        start = self.beats_to_seconds(start_beat)
        end = self.beats_to_seconds(end_beat)
        note_duration = (end - start) / (end_beat - start_beat) / self.UNITS_PER_BEAT  # note for each unit

        # Define the range for random notes
        note_intersection = self.SCALE_NOTES_STR.intersection(get_all_octaves(chord))
        note_intersection = trim_notes_vertically(note_intersection, start_note, end_note)
        note_intersection = list(map(note_name_to_number, note_intersection))

        for beat_number in range(start_beat, end_beat):
            for unit_number in range(self.UNITS_PER_BEAT):
                note_number = random.choice(list(note_intersection))
                note_start = self.units_to_seconds(beat_number, unit_number)
                note_end = note_start + note_duration
                note = pretty_midi.Note(velocity=100, pitch=note_number, start=note_start, end=note_end)
                self.instrument.notes.append(note)

    def duplicate(self, beats_from: int, beats_to: int, paste_from: int) -> None:
        copy_from = self.beats_to_seconds(beats_from)
        copy_to = self.beats_to_seconds(beats_to)
        paste_from = self.beats_to_seconds(paste_from)
        new_notes = []
        for notes in self.instrument.notes:
            if notes.start >= copy_from and notes.start < copy_to:
                new_notes.append(copy(notes))
        for notes in new_notes:
            notes.start += paste_from - copy_from
            notes.end += paste_from - copy_from
            self.instrument.notes.append(notes)

    def beats_to_seconds(self, beat: int) -> float:
        return beat * 60 / self.BPM

    def units_to_seconds(self, beat: int, unit: int) -> float:
        return self.beats_to_seconds(beat) + unit * 60 / self.BPM / self.UNITS_PER_BEAT
