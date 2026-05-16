"""The class provides .mid file preprocessing functionalities.

Supports multiple instruments: all instruments with notes are loaded and
processed.  Global transforms (BPM, quantize, segment, shorten) apply to
every instrument; per-instrument transforms (octave shift, matrix generation)
are scoped via the ``instrument_idx`` parameter.  Segment extraction uses the
union time range across all instruments.  Segment-level filters keep a segment
when *any* instrument passes.
"""

import logging
import math
from enum import Enum

import numpy as np
import pretty_midi

from riffly.constants import MAJOR_SCALE_DIFF, SCALES
from riffly.utils.exceptions import MIDIKeyError, MIDINoInstrumentsError, MIDINotConstantBeatIntervalError
from riffly.utils.general import lowest_highest_note

logger = logging.getLogger(__name__)


def get_constant_beat_interval(beats: list[float]) -> tuple[float]:
    """Check if beat differences are constant using numpy. If low amount of beats or non-constant, raise error."""
    beats = np.array(beats)

    if len(beats) < 2:
        raise MIDINotConstantBeatIntervalError()

    # Calculate differences
    diffs = np.diff(beats)

    # Check if all are approximately equal
    is_constant = np.allclose(diffs, diffs[0])

    if is_constant:
        return diffs[0]
    raise MIDINotConstantBeatIntervalError()


class OctaveShift(Enum):
    UP = "up"
    DOWN = "down"

class MIDIPreprocess:
    """Preprocesses MIDI files for input into ML networks.

    All instruments that contain notes are loaded.  Use ``instrument_idx``
    parameters on per-instrument methods to target a specific instrument.
    """

    FEW_NOTES_THRESHOLD_PERC = 0.02  # PERCENTAGE

    def __init__(self, midi_path: str, fs: None | int = None, verbose: bool = False) -> None:
        self.midi = pretty_midi.PrettyMIDI(midi_path)
        self.path = midi_path
        self.verbose = verbose
        self.preprocessed = False
        self.preprocessed_matrices: dict[int, np.ndarray] = {}
        self._global_preprocessed = False
        self.instrument_indices = self._get_instrument_indices()
        self.instruments: list[pretty_midi.Instrument] = [
            self.midi.instruments[i] for i in self.instrument_indices
        ]

        self.beat_interval = get_constant_beat_interval(self.midi.get_beats())

        # Get instrument index which has large
        if fs is None:
            self.fs = self.get_fs()
            if self.verbose:
                pass

    # ── properties (backward compatibility) ──────────────────────────────

    @property
    def n_instruments(self) -> int:
        """Number of instruments with notes."""
        return len(self.instruments)

    @property
    def instrument(self) -> pretty_midi.Instrument:
        """Backward-compatible access to the first instrument."""
        return self.instruments[0]

    @property
    def instrument_index(self) -> int:
        """Backward-compatible access to the first instrument index."""
        return self.instrument_indices[0]

    @property
    def preprocessed_matrix(self) -> np.ndarray | None:
        """Backward-compatible access to the first instrument's preprocessed matrix."""
        return self.preprocessed_matrices.get(0, None)

    @preprocessed_matrix.setter
    def preprocessed_matrix(self, value):
        if value is not None:
            self.preprocessed_matrices[0] = value
        elif 0 in self.preprocessed_matrices:
            del self.preprocessed_matrices[0]

    # ── instrument discovery ─────────────────────────────────────────────

    def _get_instrument_indices(self) -> list[int]:
        """Return indices of all instruments that have notes."""
        indices = []
        for i, instrument in enumerate(self.midi.instruments):
            if len(instrument.notes) > 0:
                indices.append(i)
        if len(indices) == 0:
            raise MIDINoInstrumentsError("MIDI file has no instruments with notes.")
        return indices

    # ── note access ──────────────────────────────────────────────────────

    def get_notes(
        self,
        segment_start_end: tuple[float, float] | None = None,
        *,
        instrument_idx: int | None = None,
    ) -> list[pretty_midi.Note]:
        """Return a list of notes, optionally filtered by instrument and/or time segment.

        Args:
            segment_start_end: Optional ``(start, end)`` time window.
            instrument_idx: If *None*, returns notes from **all** instruments
                combined.  If given, returns notes from that instrument only.
        """
        # Get base note list
        if instrument_idx is None:
            base_notes = []
            for inst in self.instruments:
                base_notes.extend(inst.notes)
        else:
            base_notes = self.instruments[instrument_idx].notes

        # No segment filter
        if segment_start_end is None:
            return base_notes

        # Prepare
        segment_start, segment_end = segment_start_end

        # Iterate
        notes_in_segment = []
        for note in base_notes:
            # Does note have no overlap with segment? '>=' for start because can't start at end
            if (
                note.end <= segment_start or note.start >= segment_end or note.start < segment_start
            ):  # Avoid abrupt start
                continue

            # Accept partial overlap only for end of segment.
            clipped_end = min(note.end, segment_end)

            # Create note
            notes_in_segment.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=clipped_end,
                ),
            )
        return notes_in_segment

    def set_notes(self, notes: list[pretty_midi.Note], instrument_idx: int = 0) -> None:
        self.instruments[instrument_idx].notes = notes

    # ── piano roll ───────────────────────────────────────────────────────

    def get_roll(self, fs: int | None = None, *, instrument_idx: int | None = None) -> np.ndarray:
        """Get piano roll for a specific instrument or summed across all.

        Args:
            fs: Sampling frequency.  If *None*, uses ``self.fs``.
            instrument_idx: If *None*, returns summed piano roll across all
                instruments.  If given, returns the roll for that instrument.
        """
        if instrument_idx is not None and instrument_idx in self.preprocessed_matrices:
            return self.preprocessed_matrices[instrument_idx]
        if fs is None:
            fs = self.fs
        if instrument_idx is None:
            # Sum piano rolls across all instruments
            rolls = [inst.get_piano_roll(fs, pedal_threshold=0) for inst in self.instruments]
            if not rolls:
                return np.zeros((128, 0))
            max_cols = max(r.shape[1] for r in rolls)
            padded = []
            for r in rolls:
                if r.shape[1] < max_cols:
                    r = np.pad(r, ((0, 0), (0, max_cols - r.shape[1])))
                padded.append(r)
            return sum(padded)
        return self.instruments[instrument_idx].get_piano_roll(fs, pedal_threshold=0)

    # ── validation helpers ───────────────────────────────────────────────

    def is_valid(self) -> bool:
        return self.does_have_key()

    def does_segment_have_enough_pitches(
        self, start: float, end: float, min_unique_pitches: int = 3, *, instrument_idx: int | None = None,
    ) -> bool:
        """Return False if the segment has fewer than `min_unique_pitches` distinct pitches."""
        notes = self.get_notes(segment_start_end=(start, end), instrument_idx=instrument_idx)
        unique_pitches = {note.pitch for note in notes}
        return len(unique_pitches) >= min_unique_pitches

    def does_segment_have_enough_notes(
        self, start: float, end: float, few_notes_count_threshold: int, *, instrument_idx: int | None = None,
    ) -> bool:
        notes = self.get_notes(segment_start_end=(start, end), instrument_idx=instrument_idx)
        if len(notes) < few_notes_count_threshold:
            return False
        return True
        matrix = reduced_instrument.get_piano_roll(fs=self.fs, pedal_threshold=0)
        first_row = None
        last_row = None
        for i in range(matrix.shape[1]):
            if matrix[:, i].sum() > 0:
                if first_row is None:
                    first_row = i
                last_row = i
        if first_row is None:
            first_row = 0
        if last_row is None:
            last_row = matrix.shape[1] - 1
        first_column = None
        last_column = None
        for i in range(matrix.shape[0]):
            if matrix[i, :].sum() > 0:
                if first_column is None:
                    first_column = i
                last_column = i
        if first_column is None:
            first_column = 0
        if last_column is None:
            last_column = matrix.shape[0] - 1

        # trim matrix
        matrix = matrix[first_column : last_column + 1, first_row : last_row + 1]
        matrix = np.where(matrix > 0.5, 1, 0)
        if matrix.size == 0:
            return False
        ratio = matrix.sum() / matrix.size
        return not (ratio < MIDIPreprocess.FEW_NOTES_THRESHOLD_PERC)

    # ── global mutation methods (apply to ALL instruments) ───────────────

    def shift_to_start(self) -> None:
        """Shift notes across all instruments so the earliest note starts at 0."""
        all_notes = self.get_notes()  # Combined from all instruments
        if len(all_notes) == 0:
            return
        start_time = min(note.start for note in all_notes)
        for inst in self.instruments:
            for note in inst.notes:
                note.start -= start_time
                note.end -= start_time

    def set_bpm_120(self) -> None:
        """Scale all note timings across all instruments to match 120 BPM."""
        if self.beat_interval is None:
            raise ValueError("Beat interval is not constant or not enough beats to determine it, cannot set BPM.")
        target_bpm = 120
        target_beat_interval = 60 / target_bpm
        current_beat_interval = self.beat_interval
        ratio = target_beat_interval / current_beat_interval

        for inst in self.instruments:
            new_notes = []
            for note in inst.notes:
                # Round to 0.5 multiples to avoid fp issues
                note.start = round(note.start * ratio * 2) / 2
                note.end = round(note.end * ratio * 2) / 2
                new_notes.append(note)
            inst.notes = new_notes

        # Apply changes
        self.beat_interval = target_beat_interval
        self.fs = self.get_fs()  # Update sampling rate to match new timings

    def set_segment(self, segment: tuple[float, float]) -> None:
        """Filter all instruments to only include notes within *segment*."""
        for idx in range(len(self.instruments)):
            new_notes = []
            inst = self.instruments[idx]
            new_instrument = pretty_midi.Instrument(program=inst.program)
            for note in inst.notes:
                logger.debug(f"Note: start={note.start}, end={note.end}, segment={segment}")
                if segment[0] <= note.start <= segment[1] and segment[0] <= note.end <= segment[1]:
                    new_notes.append(note)
            new_instrument.notes = new_notes
            # Apply changes
            self.instruments[idx] = new_instrument
            self.midi.instruments[self.instrument_indices[idx]] = new_instrument

    def does_segment_have_many_notes(
        self, start: float, end: float, many_notes_threshold: int, *, instrument_idx: int | None = None,
    ) -> bool:
        """Return True if the segment has more than ``many_notes_threshold`` notes."""
        notes = self.get_notes(segment_start_end=(start, end), instrument_idx=instrument_idx)
        if len(notes) > many_notes_threshold:
            return True
        return False

    # ── segment filtering ("any instrument" logic) ───────────────────────

    def exclude_segments_with_many_notes(self, segments, many_notes_threshold: int) -> list[tuple[float, float]]:
        """Exclude segments where *no* instrument has an acceptable note count (not too many)."""
        new_segments = []
        for start, end in segments:
            if any(
                not self.does_segment_have_many_notes(
                    start, end, many_notes_threshold=many_notes_threshold, instrument_idx=i,
                )
                for i in range(self.n_instruments)
            ):
                new_segments.append((start, end))
        return new_segments

    def exclude_segments_with_few_notes(self, segments, few_notes_count_threshold: int) -> list[tuple[float, float]]:
        """Exclude segments where *no* instrument has enough notes."""
        new_segments = []
        for start, end in segments:
            if any(
                self.does_segment_have_enough_notes(
                    start, end, few_notes_count_threshold=few_notes_count_threshold, instrument_idx=i,
                )
                for i in range(self.n_instruments)
            ):
                new_segments.append((start, end))
        return new_segments

    def exclude_segments_with_few_pitches(self, segments, min_unique_pitches: int) -> list[tuple[float, float]]:
        """Exclude segments where *no* instrument has enough distinct pitches."""
        new_segments = []
        for start, end in segments:
            if any(
                self.does_segment_have_enough_pitches(
                    start, end, min_unique_pitches=min_unique_pitches, instrument_idx=i,
                )
                for i in range(self.n_instruments)
            ):
                new_segments.append((start, end))
        return new_segments

    # ── quantize ─────────────────────────────────────────────────────────

    def quantize(self, interval: float | None = None, discard_threshold=0.7) -> None:
        """Quantize notes across all instruments to *interval*."""
        if interval is None:
            if self.beat_interval is not None:
                interval = self.beat_interval
            else:
                raise ValueError("Interval must be provided if beat interval is not constant or not enough beats.")
        for inst in self.instruments:
            new_notes = []
            for note in inst.notes:
                # 0.74 -> 0.5
                start = round(note.start / interval) * interval
                end = round(note.end / interval) * interval

                # If the start and end is the same, extend end.
                if start == end:
                    end += interval

                # Quantize
                note.start = start
                note.end = end
                new_notes.append(note)
            inst.notes = new_notes

    # ── segment extraction ───────────────────────────────────────────────

    def extend_if_short(self, columns) -> tuple[int, float | None, float | None]:
        """Extend MIDI if short by repeating.  Uses union time range across all instruments."""
        # Variables — notes from ALL instruments
        notes = self.get_notes()

        # Check if there are no notes in MIDI file.
        if len(notes) == 0:
            return [], None, None

        # Get segment duration
        song_start = min([note.start for note in notes])
        song_end = max([note.end for note in notes])
        total_duration = song_end - song_start
        min_note_duration = self.beat_interval
        segment_duration = min_note_duration * columns
        n_segments = self.threshold_round(total_duration / segment_duration, threshold=0.75)  # keep last partial interval

        # If no segments and total duration is multiple of segment duration, set n_segments to 1
        if (
            n_segments == 0 and total_duration > 0
        ):  # -> doesnt have to have exactly segment length to duplicate -> # and self.is_multiple(total_duration, segment_duration):
            self.repeat_fill_one_segment(segment_duration, total_duration)
            n_segments = 1

        return n_segments, song_start, segment_duration

    def threshold_round(self, value, threshold):
        """
        Rounds to an integer.
        Rounds up only if the fractional part exceeds `threshold`,
        otherwise rounds down.
        """
        base = math.floor(value)
        fractional = value - base
        return base + (fractional >= threshold)

    def repeat_fill_one_segment(self, segment_duration: float, total_duration: float) -> None:
        """Fill midi duration to segment duration by repeating all instruments."""
        if total_duration <= 0:
            raise ValueError("Total duration must be greater than 0 to fill to one segment.")
        
        repeat_times = round(segment_duration / total_duration)
        new_total_duration = segment_duration / repeat_times
        for idx in range(len(self.instruments)):
            new_notes = []
            new_instrument = pretty_midi.Instrument(program=self.instruments[idx].program)
            for i in range(repeat_times):
                for note in self.instruments[idx].notes:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start + i * new_total_duration,
                        end=note.end + i * new_total_duration,
                    )
                    new_notes.append(new_note)
            new_instrument.notes = new_notes
            self.instruments[idx] = new_instrument

    def is_multiple(self, a: float, b: float, tol=1e-6) -> bool:
        """Check if a is a multiple of b within a tolerance."""
        if a == 0:
            return False
        return abs(b / a - round(b / a)) < tol

    def extract_segments(self, columns: int) -> list[tuple[float, float]]:
        """Extract segments of the MIDI based on width of input into the ML model.

        Uses the union time range across all instruments.
        """
        if self.beat_interval is None:
            raise ValueError("Beat interval is not constant or not enough beats to determine it, cannot get segments.")

        n_segments, song_start, segment_duration = self.extend_if_short(columns=columns)

        # Create segments
        segments = []
        for i in range(n_segments):
            segments.append(
                (
                    song_start + (i + 0) * (segment_duration),  # start time
                    song_start + (i + 1) * (segment_duration),  # end time
                ),
            )
        return segments

    def get_segments_with_key(self, segments) -> list[tuple[float, float]]:
        """Keep segments where at least one instrument has a recognizable key."""
        intervals_with_key = []
        for start, end in segments:
            if any(
                self.does_have_key(segment_start_end=(start, end), instrument_idx=i)
                for i in range(self.n_instruments)
            ):
                intervals_with_key.append((start, end))
        logger.debug(f"Intervals with key: {len(intervals_with_key)} / {len(segments)}")
        return intervals_with_key

    def get_valid_instrument_segment_pairs(
        self,
        segments: list[tuple[float, float]],
        few_notes_count_threshold: int,
        many_notes_threshold: int,
        min_unique_pitches: int,
    ) -> list[tuple[int, int]]:
        """Return ``(instrument_idx, segment_idx)`` pairs where that instrument
        individually passes all quality filters for that segment.

        Called after segment-level filtering to determine which specific
        (instrument, segment) combinations are valid for the dataset.
        """
        pairs = []
        for seg_idx, (start, end) in enumerate(segments):
            for inst_idx in range(self.n_instruments):
                has_key = self.does_have_key(
                    segment_start_end=(start, end), instrument_idx=inst_idx,
                )
                enough_notes = self.does_segment_have_enough_notes(
                    start, end, few_notes_count_threshold, instrument_idx=inst_idx,
                )
                enough_pitches = self.does_segment_have_enough_pitches(
                    start, end, min_unique_pitches, instrument_idx=inst_idx,
                )
                not_too_many = not self.does_segment_have_many_notes(
                    start, end, many_notes_threshold, instrument_idx=inst_idx,
                )
                if has_key and enough_notes and enough_pitches and not_too_many:
                    pairs.append((inst_idx, seg_idx))
        return pairs

    # ── key detection / transposition ────────────────────────────────────

    def does_have_key(
        self,
        segment_start_end: tuple[float, float] | None = None,
        *,
        instrument_idx: int | None = None,
    ) -> bool:
        """Check if the notes could be in 1 or more keys."""
        pitches = {
            note.pitch
            for note in self.get_notes(segment_start_end=segment_start_end, instrument_idx=instrument_idx)
        }
        if len(pitches) == 0:
            return False
        return any(len(pitches.difference(SCALES[key])) == 0 for key in range(12))

    def get_fs(self) -> float:
        # get frequency of samping corresponding to a note with the minimum duration
        return 1 / self.beat_interval

    def determine_key(self, *, instrument_idx: int | None = None) -> int:
        """Return minor key number representation.

        Args:
            instrument_idx: If *None*, uses combined notes from all instruments.
        """
        # First, get all possible keys
        pitches = {note.pitch for note in self.get_notes(instrument_idx=instrument_idx)}
        keys = range(12)
        possible_keys = []
        for key in keys:
            if len(pitches.difference(SCALES[key])) == 0:
                possible_keys.append(key)  # Key matches
        # Pick the key
        if len(possible_keys) == 0:
            msg = "No key found"
            raise MIDIKeyError(msg)
        if len(possible_keys) == 1:
            return possible_keys[0]
        # prefer key close to A minor or C major
        differences = [abs(key - 9) for key in possible_keys]
        min_index = np.argmin(differences)
        return possible_keys[min_index]

    def transpose_to(self, input_key: int, target_key: int, *, instrument_idx: int | None = None) -> None:
        """Transpose notes.  If *instrument_idx* is None, transpose all instruments."""
        transpose_by = target_key - input_key
        if abs(transpose_by) >= 6:  # transpose length is smaller or equal to 6
            transpose_by = -(12 - transpose_by) if transpose_by > 0 else 12 + transpose_by

        logger.debug(f"Transposing by: {transpose_by}")
        if instrument_idx is None:
            for inst in self.instruments:
                for note in inst.notes:
                    note.pitch += transpose_by
        else:
            for note in self.instruments[instrument_idx].notes:
                note.pitch += transpose_by

    def normalize_key(self, *, instrument_idx: int | None = None) -> None:
        """Set the key to A minor (C major).

        Args:
            instrument_idx: If *None*, determines key from *all* instruments
                and transposes every instrument uniformly.  If given, determines
                key from and transposes only that instrument.
        """
        determ_key = self.determine_key(instrument_idx=instrument_idx)
        self.transpose_to(
            input_key=determ_key,
            target_key=pretty_midi.note_name_to_number("A-1"),
            instrument_idx=instrument_idx,
        )

    # ── shorten ──────────────────────────────────────────────────────────

    def shorten(self, columns) -> None:
        """Shorten all instruments to X columns."""
        min_duration = self.beat_interval
        end_time = columns * min_duration  # X columns in time
        for idx in range(len(self.instruments)):
            new_notes = []
            for note in self.get_notes(instrument_idx=idx):
                # Shorten length of melody
                if note.start > end_time:
                    continue
                new_notes.append(note)
            self.set_notes(new_notes, instrument_idx=idx)

    # ── preprocessing (per-instrument) ───────────────────────────────────

    def preprocess(
        self,
        columns: int,
        rows: int,
        octave_shift: OctaveShift = OctaveShift.UP,
        *,
        instrument_idx: int = 0,
    ) -> np.ndarray:
        """Preprocess a specific instrument into a ``(rows, columns)`` binary matrix.

        1. *(Global, once)* Shorten all instruments to *columns* time-steps.
        2. *(Per-instrument)* Normalize the key to A minor.
        3. *(Per-instrument)* Octave-shift notes.
        4. *(Per-instrument)* Generate the binary matrix.
        """
        # Global transforms — only run once per instance
        if not self._global_preprocessed:
            self.shorten(columns)
            self._global_preprocessed = True

        # Per-instrument key normalisation
        self.normalize_key(instrument_idx=instrument_idx)

        # Octave shift notes
        if octave_shift == OctaveShift.UP:
            self.octave_shift_up(rows, instrument_idx=instrument_idx)
        elif octave_shift == OctaveShift.DOWN:
            self.octave_shift_down(rows, instrument_idx=instrument_idx)
        else:
            raise ValueError(f"Invalid shift value: {octave_shift}")
    
        self.octave_shift_down(rows, instrument_idx=instrument_idx)
        matrix = self.generate_matrix(columns, rows, instrument_idx=instrument_idx)
        self.preprocessed_matrices[instrument_idx] = matrix
        self.preprocessed = True
        return matrix

    # ── octave shift (per-instrument) ────────────────────────────────────

    def octave_shift_down_notes(self, lowest_c_note: int, highest_note: int, *, instrument_idx: int = 0) -> None:
        """Shift notes within range (lowest_c_note, highest_note) element of ({0, ... 128}, {0, ... 128}).
        result is range (0, lowest_c_note - highest_note).
        """
        # TODO: lowest_c_note needs to be the lowest note or it does not work yet.
        new_notes = []
        pitch_limit = highest_note - lowest_c_note
        for note in self.get_notes(instrument_idx=instrument_idx):  # better way: find pitch where notes are the most.
            pitch = note.pitch
            pitch -= lowest_c_note  # relatively to c note bellow
            # octave shift higher notes down to range
            if pitch >= pitch_limit:
                diff = pitch - pitch_limit
                diff_mod = diff % 12
                inverse_diff = 12 - diff_mod
                pitch = pitch_limit - inverse_diff
            new_notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=pitch,
                    start=note.start,
                    end=note.end,
                ),
            )
        self.set_notes(new_notes, instrument_idx=instrument_idx)

    def octave_shift_up_notes(self, lowest_note: int, highest_c_note: int, *, instrument_idx: int = 0) -> None:
        """Shift notes within range (lowest_note, highest_c_note) element of ({0, ... 128}, {0, ... 128}).
        result is range (0, highest_c_note - lowest_note).
        """
        new_notes = []
        pitch_limit = highest_c_note - lowest_note
        for note in self.get_notes(instrument_idx=instrument_idx):
            pitch = note.pitch
            pitch -= lowest_note  # relatively to lowest note
            # octave shift lower notes up to range
            if pitch < 0:
                diff = abs(pitch)
                diff_mod = diff % 12
                inverse_diff = 12 - diff_mod
                pitch = inverse_diff
            new_notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=pitch,
                    start=note.start,
                    end=note.end,
                ),
            )
        self.set_notes(new_notes, instrument_idx=instrument_idx)

    @staticmethod
    def _get_pitch_limit_up(rows: int):
        """Defines maximum pitch range (0, pitch_limit) ascending from C.
        Row 0 is c, row 1 is d, row 2 is e... Returns semitones in the range.
        """
        result = 1  # accept c note by default
        for i in range(rows - 1):
            result += MAJOR_SCALE_DIFF[i % 7]
        return result

    @staticmethod
    def _get_pitch_limit_down(rows: int):
        """Defines maximum pitch range (0, pitch_limit) descending from C.
        Row -1 is c, row -2 is b, row -3 is a... Returns semitones in the range.
        """
        reversed_diff = list(reversed(MAJOR_SCALE_DIFF))
        result = 1  # accept c note by default
        for i in range(rows - 1):
            result += reversed_diff[i % 7]
        return result

    def octave_shift_down(self, rows: int, *, instrument_idx: int = 0) -> None:
        """Shift within range anchored at the bottom, shifting high notes down by octaves."""
        # determine lowest c note and highest note to shift
        lowest_note, _highest_note = lowest_highest_note(self.get_roll(instrument_idx=instrument_idx))
        c_note_under = (lowest_note // 12) * 12  # (complete octaves) * (semitones in an octave)
        pitch_limit = MIDIPreprocess._get_pitch_limit_up(rows)

        self.octave_shift_down_notes(c_note_under, c_note_under + pitch_limit, instrument_idx=instrument_idx)

    def octave_shift_up(self, rows: int, *, instrument_idx: int = 0) -> None:
        """Shift within range anchored at the top, shifting low notes up by octaves."""
        # determine highest c note and lowest note to shift
        _lowest_note, highest_note = lowest_highest_note(self.get_roll(instrument_idx=instrument_idx))
        c_note_above = ((highest_note // 12) + 1) * 12  # next C above highest note
        pitch_limit = MIDIPreprocess._get_pitch_limit_down(rows)

        print(f"To range: {c_note_above - pitch_limit} - {c_note_above}")
        self.octave_shift_down_notes(c_note_above - pitch_limit, c_note_above, instrument_idx=instrument_idx)

    # ── matrix generation (per-instrument) ───────────────────────────────

    def generate_matrix(self, columns: int, rows: int, *, instrument_idx: int = 0) -> np.ndarray:
        """Removes uneccessary notes outside of key for a specific instrument."""

        def simplify_piano_roll(piano_roll: np.ndarray, key: int = 9):
            # Removes empty rows (pitches outside of key), default key is A minor
            pitches_to_remove = set(range(128)) - set(SCALES[key])  # should be empty
            piano_roll.sum()
            piano_roll = np.delete(piano_roll, tuple(pitches_to_remove), axis=0)
            piano_roll.sum()
            # if sum_before != sum_after:  # There were notes off key
            #    raise MIDIInformationLossError
            return piano_roll

        piano_roll = self.get_roll(instrument_idx=instrument_idx)

        # Fill matrix columns with zeros
        if piano_roll.shape[1] < columns:
            zeros = np.zeros(shape=(piano_roll.shape[0], columns - piano_roll.shape[1]))
            piano_roll = np.concatenate([piano_roll, zeros], axis=1)

        # Remove spaces outside of key
        piano_roll = simplify_piano_roll(piano_roll)

        # Crop matrix
        piano_roll = piano_roll[:rows, :columns]

        # Clip high values (overlapping notes are usually the result of this, by default its 0-127) to 0-127
        piano_roll = np.clip(piano_roll, 0, 127)

        # Normalize values 0-1 (there must be 1)
        max_val = piano_roll.max()
        if max_val > 0:
            piano_roll = piano_roll / piano_roll.max()

        # TODO: Clean up, let us make the mask binary
        return np.where(piano_roll > 0.5, 1, 0)
