"""Includes a class with postprocessing processes so model's output can be played or exported."""

import random

import numpy as np
import pretty_midi

from riffly.constants import MAJOR_SCALE, Sounds808


class MIDIPostprocess:
    """Postprocesses the output of ML networks so it can be played or exported."""

    NOTE_DURATION = 2 ** (-3)

    def __init__(
        self,
        matrix,
        key_shift: int = 0,
        octave: int = 3,
        verbose: bool = False,
        connect_notes=True,
    ) -> None:
        self.matrix = matrix
        self.key_shift = key_shift
        self.octave = octave
        self.midi = pretty_midi.PrettyMIDI()
        self.instrument = pretty_midi.Instrument(
            program=pretty_midi.instrument_name_to_program(
                pretty_midi.constants.INSTRUMENT_MAP[0],
            ),
        )
        self.midi.instruments.append(self.instrument)
        self.columns = matrix.shape[1]

        if connect_notes:
            self.instrument.notes = self._convert_notes_connect(matrix)
        else:
            self.instrument.notes = self.convert_notes(matrix)

        if verbose:
            pass

    def convert_notes(self, matrix):
        """Converts the piano roll matrix to PrettyMIDI notes without connecting them."""
        notes = []
        for col in range(matrix.shape[1]):
            for row in range(matrix.shape[0]):
                if matrix[row, col] > 0.5:
                    pitch = self._get_pitch(row)
                    velocity = int(matrix[row, col] * 100)
                    notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=col * MIDIPostprocess.NOTE_DURATION,
                            end=(col + 1) * MIDIPostprocess.NOTE_DURATION,
                        ),
                    )
        return notes

    def _convert_notes_connect(self, matrix: np.ndarray) -> list[pretty_midi.Note]:
        """Converts the piano roll matrix to PrettyMIDI notes, connecting consecutive notes."""
        notes = []
        for row in range(matrix.shape[0]):
            start_time = None
            intensity_sum = 0
            note_count = 0
            for col in range(
                matrix.shape[1] + 1,
            ):  # + 1 to to allow for the note that ends at the end to be added
                if col < matrix.shape[1] and matrix[row, col] > 0.5:
                    if start_time is None:  # Start a new note
                        start_time = col * MIDIPostprocess.NOTE_DURATION
                        note_count = 1
                    else:  # Extending note
                        note_count += 1
                    intensity_sum += matrix[row, col]
                elif start_time is not None:
                    # Ending note
                    end_time = col * MIDIPostprocess.NOTE_DURATION
                    velocity = int(intensity_sum / note_count * 100)
                    pitch = self._get_pitch(row)
                    # Create note
                    notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start_time,
                            end=end_time,
                        ),
                    )
                    start_time = None
                    intensity_sum = 0
                    note_count = 0
        return notes

    def _get_pitch(self, row: int) -> int:
        """Gets the MIDI pitch number for a given row in the piano roll matrix."""
        pitch = MAJOR_SCALE[row % 7]  # Relative to one octave
        pitch += 12 * self.octave  # Shift up by octaves
        pitch += 12 * (row // 7)  # How many octaves does it have
        return int(pitch)  # Convert

    def _get_808_pattern(self, sound808: Sounds808 = Sounds808.PUNCHY, time_multiplier: int = 1) -> list[dict]:
        """Generates an 808 bass pattern based on the melody notes and sound type.

        For each 808 hit, finds the lowest melody note near that timestamp,
        constrains the pitch to A4–G#5, and returns the pattern as a list of
        note dicts with 'midi', 'time', and 'duration' keys.

        Times and durations are in raw NOTE_DURATION units (not speed-adjusted),
        matching the melody note times from the instrument.
        """
        x = MIDIPostprocess.NOTE_DURATION * time_multiplier  # One step in raw time
        total_duration808 = MIDIPostprocess.NOTE_DURATION * 16 * time_multiplier

        # Choose a rhythmic pattern based on sound type
        if sound808 == Sounds808.PUNCHY:
            if random.random() > 0.5:
                timestamps = [0, 3 * x, 6 * x, 10 * x, 13 * x]  # Excessive
            else:
                timestamps = [0, 3 * x, 10 * x, 13 * x]  # Hole in middle
        elif sound808 == Sounds808.SMOOTH:
            if random.random() > 0.5:
                timestamps = [0, 4 * x, 8 * x, 12 * x]  # 4 even
            else:
                timestamps = [0, 3 * x, 10 * x, 13 * x]  # Hole in middle
        else:
            raise ValueError(f"Invalid sound choice for 808 from class {Sounds808}.")

        # Create durations automatically (fill gaps between timestamps)
        durations808 = []
        intervals = timestamps + [16 * x]
        for i in range(len(timestamps)):
            durations808.append(intervals[i + 1] - intervals[i])

        # MIGHT REMOVE
        shifted_notes = [
            pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start * time_multiplier,
                end=note.end * time_multiplier,
            )
            for note in self.instrument.notes
        ]

        # Extend timestamps/durations across all 16-step blocks
        new_timestamps = []
        new_durations808 = []
        loops_of_808 = int(self.columns / 16 * time_multiplier)
        print(f"808 loops {loops_of_808} times")
        for i in range(loops_of_808):
            new_timestamps.extend([t + i * total_duration808 for t in timestamps])
            new_durations808.extend(durations808)
        timestamps = new_timestamps
        durations808 = new_durations808

        # Find the lowest melody pitch near each timestamp
        pitches = [float("-inf") for _ in range(len(timestamps))]
        for i, timestamp in enumerate(timestamps):
            if timestamp % total_duration808 != 0:
                start = timestamp - 1 * x
            else:
                start = timestamp
            # Basically start a bit earlier and end a bit earlier (checking)
            end = timestamp + 3 * x  # 3 * is the duration of the 808

            # Find the lowest pitch among notes playing during this window
            best_pitch = float("inf")
            found = False
            for note in shifted_notes:
                # Check if the note overlaps with the [start, end) window
                if note.start < end and note.end > start:
                    found = True
                    if note.pitch < best_pitch:
                        best_pitch = note.pitch
            if not found:
                continue
            pitches[i] = best_pitch

        # Fill -inf values with future 808 pitches
        is_filling = False
        for i in reversed(range(len(pitches))):
            if pitches[i] != float("-inf"):
                is_filling = True
            elif pitches[i] == float("-inf") and is_filling:
                pitches[i] = pitches[i + 1]

        # Fill remaining -inf values with past 808 pitches
        is_filling = False
        for i in range(len(pitches)):
            if pitches[i] != float("-inf"):
                is_filling = True
            elif pitches[i] == float("-inf") and is_filling:
                pitches[i] = pitches[i - 1]

        # Constrain pitches to A4–G#5 range via octave shifting
        A4 = 69
        G_sharp_5 = 80
        for i in range(len(pitches)):
            if pitches[i] == float("-inf"):
                continue
            while pitches[i] < A4:
                pitches[i] += 12
            while pitches[i] > G_sharp_5:
                pitches[i] -= 12

        # Build the pattern note list
        pattern = []
        for i, timestamp in enumerate(timestamps):
            if pitches[i] == float("-inf"):
                continue
            pattern.append(
                {
                    "midi": int(pitches[i]),
                    "time": timestamp,
                    "duration": durations808[i],
                },
            )
        return pattern

    def toJSON(
        self,
        time_multiplier: float = 1,
        add_808: bool = False,
        sound808: Sounds808 | None = None,
    ):
        """Exports the MIDI data to a JSON format.

        Example:
        {
            "notes": [
                {
                    "midi": 60,
                    "time": 0.0,
                    "duration": 0.5
                },
                ...
            ],
            "scale": "A minor/C major",
            "bpm": 120,
            "beats_per_bar": None,
            "units_per_beat": None,
            "max_time": 16.0,
            "808": {
                "notes": [{"midi": 72, "time": 0.0, "duration": 0.375}, ...],
                "type": "PUNCHY"
            }
        }

        """
        if add_808 and sound808 is None:
            sound808 = random.choice(list(Sounds808))
        data = {}
        data["notes"] = []
        data["scale"] = "A minor/C major"
        data["bpm"] = 120
        data["beats_per_bar"] = None
        data["units_per_beat"] = None
        data["note_duration"] = MIDIPostprocess.NOTE_DURATION * time_multiplier
        data["max_time"] = self.columns * MIDIPostprocess.NOTE_DURATION * time_multiplier

        for note in self.instrument.notes:
            data["notes"].append(
                {
                    "midi": note.pitch,
                    "time": note.start * time_multiplier,
                    "duration": (note.end - note.start) * time_multiplier,
                },
            )
        # Sort notes by time
        data["notes"] = sorted(data["notes"], key=lambda x: x["time"])

        if add_808:
            length = self.columns // 16
            pattern = self._get_808_pattern(sound808, time_multiplier=time_multiplier * length)
            type_name = sound808.value                   
            data["808"] = {
                "notes": [
                    {
                        "midi": note["midi"],
                        "time": note["time"],
                        "duration": note["duration"],
                    }
                    for note in pattern
                ],
                "type": type_name,
            }

        return data