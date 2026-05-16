"""Defines dataset classes for MIDI generation networks.

Multi-instrument support: each valid ``(file, instrument, segment)``
combination becomes a separate dataset item.  The cache format uses
three lines per file: filepath, segments, and instrument-segment pairs.
"""

import os
import warnings

import numpy as np
import torch
from mido.midifiles.meta import KeySignatureError  # pretty midi sometimes throws this error
from tqdm import tqdm

from riffly.processes.preprocess import MIDIPreprocess, OctaveShift
from riffly.utils.exceptions import MIDINoInstrumentsError, MIDINotConstantBeatIntervalError

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Tempo, Key or Time signature change events found on "
    "non-zero tracks.  This is not a valid type 0 or type 1 "
    "MIDI file.  Tempo, Key or Time Signature may be wrong.",
)


class MIDIDataset(torch.utils.data.Dataset):
    """Dataset loading MIDI's and returning preprocessed tensors.

    Each dataset item is a ``(file, instrument, segment)`` triple.
    ``file_segment_indexes`` stores 3-tuples
    ``(file_index, instrument_idx, segment_index)``.
    """

    CACHE_FOLDER_NAME = "midi_cache"

    def __init__(
        self,
        columns: int,
        rows: int,
        path: str | None = None,
        files: list[str] | None = None,
        flatten=True,
        load_cache: bool = True,
        limit_found_files: None | int = None,
        parent_cache_folder: str = "data",
        few_notes_count_threshold: int = 80,
        many_notes_threshold: int = 1000,
        min_unique_pitches: int = 3,
        octave_shift: OctaveShift = OctaveShift.UP,
    ) -> None:

        # Initialize variables
        self.columns = columns
        self.rows = rows
        self.flatten = flatten
        self.limit_found_files = None
        self.dataset_path = None
        self.cache_key = None
        self.cache_folder = None
        self.cache_path = None
        self.segments: list[list[tuple[int, int]]] = None
        self.file_segment_indexes = None  # list of (file_idx, instrument_idx, segment_idx)
        self.few_notes_count_threshold = few_notes_count_threshold
        self.many_notes_threshold = many_notes_threshold
        self.min_unique_pitches = min_unique_pitches
        self.octave_shift = octave_shift

        # If defined, load list of files from 'files' argument
        if files is not None:
            self.files = files
            return

        # Continue initializing variables
        path = path.replace("\\", "/")
        self.dataset_path = path
        # cache_key is defined by the folder — __multi suffix invalidates old caches
        self.cache_key = path.split("/")[-2] + "__" + path.split("/")[-1]
        self.cache_key += f"__{columns}columns__multi"
        if limit_found_files is not None:
            self.cache_key += f"__{limit_found_files}files"
        parent_cache_folder = os.path.abspath(parent_cache_folder.replace("\\", "/")).replace(
            "\\",
            "/",
        )
        self.cache_folder = parent_cache_folder + "/" + MIDIDataset.CACHE_FOLDER_NAME
        # Create the cache folder path recursively if it does not exist
        os.makedirs(self.cache_folder, exist_ok=True)
        self.cache_path = self.cache_folder + "/" + self.cache_key + ".mycache"
        self.limit_found_files = limit_found_files

        # Load cache (3 lines per file: filepath, segments, inst:seg pairs)
        if load_cache and os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as f:
                lines = f.read().splitlines()
            # Limit files
            if self.limit_found_files is not None:
                lines = lines[: (self.limit_found_files * 3)]  # lines come in groups of 3
            self.files = []
            self.segments = []
            self.file_segment_indexes = []
            for line_index in range(0, len(lines), 3):
                # Line 1: filepath
                filename = lines[line_index]
                self.files.append(filename)
                file_index = line_index // 3
                # Line 2: segments
                intervals = lines[line_index + 1]
                intervals = intervals.split(";")
                file_segments = []
                for interval in intervals:
                    start, end = interval.split(",")
                    file_segments.append((float(start), float(end)))
                self.segments.append(file_segments)
                # Line 3: instrument-segment pairs (inst_idx:seg_idx;...)
                pairs_str = lines[line_index + 2]
                for pair in pairs_str.split(";"):
                    inst_idx, seg_idx = pair.split(":")
                    self.file_segment_indexes.append((file_index, int(inst_idx), int(seg_idx)))

            self.files = [file.replace("\\", "/") for file in self.files]
            # check if the first one exists then exit
            if os.path.exists(self.files[0]):
                return
        if os.path.exists(parent_cache_folder) is False:
            x = input(f"Cache parent folder does not exist, create '{parent_cache_folder}'? (y/n)")
            if x.lower() == "y":
                os.mkdir(parent_cache_folder)
            else:
                msg = f"Cache folder at {parent_cache_folder} does not exist."
                raise FileNotFoundError(msg)

        # Select suitable files
        self.files = self.recursive_listdir()
        self.file_segment_indexes = self.filter_files()

        # Cache files (3 lines per file)
        if os.path.exists(self.cache_folder) is False:
            os.mkdir(self.cache_folder)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            for file_index, file in enumerate(self.files):
                # Line 1: filepath
                # Line 2: segments (start,end;start,end;...)
                # Line 3: instrument-segment pairs (inst_idx:seg_idx;...)
                intervals = ";".join(f"{segment[0]},{segment[1]}" for segment in self.segments[file_index])
                file_pairs = [
                    (inst_idx, seg_idx)
                    for fi, inst_idx, seg_idx in self.file_segment_indexes
                    if fi == file_index
                ]
                pairs_str = ";".join(f"{inst_idx}:{seg_idx}" for inst_idx, seg_idx in file_pairs)
                f.write(file.replace("\\", "/") + "\n" + intervals + "\n" + pairs_str + "\n")

    def filter_files(self):
        # If not cached filter midi file paths ↓
        remove_reasons = {
            "KeySignatureError": 0,
            "EOFError": 0,
            "OSError": 0,
            "NoKey": 0,
            "FewNotes": 0,
            "ManyNotes": 0,
            "InformationLost": 0,
            "UnknownError": 0,
            "MaxTickTooLarge": 0,
            "MIDINoInstrumentsError": 0,
            "MinUniquePitches": 0,
            "NotConstantBeatInterval": 0,
            "SEGMENT.MinUniquePitches": 0,
            "SEGMENT.FewNotes": 0,
            "SEGMENT.NoKey": 0,
            "SEGMENT.ManyNotes": 0,
        }
        remove_reasons["NotMIDI"] = self.exclude_non_midi_files()

        # Remove invalid MIDI's using preprocess class
        new_files = []
        new_segments = []
        file_segment_indexes = []
        file_index = 0
        file_count = 0
        pbar = tqdm(self.files, "Final checkup")
        for file in pbar:
            try:
                # Update status bar
                remove_reasons_non_zero = {key: value for key, value in remove_reasons.items() if value > 0}
                pbar.set_postfix(remove_reasons_non_zero)
                pbar.set_description(
                    f"Final checkup, files kept: {file_count}, total_segments: {len(file_segment_indexes)}",
                )

                # Convert to midi preprocess (loads ALL instruments)
                mp = MIDIPreprocess(file, verbose=False)
                if mp.beat_interval is None:
                    remove_reasons["NoBeatConstantInterval"] += 1
                    continue
            except KeySignatureError:
                remove_reasons["KeySignatureError"] += 1
                continue
            except MIDINoInstrumentsError:
                remove_reasons["MIDINoInstrumentsError"] += 1
                continue
            except EOFError:
                remove_reasons["EOFError"] += 1
                continue
            except MIDINotConstantBeatIntervalError:
                remove_reasons["NotConstantBeatInterval"] += 1
                continue
            except ValueError as e:
                # For example: ValueError: MIDI file has a largest tick of 4294968753, it is likely corrupt
                if "largest tick" in str(e) and "corrupt" in str(e):
                    remove_reasons["MaxTickTooLarge"] += 1
                    continue
                if "data byte must be in range 0..127" in str(e):
                    remove_reasons["InformationLost"] += 1
                    continue
                # Re-raise other ValueErrors that aren't this specific corruption
                raise e
            except OSError as e:
                if str(e) == "data byte must be in range 0..127":
                    remove_reasons["OSError"] += 1
                elif "MThd not found" in str(e):
                    remove_reasons["OSError"] += 1
                    continue
                elif "no MTrk header at start of track" in str(e):
                    remove_reasons["OSError"] += 1
                    continue
                else:
                    raise e
            except Exception as e:
                print(f"Unknown error {type(e)}: {e} for file {file}")
                raise e

            # Prepare object (global transforms apply to all instruments)
            mp.set_bpm_120()
            mp.quantize()

            # Segment operations (union time range across all instruments)
            segments: list[tuple[float, float]] = mp.extract_segments(self.columns)

            # Segment-level filters (keep segment if ANY instrument passes)

            # Only major/minor key
            before_key_filter = len(segments)
            segments = mp.get_segments_with_key(segments)
            after_key_filter = len(segments)
            remove_reasons["SEGMENT.NoKey"] += before_key_filter - after_key_filter

            # Few notes
            before_note_count_filter = len(segments)
            segments = mp.exclude_segments_with_few_notes(
                segments,
                few_notes_count_threshold=self.few_notes_count_threshold,
            )
            after_note_count_filter = len(segments)
            if before_note_count_filter > after_note_count_filter:
                remove_reasons["SEGMENT.FewNotes"] += before_note_count_filter - after_note_count_filter

            # Few pitches
            before_pitch_filter = len(segments)
            segments = mp.exclude_segments_with_few_pitches(segments, min_unique_pitches=self.min_unique_pitches)
            after_pitch_filter = len(segments)
            remove_reasons["SEGMENT.MinUniquePitches"] += before_pitch_filter - after_pitch_filter

            # Many notes
            before_many_notes_filter = len(segments)
            segments = mp.exclude_segments_with_many_notes(segments, many_notes_threshold=self.many_notes_threshold)
            after_many_notes_filter = len(segments)
            remove_reasons["SEGMENT.ManyNotes"] += before_many_notes_filter - after_many_notes_filter

            # Save segments and its file for __getitem__
            if len(segments) == 0:
                if before_note_count_filter > after_note_count_filter:
                    remove_reasons["FewNotes"]
                elif before_pitch_filter > after_pitch_filter:
                    remove_reasons["MinUniquePitches"]
                elif before_key_filter > after_key_filter:
                    remove_reasons["NoKey"]
                elif before_many_notes_filter > after_many_notes_filter:
                    remove_reasons["ManyNotes"]
                else:
                    raise Exception(f"No segments left but no filter reason found. File: {file}")

                continue

            # Per-instrument filtering: which (instrument, segment) pairs
            # individually pass all quality filters
            valid_pairs = mp.get_valid_instrument_segment_pairs(
                segments,
                few_notes_count_threshold=self.few_notes_count_threshold,
                many_notes_threshold=self.many_notes_threshold,
                min_unique_pitches=self.min_unique_pitches,
            )

            if len(valid_pairs) == 0:
                continue

            for inst_idx, seg_idx in valid_pairs:
                file_segment_indexes.append((file_index, inst_idx, seg_idx))
            file_index += 1
            new_segments.append(segments)
            new_files.append(file)
            file_count += 1


        self.files = new_files
        self.segments = new_segments

        initial_file_count = len(self.files)
        print(f"Found {initial_file_count}, kept {len(self.files)} midi files.")
        print(f"Total segments: {len(file_segment_indexes)}.")
        if initial_file_count != len(self.files):
            print(f"Removed files because of reasons: {remove_reasons}")
        return file_segment_indexes

    def recursive_listdir(self):
        file_list = []
        count = 0
        for root, _dirs, files in os.walk(self.dataset_path):
            for file in files:
                file_list.append(os.path.join(root, file))
                count += 1
                if self.limit_found_files is not None and count >= self.limit_found_files:
                    break
            if self.limit_found_files is not None and count >= self.limit_found_files:
                break
        return file_list

    def exclude_files_with_few_notes(self, mp: MIDIPreprocess) -> bool:
        # percentage of notes has to be higher than X percent
        matrix = mp.preprocess(columns=self.columns, rows=self.rows, octave_shift=self.octave_shift)
        # if unit is bigger than 0 make it 1
        matrix = np.where(matrix > 0.5, 1, 0)
        # if matrix is X% full keep
        if matrix.sum() / matrix.size > 0.02:
            return False  # Exclude
        return True

    def exclude_files_with_no_key(self, mp: MIDIPreprocess) -> bool:
        # remove files that are not in key
        return not mp.does_have_key()

    def exclude_non_midi_files(self):
        # remove files that are not midi
        new_files = []
        initial_file_count = len(self.files)
        file_count = 0
        # pbar = tqdm(self.files, "Excluding non-midi's")
        for file in self.files:
            # check extension
            if file.endswith(".mid"):
                new_files.append(file)
                file_count += 1
            else:
                pass
        # Show files kept (after the loop since it slows down the loop)
        # pbar.set_postfix({'kept': file_count})

        self.files = new_files
        return initial_file_count - len(self.files)

    def __len__(self) -> int:
        return len(self.file_segment_indexes)

    def __getitem__(self, idx) -> tuple[str, torch.Tensor]:
        """Get item of dataset.

        Each item corresponds to a ``(file, instrument, segment)`` triple.
        """
        file_index, instrument_idx, segment_index = self.file_segment_indexes[idx]
        midi_path = self.files[file_index]
        segment: tuple[float, float] = self.segments[file_index][segment_index]

        # Preprocess midi (loads all instruments, transforms are global)
        mp = MIDIPreprocess(midi_path, verbose=False)
        mp.set_bpm_120()
        mp.quantize()

        mp.set_segment(segment)
        mp.extend_if_short(self.columns)
        mp.shift_to_start()
        matrix = mp.preprocess(
            columns=self.columns,
            rows=self.rows,
            octave_shift=self.octave_shift,
            instrument_idx=instrument_idx,
        )

        # Convert to tensor
        matrix = matrix.astype(np.float32)
        tensor = torch.as_tensor(matrix)

        # Flatten 2D array to 1D if specified
        if self.flatten:
            tensor = tensor.flatten()
        return midi_path, tensor
