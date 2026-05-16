"""Includes postprocessing version with interactive functionality for plotting and playing."""

from copy import deepcopy

import numpy as np

from riffly.constants import WAVE_FUNCTIONS, Sounds808, Wave
from riffly.plots import plot_piano_roll
from riffly.processes.postprocess import MIDIPostprocess
from riffly.synth import synth_808, synth_clap, synth_hat


class InteractivePostprocess(MIDIPostprocess):
    """Includes postprocessing version with interactive functionality for plotting and playing."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _synthesize_audio(
        self,
        wave: Wave = Wave.E_PIANO,
        time_multiplier: float = 1,
        repeat: int = 1,
        drop_repetitions: int = 0,
        clap_interval: float | None = 1,
        hat_interval: float | None = 0.125,
        add_808: bool = False,
        octave_shift: int = 0,
        sound808: Sounds808 = Sounds808.PUNCHY,
    ) -> np.ndarray:
        """Synthesizes audio with all effects applied.

        Args:
            drop_repetitions: Number of initial repetitions without any drums
                (hats, claps, 808). Must be <= repeat. Creates a build-up effect
                before the beat drops.
        """
        import librosa

        if drop_repetitions > repeat:
            raise ValueError(f"drop_repetitions ({drop_repetitions}) must be <= repeat ({repeat}).")

        wave_fn = WAVE_FUNCTIONS[wave]

        # Octave shift
        new_midi = deepcopy(self.midi)
        for note in new_midi.instruments[0].notes:
            note.pitch += octave_shift * 12
            note.pitch = max(0, min(127, note.pitch))  # clip

        # Get audio data from Pretty Midi
        audio_data = new_midi.synthesize(wave=wave_fn)

        # Subtract/Pad repeat offset (normalize to block)
        length_seconds = self.columns * MIDIPostprocess.NOTE_DURATION
        if len(self.instrument.notes) == 0:
            # audio_data is equal to empty data equal to length_seconds
            audio_data = np.zeros(int(44100 * length_seconds))
        else:
            max_end_time = max([note.end for note in self.instrument.notes])  # in seconds
            if max_end_time == length_seconds:
                repeat_offset = -1
            elif max_end_time == length_seconds - 0.5:
                repeat_offset = -0.5
            else:
                sample_time = len(audio_data) / 44100
                if sample_time < length_seconds:
                    repeat_offset = length_seconds - sample_time
                elif sample_time == length_seconds:
                    repeat_offset = 0
                else:  # sample_time > length_seconds
                    repeat_offset = length_seconds - sample_time  # Audio is longer than expected.
            repeat_offset_samples = int(repeat_offset * 44100)
            if repeat_offset_samples < 0:  # trim
                audio_data = audio_data[: len(audio_data) + repeat_offset_samples]
            elif repeat_offset_samples > 0:  # pad
                audio_data = np.pad(audio_data, (0, repeat_offset_samples))

        # Speed up audio while preserving pitch using librosa
        audio_data_fast = librosa.effects.time_stretch(y=audio_data, rate= 1/time_multiplier) * 0.1

        # Build the drop (no drums) and full (with drums) segments
        audio_no_drums = audio_data_fast.copy()

        # Add clap at interval in seconds
        if clap_interval is not None:
            clap_data = synth_clap(sr=44100)
            interval_samples = int(clap_interval * 44100)
            current_pos = interval_samples // 2
            while current_pos < len(audio_data_fast):
                end_pos = min(current_pos + len(clap_data), len(audio_data_fast))
                audio_data_fast[current_pos:end_pos] += clap_data[: end_pos - current_pos] * 0.1  # times loudness
                current_pos += interval_samples
        if hat_interval is not None:
            hat_data = synth_hat(sr=44100)
            interval_samples = int(hat_interval * 44100)
            current_pos = 0
            while current_pos < len(audio_data_fast):
                end_pos = min(current_pos + len(hat_data), len(audio_data_fast))
                audio_data_fast[current_pos:end_pos] += hat_data[: end_pos - current_pos] * 0.1
                current_pos += interval_samples
        if add_808:
            self._add_808(audio_data_fast, time_multiplier=time_multiplier, sound=sound808)

        # Repeat: first drop_repetitions without drums, then the rest with drums
        parts = []
        if drop_repetitions > 0:
            parts.append(np.tile(audio_no_drums, drop_repetitions))
        full_repetitions = repeat - drop_repetitions
        if full_repetitions > 0:
            parts.append(np.tile(audio_data_fast, full_repetitions))
        repeated_audio = np.concatenate(parts)
        return repeated_audio

    def play(
        self,
        wave: Wave = Wave.E_PIANO,
        time_multiplier: float = 1,
        repeat: int = 1,
        drop_repetitions: int = 0,
        clap_interval: float | None = 1,
        hat_interval: float | None = 0.125,
        add_808: bool = False,
        sound808: Sounds808 = Sounds808.PUNCHY,
        octave_shift: int = 0,
    ) -> None:
        """Plays the MIDI using sounddevice with options."""
        import sounddevice as sd

        audio = self._synthesize_audio(
            wave=wave,
            time_multiplier=time_multiplier,
            repeat=repeat,
            drop_repetitions=drop_repetitions,
            clap_interval=clap_interval,
            hat_interval=hat_interval,
            add_808=add_808,
            octave_shift=octave_shift,
            sound808=sound808,
        )

        self.last_audio = audio
        sd.play(audio, 44100, blocking=False)
        sd.wait()

    def export_beat(
        self,
        path: str,
        wave: Wave = Wave.E_PIANO,
        time_multiplier: float = 1,
        repeat: int = 1,
        drop_repetitions: int = 0,
        clap_interval: float | None = 1,
        hat_interval: float | None = 0.125,
        add_808: bool = False,
        octave_shift: int = 0,
        sound808: Sounds808 = Sounds808.PUNCHY,
    ) -> None:
        """Exports the beat with all effects to a WAV file."""
        import scipy

        audio = self._synthesize_audio(
            wave=wave,
            time_multiplier=time_multiplier,
            repeat=repeat,
            drop_repetitions=drop_repetitions,
            clap_interval=clap_interval,
            hat_interval=hat_interval,
            add_808=add_808,
            octave_shift=octave_shift,
            sound808=sound808,
        )
        scipy.io.wavfile.write(path, 44100, audio.astype(np.float32))

    def export_last_played(self, path: str) -> None:
        """Exports the last played audio to a WAV file."""
        import scipy

        scipy.io.wavfile.write(path, 44100, self.last_audio.astype(np.float32))

    def export_midi(self, path: str, time_multiplier: float = 1) -> None:
        """Exports the current MIDI to a MIDI file.

        Args:
            path: Output file path.
            time_multiplier: Stretches timing by this factor. Default 1 (no change).
                2 will make the MIDI 2x longer (slower), 0.5 will make it 2x shorter (faster).
        """
        if time_multiplier == 1:
            self.midi.write(path)
        else:
            midi_copy = deepcopy(self.midi)
            for instrument in midi_copy.instruments:
                for note in instrument.notes:
                    note.start *= time_multiplier
                    note.end *= time_multiplier
            midi_copy.write(path)

    def display(self, title=None) -> None:
        """Displays the piano roll of the current MIDI."""
        plot_piano_roll(
            self.matrix,
            stretch=False,
            preprocessed=True,
            quantize_interval=None,
            title=title,
        )

    def _add_808(self, audio_data, time_multiplier, sound: Sounds808) -> None:
        """Mix synthesized 808 hits at the rhythm pattern's timestamps."""
        # Reuse pattern logic from base class (returns times in raw NOTE_DURATION units)
        pattern = self._get_808_pattern(sound, time_multiplier=time_multiplier)

        speed = time_multiplier * 0.5
        for note in pattern:
            timestamp = note["time"] / speed
            duration_seconds = note["duration"] / speed
            pitch = int(note["midi"])

            hit = synth_808(midi_pitch=pitch, duration_sec=duration_seconds, sound=sound, sr=44100)

            # Tail fade-out (last 10%) to avoid overlap clicks when consecutive hits collide.
            fade_out_length = int(len(hit) * 0.10)
            if fade_out_length > 0:
                hit = hit.copy()
                hit[-fade_out_length:] *= np.linspace(1.0, 0.0, fade_out_length)

            start_sample = int(timestamp * 44100)
            if start_sample >= len(audio_data):
                continue
            end_sample = min(start_sample + len(hit), len(audio_data))
            audio_data[start_sample:end_sample] += hit[: end_sample - start_sample] * 0.3
