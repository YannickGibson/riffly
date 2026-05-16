"""3-voice (top melody / mid chord / bass) post-processing for piano-roll generations.

Splits a single 2D piano-roll matrix into a top voice (highest active pitch
per timestep, monophonic) and a mid voice (everything below the top), then
generates a bass voice through one of several modes (default: rhythmic 808
pattern derived from the existing ``MIDIPostprocess._get_808_pattern``
utility, transposed down into bass register). Bundles into one
``pretty_midi.PrettyMIDI`` with three ``Instrument``s and exports as a
multi-track ``.mid`` plus a mixed ``.wav`` synthesized per-voice with
distinct wave functions so the parts stay audibly separable.

Lives next to ``MIDIPostprocess`` / ``InteractivePostprocess`` rather than
modifying them — those still drive the canonical single-track exports and
are referenced by deployed code paths.

Bass modes (``bass_mode`` arg):
  - ``"808"`` (default): use the 808 rhythmic pattern from the full melody,
    transposed down ``bass_octave_shift`` octaves. Short, percussive notes;
    chord keeps every original active pitch except the top.
  - ``"off"``: silent bass track.
  - ``"lowest_carved"``: original behavior — lowest active row per timestep
    becomes bass and is removed from the chord.
  - ``"lowest_carved_octave_down"``: same as ``lowest_carved`` but the bass
    pitch is transposed down one octave so it sits below the chord.
"""
from __future__ import annotations

import numpy as np
import pretty_midi
import scipy.io.wavfile

from riffly.constants import WAVE_FUNCTIONS, Sounds808, Wave
from riffly.processes.piano_sampler import PianoSampler
from riffly.processes.postprocess import MIDIPostprocess


def decompose_three_voices(
    matrix: np.ndarray,
    threshold: float = 0.5,
    carve_bass: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a piano roll into (top, mid, bass) row-disjoint matrices.

    For each timestep ``t``:
      - top:  only the highest active row is kept (monophonic).
      - mid:  every active row strictly below the top is kept (polyphonic).
      - bass: empty if ``carve_bass`` is False (default — let the caller
        synthesize bass from a separate source like an 808 pattern).

    When ``carve_bass=True`` the lowest active row is moved into the bass
    matrix and removed from mid (legacy behavior; rarely the right musical
    choice — the bass line ends up doubling chord voicing).

    Active = ``matrix[r, t] > threshold``. Activations are preserved as-is so
    velocity downstream still sees the original values.
    """
    rows, cols = matrix.shape
    active = matrix > threshold
    top = np.zeros_like(matrix)
    mid = np.zeros_like(matrix)
    bass = np.zeros_like(matrix)
    for t in range(cols):
        rows_active = np.flatnonzero(active[:, t])
        if rows_active.size == 0:
            continue
        hi = int(rows_active[-1])
        top[hi, t] = matrix[hi, t]
        if not carve_bass:
            for r in rows_active:
                if r != hi:
                    mid[r, t] = matrix[r, t]
            continue
        lo = int(rows_active[0])
        if hi == lo:
            continue
        bass[lo, t] = matrix[lo, t]
        if rows_active.size > 2:
            inner = rows_active[1:-1]
            mid[inner, t] = matrix[inner, t]
    return top, mid, bass


# General MIDI program codes — give each voice a distinct timbre so the
# multi-track .mid sounds like 3 instruments when imported into a DAW.
PROGRAM_TOP = 80   # Lead 1 (square)
PROGRAM_MID = 0    # Acoustic Grand Piano
PROGRAM_BASS = 33  # Electric Bass (finger)

BASS_MODES = ("808", "off", "lowest_carved", "lowest_carved_octave_down")


def _scale_velocities(notes: list[pretty_midi.Note], scale: float) -> list[pretty_midi.Note]:
    """Return a copy of ``notes`` with ``velocity`` multiplied by ``scale``, clamped to [1, 127]."""
    out: list[pretty_midi.Note] = []
    for n in notes:
        v = max(1, min(127, int(round(n.velocity * scale))))
        out.append(pretty_midi.Note(velocity=v, pitch=n.pitch, start=n.start, end=n.end))
    return out


def _bass_notes_from_808(
    matrix: np.ndarray,
    octave: int,
    key_shift: int,
    connect_notes: bool,
    bass_octave_shift: int,
    sound808: Sounds808,
) -> list[pretty_midi.Note]:
    """Run ``MIDIPostprocess._get_808_pattern`` on the full matrix, transpose
    each pattern note down by ``bass_octave_shift`` octaves, return as
    pretty_midi notes ready to attach to an Instrument.
    """
    full = MIDIPostprocess(matrix, key_shift=key_shift, octave=octave, connect_notes=connect_notes)
    if not full.instrument.notes:
        return []
    pattern = full._get_808_pattern(sound808=sound808)
    notes: list[pretty_midi.Note] = []
    shift = 12 * bass_octave_shift
    for p in pattern:
        pitch = max(0, min(127, int(p["midi"]) + shift))
        notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=float(p["time"]),
                end=float(p["time"] + p["duration"]),
            ),
        )
    return notes


class MultiTrackPostprocess:
    """Take a single piano roll, split into 3 voices, expose multi-track export."""

    def __init__(
        self,
        matrix: np.ndarray,
        key_shift: int = 0,
        octave: int = 3,
        connect_notes: bool = True,
        bass_mode: str = "808",
        bass_octave_shift: int = -3,
        sound808: Sounds808 = Sounds808.PUNCHY,
        top_velocity_scale: float = 0.65,
        mid_velocity_scale: float = 1.25,
        bass_velocity_scale: float = 1.0,
    ) -> None:
        if bass_mode not in BASS_MODES:
            raise ValueError(f"bass_mode {bass_mode!r} not in {BASS_MODES}")

        self.matrix = matrix
        self.columns = matrix.shape[1]
        self.octave = octave
        self.bass_mode = bass_mode

        carve = bass_mode in ("lowest_carved", "lowest_carved_octave_down")
        top_mat, mid_mat, bass_mat = decompose_three_voices(matrix, carve_bass=carve)

        # Reuse MIDIPostprocess's note-builder per voice — gives us
        # connect_notes merging and the row→pitch math for free.
        self._top = MIDIPostprocess(top_mat, key_shift=key_shift, octave=octave, connect_notes=connect_notes)
        self._mid = MIDIPostprocess(mid_mat, key_shift=key_shift, octave=octave, connect_notes=connect_notes)

        if bass_mode == "off":
            bass_notes: list[pretty_midi.Note] = []
        elif bass_mode == "808":
            bass_notes = _bass_notes_from_808(
                matrix, octave=octave, key_shift=key_shift, connect_notes=connect_notes,
                bass_octave_shift=bass_octave_shift, sound808=sound808,
            )
        else:  # lowest_carved or lowest_carved_octave_down
            bass_proc = MIDIPostprocess(bass_mat, key_shift=key_shift, octave=octave, connect_notes=connect_notes)
            bass_notes = list(bass_proc.instrument.notes)
            if bass_mode == "lowest_carved_octave_down":
                for n in bass_notes:
                    n.pitch = max(0, min(127, n.pitch - 12))

        self.midi = pretty_midi.PrettyMIDI()
        self.top_instrument = pretty_midi.Instrument(program=PROGRAM_TOP, name="top")
        self.top_instrument.notes = _scale_velocities(self._top.instrument.notes, top_velocity_scale)
        self.mid_instrument = pretty_midi.Instrument(program=PROGRAM_MID, name="mid")
        self.mid_instrument.notes = _scale_velocities(self._mid.instrument.notes, mid_velocity_scale)
        self.bass_instrument = pretty_midi.Instrument(program=PROGRAM_BASS, name="bass")
        self.bass_instrument.notes = _scale_velocities(bass_notes, bass_velocity_scale)
        self.midi.instruments.extend([self.top_instrument, self.mid_instrument, self.bass_instrument])

    def export_midi(self, path: str) -> None:
        """Write the 3-track .mid file. ``pretty_midi`` preserves all instruments."""
        self.midi.write(path)

    def export_wav(
        self,
        path: str,
        top_wave: Wave = Wave.SQUARE,
        # NOTE: Wave.E_PIANO sounds broken / noisy in practice (see
        # riffly/constants.py — its formula has a modulo that zeroes half
        # the waveform). Do not use it here. SAWTOOTH is the chord default
        # when no PianoSampler is supplied.
        mid_wave: Wave = Wave.SAWTOOTH,
        bass_wave: Wave = Wave.SAWTOOTH,
        top_gain: float = 0.03,
        mid_gain: float = 0.65,
        bass_gain: float = 0.07,
        sr: int = 44100,
        mid_sampler: PianoSampler | None = None,
        repeat: int = 2,
    ) -> None:
        """Synthesize each voice with its own wave fn + gain, sum, write to .wav.

        We can't use ``self.midi.synthesize(wave=…)`` directly: it applies one
        wave to all instruments, which makes the parts indistinguishable. The
        per-voice gains (mid slightly hotter than top, bass quieter) are tuned
        by ear so the mix doesn't let the bass mask the melody.
        """
        # Optional sample-based piano render for the mid (chord) voice.
        audios: list[np.ndarray] = []
        if mid_sampler is not None and mid_sampler.available and self.mid_instrument.notes:
            audios.append(mid_sampler.render(self.mid_instrument.notes) * mid_gain)
            voices = [
                (self.top_instrument, top_wave, top_gain),
                (self.bass_instrument, bass_wave, bass_gain),
            ]
        else:
            voices = [
                (self.top_instrument, top_wave, top_gain),
                (self.mid_instrument, mid_wave, mid_gain),
                (self.bass_instrument, bass_wave, bass_gain),
            ]
        for inst, wave, gain in voices:
            if not inst.notes:
                continue
            single = pretty_midi.PrettyMIDI()
            single.instruments.append(inst)
            audios.append(single.synthesize(fs=sr, wave=WAVE_FUNCTIONS[wave]) * gain)
        if not audios:
            scipy.io.wavfile.write(path, sr, np.zeros(0, dtype=np.float32))
            return
        n = max(a.shape[0] for a in audios)
        mix = np.zeros(n, dtype=np.float64)
        for a in audios:
            mix[: a.shape[0]] += a
        # Trim trailing tail aggressively. Two-step:
        # 1) Cap to last melody/chord note end + 50 ms (bass-only tails feel
        #    like silence to a listener and stretch the file unnecessarily).
        # 2) Then walk backward from the end to the last sample louder than
        #    -20 dB of peak — anything quieter for >50 ms gets cut.
        melody_notes = list(self.top_instrument.notes) + list(self.mid_instrument.notes)
        if melody_notes:
            content_end = max(n.end for n in melody_notes)
            mix = mix[: min(int((content_end + 0.05) * sr), mix.shape[0])]
        peak = float(np.abs(mix).max())
        if peak > 0:
            floor = peak * 0.1  # -20 dB
            non_silent = np.flatnonzero(np.abs(mix) > floor)
            if non_silent.size > 0:
                last = int(non_silent[-1])
                mix = mix[: min(last + int(0.05 * sr), mix.shape[0])]
        # Tile the trimmed loop ``repeat`` times so the wav plays the song
        # back-to-back. Done AFTER trim so the seam between repetitions is
        # tight (no trailing silence between loops).
        if repeat > 1 and mix.shape[0] > 0:
            mix = np.tile(mix, repeat)
        scipy.io.wavfile.write(path, sr, mix.astype(np.float32))
