"""Exceptions used across the project."""


class MIDIError(Exception):
    pass


class MIDIKeyError(MIDIError):
    pass


class MIDIInformationLostError(MIDIError):
    pass


class MIDINoInstrumentsError(MIDIError):
    pass


class MIDINotConstantBeatIntervalError(MIDIError):
    """Is raised if the loaded MIDI file has variable beat stamps or has no beat stamps."""
