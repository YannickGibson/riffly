# Provide interface for processes submodule
from .interactive.postprocess import InteractivePostprocess
from .interactive.preprocess import InteractivePreprocess
from .multi_track import MultiTrackPostprocess, decompose_three_voices
from .piano_sampler import PianoSampler
from .postprocess import MIDIPostprocess
from .preprocess import MIDIPreprocess

__all__ = [
    "InteractivePostprocess",
    "InteractivePreprocess",
    "MIDIPostprocess",
    "MIDIPreprocess",
    "MultiTrackPostprocess",
    "PianoSampler",
    "decompose_three_voices",
]
