"""Voice activity detection: speech vs. non-speech segmentation.

Milestone 2. Will use silero-vad (no auth token required, CPU-fast) to produce
a list of (start_s, end_s, is_speech) segments. Feeds: silence.py (gap
analysis), noise.py (non-speech regions are where background noise is
measured), and speaking-rate features in emotion.py.
"""

from dataclasses import dataclass


@dataclass
class SpeechSegment:
    start_s: float
    end_s: float


def detect_speech_segments(wav_path: str) -> list[SpeechSegment]:
    raise NotImplementedError("Milestone 2: silero-vad integration")
