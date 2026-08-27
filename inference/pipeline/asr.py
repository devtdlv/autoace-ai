"""Local speech-to-text via faster-whisper (CTranslate2, CPU int8).

Milestone 2. Produces the transcript plus word-level timestamps, which feed:
- emotion.py (lexical/semantic signal via a text-emotion classifier)
- prosody_features.py (speaking rate = words / speech duration)

Model size is deliberately small ("small" or "base", int8 quantized) given
the 2 vCPU / 3.8GB deployment target — this is a batch-latency vs. accuracy
tradeoff that will be measured and documented in docs/latency_analysis.md.
"""

from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start_s: float
    end_s: float


@dataclass
class Transcript:
    text: str
    words: list[Word]
    language: str
    asr_confidence: float  # mean word-level logprob-derived confidence, 0-1


def transcribe(wav_path: str) -> Transcript:
    raise NotImplementedError("Milestone 2: faster-whisper integration")
