"""Local speech-to-text via faster-whisper (CTranslate2, CPU int8).

Model size ("small", int8 quantized) is a deliberate latency/accuracy
tradeoff for the 2 vCPU / 3.8GB deployment target — measured and documented
in docs/latency_analysis.md. Produces the transcript plus word-level
timestamps, which feed:
  - emotion.py (lexical/semantic signal via a text-emotion classifier)
  - prosody_features.py (speaking rate = words / speech duration)
"""

import math
from dataclasses import dataclass
from functools import lru_cache

from faster_whisper import WhisperModel

MODEL_SIZE = "small"


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
    asr_confidence: float  # derived from mean word-level logprob, 0-1


@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    # Cached process-wide — model load is the expensive part, reused across
    # every clip in a batch rather than reloaded per file.
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe(wav_path: str) -> Transcript:
    model = _get_model()
    segments, info = model.transcribe(wav_path, word_timestamps=True)

    words: list[Word] = []
    text_parts: list[str] = []
    logprobs: list[float] = []
    for segment in segments:
        text_parts.append(segment.text)
        if segment.avg_logprob is not None:
            logprobs.append(segment.avg_logprob)
        for w in segment.words or []:
            words.append(Word(text=w.word, start_s=w.start, end_s=w.end))

    # avg_logprob is a per-segment log-probability (typically in roughly
    # [-1, 0] for confident transcriptions); map to a 0-1 confidence via a
    # bounded exponential rather than a raw linear rescale, since logprob
    # isn't linearly related to "confidence" as a probability.
    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else -1.0
    asr_confidence = max(0.0, min(1.0, math.exp(mean_logprob)))

    return Transcript(
        text=" ".join(p.strip() for p in text_parts).strip(),
        words=words,
        language=info.language,
        asr_confidence=asr_confidence,
    )
