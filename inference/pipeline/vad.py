"""Voice activity detection: speech vs. non-speech segmentation, via
silero-vad (ONNX backend — small, fast, no GPU, no HF auth token needed).

Feeds: silence.py (gap analysis), noise.py (non-speech regions are where
background noise is measured), and speaking-rate features in emotion.py.
"""

import wave
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

EXPECTED_SAMPLE_RATE = 16000


@dataclass
class SpeechSegment:
    start_s: float
    end_s: float


@lru_cache(maxsize=1)
def _get_model():
    # onnx=True: avoids loading a full torch model, cheaper at runtime on a
    # small CPU box. Cached process-wide — loaded once, reused across calls.
    return load_silero_vad(onnx=True)


def _load_mono_16k_tensor(wav_path: str) -> torch.Tensor:
    """Decode the already-normalized (preprocess.py) 16kHz mono PCM16 WAV
    ourselves via the stdlib `wave` module. Deliberately avoids
    silero_vad.read_audio / torchaudio — torchaudio's audio-backend layer is
    a known version-fragile dependency, and we don't need its generality
    since every file reaching this function has already been normalized.
    """
    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != EXPECTED_SAMPLE_RATE or wf.getsampwidth() != 2:
            raise ValueError(
                f"{wav_path} is not mono 16kHz PCM16 — run preprocess.normalize_audio first"
            )
        raw = wf.readframes(wf.getnframes())
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    float32 = (pcm16.astype(np.float32) / 32768.0).copy()
    return torch.from_numpy(float32)


def detect_speech_segments(wav_path: str) -> list[SpeechSegment]:
    model = _get_model()
    wav = _load_mono_16k_tensor(wav_path)
    timestamps = get_speech_timestamps(
        wav, model, sampling_rate=EXPECTED_SAMPLE_RATE, return_seconds=True
    )
    return [SpeechSegment(start_s=t["start"], end_s=t["end"]) for t in timestamps]


def total_duration_s(wav_path: str) -> float:
    wav = _load_mono_16k_tensor(wav_path)
    return len(wav) / EXPECTED_SAMPLE_RATE
