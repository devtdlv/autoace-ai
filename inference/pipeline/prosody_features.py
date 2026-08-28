"""Deterministic acoustic prosody features, extracted only from speech
segments (per vad.py) so that non-speech noise never contaminates the
emotion signal.

Milestone 3. Uses librosa: pitch (F0 via pyin), energy/RMS contour,
spectral centroid, pause ratio, and speaking rate (requires word timestamps
from asr.py). Deliberately excludes raw loudness/SNR as a feature — the
trial spec explicitly warns against inferring frustration from loudness
alone. `energy_rms_mean` is kept only for completeness/debugging;
emotion.py's fusion logic must only ever consume `energy_rms_std` (contour
variance), never the absolute mean, for exactly this reason.
"""

from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
FRAME_LENGTH = 1024
HOP_LENGTH = 256

# pyin's search range: typical human voice fundamental frequency, wide
# enough to cover male/female/child callers without picking up harmonics.
F0_MIN_HZ = 65.0
F0_MAX_HZ = 400.0


@dataclass
class ProsodyFeatures:
    pitch_mean_hz: float
    pitch_std_hz: float
    pitch_range_hz: float
    energy_rms_mean: float
    energy_rms_std: float
    speaking_rate_wps: float
    pause_ratio: float
    spectral_centroid_mean: float


def load_speech_only_audio(wav_path: str, speech_segments) -> np.ndarray:
    """Concatenate only the speech-segment samples — non-speech (including
    background noise) must never contaminate the prosody signal. Shared with
    emotion.py, which runs its SER model on the same speech-only audio.
    """
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{wav_path} is not {SAMPLE_RATE}Hz — run preprocess.normalize_audio first"
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    chunks = []
    for seg in speech_segments:
        start = max(0, int(seg.start_s * SAMPLE_RATE))
        end = min(len(audio), int(seg.end_s * SAMPLE_RATE))
        if end > start:
            chunks.append(audio[start:end])
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def _empty_features(pause_ratio: float) -> ProsodyFeatures:
    return ProsodyFeatures(
        pitch_mean_hz=0.0,
        pitch_std_hz=0.0,
        pitch_range_hz=0.0,
        energy_rms_mean=0.0,
        energy_rms_std=0.0,
        speaking_rate_wps=0.0,
        pause_ratio=pause_ratio,
        spectral_centroid_mean=0.0,
    )


def extract_prosody_features(wav_path: str, speech_segments, words) -> ProsodyFeatures:
    speech_audio = load_speech_only_audio(wav_path, speech_segments)
    speech_duration_s = sum(seg.end_s - seg.start_s for seg in speech_segments)
    total_duration_s = sf.info(wav_path).duration

    if len(speech_audio) < FRAME_LENGTH or speech_duration_s <= 0:
        # No usable speech (near-silent or entirely non-speech clip). Return
        # zeroed features rather than raising — downstream fusion treats a
        # zeroed prosody signal as low-confidence via tone_confidence, not
        # as a crash.
        pause_ratio = 1.0 if total_duration_s > 0 else 0.0
        return _empty_features(pause_ratio)

    f0, voiced_flag, _ = librosa.pyin(
        speech_audio,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=SAMPLE_RATE,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    voiced_mask = voiced_flag if voiced_flag is not None else ~np.isnan(f0)
    voiced_f0 = f0[voiced_mask & ~np.isnan(f0)]

    if len(voiced_f0) > 0:
        pitch_mean_hz = float(np.mean(voiced_f0))
        pitch_std_hz = float(np.std(voiced_f0))
        pitch_range_hz = float(np.max(voiced_f0) - np.min(voiced_f0))
    else:
        pitch_mean_hz = pitch_std_hz = pitch_range_hz = 0.0

    rms = librosa.feature.rms(
        y=speech_audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    energy_rms_mean = float(np.mean(rms))
    energy_rms_std = float(np.std(rms))

    centroid = librosa.feature.spectral_centroid(
        y=speech_audio, sr=SAMPLE_RATE, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    spectral_centroid_mean = float(np.mean(centroid))

    word_count = len(words)
    speaking_rate_wps = word_count / speech_duration_s if speech_duration_s > 0 else 0.0

    pause_ratio = (
        max(0.0, min(1.0, 1.0 - (speech_duration_s / total_duration_s)))
        if total_duration_s > 0
        else 0.0
    )

    return ProsodyFeatures(
        pitch_mean_hz=pitch_mean_hz,
        pitch_std_hz=pitch_std_hz,
        pitch_range_hz=pitch_range_hz,
        energy_rms_mean=energy_rms_mean,
        energy_rms_std=energy_rms_std,
        speaking_rate_wps=speaking_rate_wps,
        pause_ratio=pause_ratio,
        spectral_centroid_mean=spectral_centroid_mean,
    )
