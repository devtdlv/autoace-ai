"""Deterministic acoustic prosody features, extracted only from speech
segments (per vad.py) so that non-speech noise never contaminates the
emotion signal.

Milestone 3. Uses librosa: pitch (F0 via pyin), energy/RMS contour,
spectral centroid, jitter/shimmer-style pitch stability, pause ratio, and
speaking rate (requires word timestamps from asr.py). Deliberately excludes
raw loudness/SNR as a feature — the trial spec explicitly warns against
inferring frustration from loudness alone, so loudness-correlated features
are normalized out or omitted rather than fed directly to the tone decision.
"""

from dataclasses import dataclass


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


def extract_prosody_features(wav_path: str, speech_segments, words) -> ProsodyFeatures:
    raise NotImplementedError("Milestone 3: librosa prosody extraction")
