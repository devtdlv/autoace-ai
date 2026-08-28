"""Background noise: presence, type, severity.

Milestone 4. Deliberately consumes ONLY non-speech regions (per vad.py) and
the noise floor beneath speech — never audio_quality's distortion/clipping
signals — so noise detection cannot be a proxy for poor technical quality
(the trial spec explicitly separates these two).

Noise type is matched against a small canonical vocabulary via
spectral-signature (MFCC + spectral centroid/bandwidth) cosine similarity
against a reference bank built by scripts/build_noise_reference_bank.py from
a curated subset of ESC-50 (public environmental-sound data) — never from
the 3 labeled production calls. See that script for the exact category list
and rationale.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from inference.pipeline.schema import BackgroundNoiseSeverity

SAMPLE_RATE = 16000
N_MFCC = 13
MIN_NON_SPEECH_SAMPLES = int(0.5 * SAMPLE_RATE)
REFERENCE_BANK_PATH = Path(__file__).resolve().parent.parent / "models" / "noise_reference_bank.npz"

# Below this cosine similarity to every reference category, the match is
# too weak to name a noise_type confidently (presence/severity can still be
# reported from energy alone).
MATCH_CONFIDENCE_FLOOR = 0.55

# Relative RMS energy of non-speech regions vs. speech regions -> severity.
# A quiet line's non-speech energy floor sits well below this.
PRESENCE_FLOOR = 0.075
SEVERITY_LOW_MAX = 0.15
SEVERITY_MEDIUM_MAX = 0.40


@dataclass
class NoiseResult:
    present: bool
    noise_type: str  # "" when not present
    severity: BackgroundNoiseSeverity
    confidence: float


def _feature_vector(audio: np.ndarray, sr: int) -> np.ndarray:
    """Must stay in sync with scripts/build_noise_reference_bank.py's
    identically-named function — same feature family, or cosine similarity
    against the reference bank is meaningless.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    return np.concatenate([
        mfcc.mean(axis=1),
        [centroid.mean(), bandwidth.mean()],
    ]).astype(np.float32)


@lru_cache(maxsize=1)
def _load_reference_bank():
    if not REFERENCE_BANK_PATH.exists():
        return None
    data = np.load(REFERENCE_BANK_PATH, allow_pickle=False)
    return list(data["category_names"]), data["vectors"]


def _speech_mask(n_samples: int, sr: int, speech_segments) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for seg in speech_segments:
        start = max(0, int(seg.start_s * sr))
        end = min(n_samples, int(seg.end_s * sr))
        mask[start:end] = True
    return mask


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def analyze_noise(wav_path: str, speech_segments) -> NoiseResult:
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    mask = _speech_mask(len(audio), sr, speech_segments)
    speech, non_speech = audio[mask], audio[~mask]
    speech_energy = float(np.sqrt(np.mean(speech ** 2))) if len(speech) > 0 else 0.0

    if len(non_speech) < MIN_NON_SPEECH_SAMPLES or speech_energy <= 1e-6:
        # Not enough non-speech audio (or no speech to compare against) to
        # say anything meaningful — report absent rather than guess.
        return NoiseResult(present=False, noise_type="", severity=BackgroundNoiseSeverity.none, confidence=0.5)

    non_speech_energy = float(np.sqrt(np.mean(non_speech ** 2)))
    relative_energy = min(1.0, non_speech_energy / speech_energy)
    present = relative_energy > PRESENCE_FLOOR

    if not present:
        return NoiseResult(present=False, noise_type="", severity=BackgroundNoiseSeverity.none, confidence=0.7)

    if relative_energy <= SEVERITY_LOW_MAX:
        severity = BackgroundNoiseSeverity.low
    elif relative_energy <= SEVERITY_MEDIUM_MAX:
        severity = BackgroundNoiseSeverity.medium
    else:
        severity = BackgroundNoiseSeverity.high

    bank = _load_reference_bank()
    if bank is None:
        # scripts/build_noise_reference_bank.py hasn't been run yet — still
        # report presence/severity from energy alone, just without a
        # noise_type label, reflected honestly via confidence.
        return NoiseResult(present=True, noise_type="", severity=severity, confidence=0.5)

    category_names, vectors = bank
    query = _feature_vector(non_speech, sr)
    similarities = [_cosine_similarity(query, v) for v in vectors]
    best_idx = int(np.argmax(similarities))
    best_similarity = similarities[best_idx]

    if best_similarity >= MATCH_CONFIDENCE_FLOOR:
        noise_type = category_names[best_idx]
        confidence = 0.5 + min(0.45, (best_similarity - MATCH_CONFIDENCE_FLOOR) * 1.5)
    else:
        noise_type = ""
        confidence = 0.5

    return NoiseResult(present=True, noise_type=noise_type, severity=severity, confidence=confidence)
