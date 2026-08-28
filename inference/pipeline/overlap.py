"""Speaker overlap detection — two+ speakers talking simultaneously enough
to affect understanding/analysis.

Milestone 4. pyannote.audio's overlapped-speech-detection pipeline — the
originally preferred, purpose-built option — is gated behind a Hugging Face
auth token + license acceptance not available in this environment. Per this
module's own documented escape hatch, this implements the stated fallback
instead: a multi-band energy/harmonic-collision heuristic. Per voiced
frame, a single speaker's energy in the speech band should concentrate near
their estimated f0 and its harmonics; energy elsewhere in the band that a
single voice's harmonic series can't explain is evidence of a second,
simultaneous voice. This is weaker in practice than a purpose-built OSD
model (documented in docs/failure_modes_and_next_steps.md) but requires no
external credentials.
"""

from dataclasses import dataclass

import librosa
import numpy as np

from inference.pipeline.prosody_features import load_speech_only_audio

SAMPLE_RATE = 16000
FRAME_LENGTH = 1024
HOP_LENGTH = 512
F0_MIN_HZ = 65.0
F0_MAX_HZ = 400.0
SPEECH_BAND_MAX_HZ = 3400.0
HARMONIC_BANDWIDTH_HZ = 60.0  # width of the "explained by this harmonic" window
N_HARMONICS = 8
MIN_FRAME_ENERGY = 1e-5  # ignore near-silent frames entirely

UNEXPLAINED_ENERGY_THRESHOLD = 0.55     # per-frame: candidate overlap frame
CANDIDATE_FRAME_RATIO_THRESHOLD = 0.08  # sustained across the call -> present


@dataclass
class OverlapResult:
    present: bool
    confidence: float


def candidate_frame_ratio(wav_path: str, speech_segments) -> tuple[float, int]:
    """Returns (candidate_ratio, voiced_frame_count) — the raw signal behind
    detect_overlap's present/confidence bucketing. Exposed separately so
    tests (and future calibration) can compare the continuous signal, not
    just the thresholded boolean.
    """
    speech_audio = load_speech_only_audio(wav_path, speech_segments)
    if len(speech_audio) < FRAME_LENGTH * 3:
        return 0.0, 0

    f0, voiced_flag, _ = librosa.pyin(
        speech_audio,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=SAMPLE_RATE,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    spec = np.abs(librosa.stft(speech_audio, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=FRAME_LENGTH)
    speech_band = freqs <= SPEECH_BAND_MAX_HZ

    n_frames = min(spec.shape[1], len(f0))
    voiced_frames = 0
    candidate_frames = 0
    for i in range(n_frames):
        if voiced_flag is not None and not voiced_flag[i]:
            continue
        if np.isnan(f0[i]):
            continue
        frame_spectrum = spec[:, i]
        total_energy = float(np.sum(frame_spectrum[speech_band] ** 2))
        if total_energy < MIN_FRAME_ENERGY:
            continue
        voiced_frames += 1

        harmonic_mask = np.zeros_like(freqs, dtype=bool)
        for h in range(1, N_HARMONICS + 1):
            center = f0[i] * h
            if center > SPEECH_BAND_MAX_HZ:
                break
            harmonic_mask |= np.abs(freqs - center) <= (HARMONIC_BANDWIDTH_HZ / 2)
        harmonic_mask &= speech_band

        harmonic_energy = float(np.sum(frame_spectrum[harmonic_mask] ** 2))
        unexplained_ratio = 1.0 - (harmonic_energy / total_energy)
        if unexplained_ratio >= UNEXPLAINED_ENERGY_THRESHOLD:
            candidate_frames += 1

    if voiced_frames == 0:
        return 0.0, 0
    return candidate_frames / voiced_frames, voiced_frames


def detect_overlap(wav_path: str, speech_segments) -> OverlapResult:
    candidate_ratio, voiced_frames = candidate_frame_ratio(wav_path, speech_segments)
    if voiced_frames == 0:
        return OverlapResult(present=False, confidence=0.5)

    present = candidate_ratio >= CANDIDATE_FRAME_RATIO_THRESHOLD
    distance = abs(candidate_ratio - CANDIDATE_FRAME_RATIO_THRESHOLD)
    confidence = 0.5 + min(0.4, distance * 3.0)

    return OverlapResult(present=present, confidence=confidence)
