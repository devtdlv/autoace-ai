"""Technical audio quality — independent of emotional tone and independent
of background noise presence (poor quality alone must not imply noise).

Milestone 4. Deterministic signal checks on the whole normalized wav
(deliberately not restricted to non-speech regions, unlike noise.py — a
distorted/clipped signal is a property of the recording chain, not of
what's happening in the room): clipping percentage, a muffled/telephone-
band check, dropout ("spectral gap") detection consistent with codec/
packet loss, and an echo/reverberation estimate. Combined into a single
score and bucketed; never reads noise.py's non-speech energy signal, so a
noisy-but-technically-clean recording and a quiet-but-distorted one score
independently, per the trial spec's requirement that these not be
conflated.
"""

from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

from inference.pipeline.schema import AudioQuality

FRAME_LENGTH = 1024
HOP_LENGTH = 256
CLIPPING_THRESHOLD = 0.998

# Standard telephone passband cutoff. Call audio is *expected* to be
# band-limited here — only a near-total absence of energy above this is
# evidence of extra degradation (muffled mic, aggressive lowpass), not the
# normal telephone band itself.
TELEPHONE_CUTOFF_HZ = 3400.0
DROPOUT_WINDOW_S = 0.1

QUALITY_SCORE_CLEAR_MIN = 0.75
QUALITY_SCORE_SLIGHT_MIN = 0.45


@dataclass
class QualityResult:
    quality: AudioQuality
    confidence: float


def _clipping_ratio(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= CLIPPING_THRESHOLD))


def _high_freq_energy_ratio(audio: np.ndarray, sr: int) -> float:
    if len(audio) < FRAME_LENGTH:
        return 1.0
    spec = np.abs(librosa.stft(audio, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)
    total = spec.sum()
    if total <= 0:
        return 0.0
    high = spec[freqs >= TELEPHONE_CUTOFF_HZ, :].sum()
    return float(high / total)


def _dropout_ratio(audio: np.ndarray, sr: int) -> float:
    """Fraction of short interior windows whose energy collapses relative to
    their immediate neighbors — a proxy for codec/packet-loss glitches, as
    distinct from ordinary pauses (which drop gradually across many windows,
    not as an isolated single-window collapse).
    """
    window = int(DROPOUT_WINDOW_S * sr)
    if window <= 0 or len(audio) < window * 3:
        return 0.0
    n_windows = len(audio) // window
    if n_windows < 3:
        return 0.0
    energies = np.array([
        np.sqrt(np.mean(audio[i * window:(i + 1) * window] ** 2) + 1e-12)
        for i in range(n_windows)
    ])
    dropouts = 0
    for i in range(1, n_windows - 1):
        neighbor_avg = (energies[i - 1] + energies[i + 1]) / 2.0
        if neighbor_avg > 1e-4 and energies[i] < 0.15 * neighbor_avg:
            dropouts += 1
    return dropouts / (n_windows - 2)


def _echo_score(audio: np.ndarray, sr: int) -> float:
    """Short-lag autocorrelation of the amplitude envelope. Real echo/reverb
    shows up as an above-floor correlation peak at a lag matching a typical
    echo delay (40-300ms); returns that peak's strength relative to the
    zero-lag value.
    """
    if len(audio) < sr:
        return 0.0
    envelope = np.abs(audio)
    envelope = envelope - envelope.mean()
    if np.allclose(envelope, 0):
        return 0.0
    autocorr = librosa.autocorrelate(envelope, max_size=int(0.3 * sr) + 1)
    if autocorr[0] <= 0:
        return 0.0
    lo, hi = int(0.04 * sr), int(0.3 * sr)
    if hi >= len(autocorr) or hi <= lo:
        return 0.0
    peak = float(np.max(autocorr[lo:hi]))
    return max(0.0, peak / autocorr[0])


def analyze_quality(wav_path: str) -> QualityResult:
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    clipping = _clipping_ratio(audio)
    high_freq_ratio = _high_freq_energy_ratio(audio, sr)
    dropout = _dropout_ratio(audio, sr)
    echo = _echo_score(audio, sr)

    # Each penalty in [0, 1], higher = worse. Thresholds are heuristic
    # starting points (see module docstring), calibrated later against
    # labeled/synthetic validation data.
    clipping_penalty = min(1.0, clipping / 0.02)                          # >2% clipped samples -> max
    muffled_penalty = min(1.0, max(0.0, (0.02 - high_freq_ratio) / 0.02)) # near-zero energy above 3.4kHz -> max
    dropout_penalty = min(1.0, dropout / 0.1)                             # >10% of windows dropping out -> max
    echo_penalty = min(1.0, max(0.0, (echo - 0.15) / 0.35))

    penalties = [clipping_penalty, muffled_penalty, dropout_penalty, echo_penalty]
    # Divide by 2, not 4: a single fully-maxed artifact should already tank
    # the score, not require all four to be bad.
    quality_score = 1.0 - min(1.0, sum(penalties) / 2.0)

    if quality_score >= QUALITY_SCORE_CLEAR_MIN:
        quality = AudioQuality.clear
    elif quality_score >= QUALITY_SCORE_SLIGHT_MIN:
        quality = AudioQuality.slightly_impaired
    else:
        quality = AudioQuality.severely_impaired

    # Confidence: distance from the nearest bucket boundary — a score right
    # at a threshold is a genuinely ambiguous call.
    boundaries = [0.0, QUALITY_SCORE_SLIGHT_MIN, QUALITY_SCORE_CLEAR_MIN, 1.0]
    distance_to_boundary = min(abs(quality_score - b) for b in boundaries)
    confidence = 0.5 + min(0.45, distance_to_boundary * 2.0)

    return QualityResult(quality=quality, confidence=confidence)
