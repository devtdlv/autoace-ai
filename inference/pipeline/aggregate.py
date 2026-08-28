"""Assembles per-module results into the final Prediction schema and derives
the overall `confidence` field.

Milestone 5. `confidence` is NOT self-reported by any single model — it is
a weighted combination of the confidence each module already computed
(emotion sub-signal agreement, ASR confidence, quality/noise/overlap match
confidence), then scaled down further when measured audio quality is poor
— every downstream signal (ASR, prosody, SER) is genuinely noisier on bad
audio, independent of what each module reported about itself. Calibration
approach (why these weights, not fit ones) documented in
docs/validation_results.md.
"""

from inference.pipeline.emotion import EmotionResult
from inference.pipeline.noise import NoiseResult
from inference.pipeline.overlap import OverlapResult
from inference.pipeline.quality import QualityResult
from inference.pipeline.schema import AudioQuality, Prediction
from inference.pipeline.silence import SilenceResult

# Weights for the sub-confidence blend. Emotion and ASR dominate because
# they're the two signals most reliant on transcription actually working;
# noise/overlap match confidence contribute less since they're secondary,
# not headline, fields.
WEIGHT_EMOTION = 0.40
WEIGHT_ASR = 0.25
WEIGHT_QUALITY = 0.20
WEIGHT_NOISE = 0.10
WEIGHT_OVERLAP = 0.05

# Applied on top of the blend above — bad audio makes every upstream
# measurement less trustworthy, regardless of what each module self-reported.
QUALITY_PENALTY = {
    AudioQuality.clear: 1.0,
    AudioQuality.slightly_impaired: 0.9,
    AudioQuality.severely_impaired: 0.7,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def aggregate(
    emotion_result: EmotionResult,
    noise_result: NoiseResult,
    quality_result: QualityResult,
    overlap_result: OverlapResult,
    silence_result: SilenceResult,
    asr_confidence: float,
) -> Prediction:
    blended = (
        WEIGHT_EMOTION * emotion_result.tone_confidence
        + WEIGHT_ASR * _clamp01(asr_confidence)
        + WEIGHT_QUALITY * quality_result.confidence
        + WEIGHT_NOISE * noise_result.confidence
        + WEIGHT_OVERLAP * overlap_result.confidence
    )
    confidence = _clamp01(blended * QUALITY_PENALTY[quality_result.quality])

    return Prediction(
        emotional_tone=emotion_result.tone,
        emotional_intensity=emotion_result.intensity,
        background_noise_present=noise_result.present,
        background_noise_type=noise_result.noise_type,
        background_noise_severity=noise_result.severity,
        audio_quality=quality_result.quality,
        speaker_overlap_present=overlap_result.present,
        long_silence_present=silence_result.present,
        confidence=confidence,
    )
