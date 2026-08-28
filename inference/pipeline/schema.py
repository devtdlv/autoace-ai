"""Required prediction schema, per the AutoAce trial spec. Exact field names/enums/types."""

from enum import Enum

from pydantic import BaseModel, Field


class EmotionalTone(str, Enum):
    neutral = "neutral"
    satisfied = "satisfied"
    frustrated = "frustrated"
    upset = "upset"
    distressed = "distressed"


class EmotionalIntensity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class BackgroundNoiseSeverity(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"


class AudioQuality(str, Enum):
    clear = "clear"
    slightly_impaired = "slightly_impaired"
    severely_impaired = "severely_impaired"


class Prediction(BaseModel):
    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    background_noise_present: bool
    background_noise_type: str = ""
    background_noise_severity: BackgroundNoiseSeverity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)


def fallback_prediction() -> Prediction:
    """A schema-valid, conservative guess for when a clip couldn't be
    processed at all (corrupt/unsupported file, unexpected pipeline
    exception). confidence=0.0 is the honest signal here — every other
    field is a safe default, not a real inference. Used so one
    unprocessable hidden-set file yields a still-scoreable row instead of
    a bare gap, without ever pretending an unrecoverable failure is a real
    prediction.
    """
    return Prediction(
        emotional_tone=EmotionalTone.neutral,
        emotional_intensity=EmotionalIntensity.low,
        background_noise_present=False,
        background_noise_type="",
        background_noise_severity=BackgroundNoiseSeverity.none,
        audio_quality=AudioQuality.severely_impaired,
        speaker_overlap_present=False,
        long_silence_present=False,
        confidence=0.0,
    )
