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
