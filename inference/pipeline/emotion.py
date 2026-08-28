"""Emotional tone + intensity.

Milestone 3. Fusion of three independent signals:
  1. Prosody features (prosody_features.py) — pitch variance/range, speaking
     rate, pause pattern, energy *contour* (never absolute energy — avoids
     the loudness-implies-frustration confound the trial spec warns
     against). Used only to set intensity/severity, not valence: prosody
     alone can't distinguish "excited-happy" from "agitated-angry".
  2. `superb/wav2vec2-base-superb-er` — small pretrained SER model
     (wav2vec2-base scale, 4-class: neu/hap/ang/sad), run on the
     speech-only audio (per vad.py's segments).
  3. `j-hartmann/emotion-english-distilroberta-base` — small pretrained
     text-emotion classifier (7-class Ekman), run on the ASR transcript —
     the lexical/semantic signal that catches cases with no prosodic
     marker (e.g. calm-voiced but clearly dissatisfied wording).

Fusion (classify_emotion) is a calibrated rule layer, not a model trained on
the 3 labeled samples (n=3 is not enough to fit parameters without
overfitting — see docs/validation_results.md). The 3 labeled calls, when
available, are used only to sanity-check the thresholds below after the
fact; the thresholds themselves are heuristic starting points, not fit
values.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from inference.pipeline.prosody_features import ProsodyFeatures
from inference.pipeline.schema import EmotionalIntensity, EmotionalTone

SER_MODEL_ID = "superb/wav2vec2-base-superb-er"
TEXT_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
SER_SAMPLE_RATE = 16000

# SER labels: neu, hap, ang, sad. Text labels: anger, disgust, fear, joy,
# neutral, sadness, surprise. `surprise` is deliberately NOT treated as
# negative — a surprised caller isn't necessarily a dissatisfied one.
SER_NEGATIVE = ("ang", "sad")
SER_POSITIVE = ("hap",)
TEXT_NEGATIVE = ("anger", "disgust", "fear", "sadness")
TEXT_POSITIVE = ("joy",)

# Prosody "agitation" thresholds — heuristic starting points (see module
# docstring), not fit on data. Revisit once labeled calls are available.
PITCH_RANGE_LOW_HZ = 60.0
PITCH_RANGE_HIGH_HZ = 150.0
ENERGY_STD_LOW = 0.02
ENERGY_STD_HIGH = 0.08

# Negative-bucket severity -> {frustrated, upset, distressed} thresholds.
NEGATIVE_SEVERITY_FRUSTRATED_MAX = 0.45
NEGATIVE_SEVERITY_UPSET_MAX = 0.70


@dataclass
class EmotionResult:
    tone: EmotionalTone
    intensity: EmotionalIntensity
    tone_confidence: float  # component confidence, feeds aggregate.py


@lru_cache(maxsize=1)
def _get_ser_model():
    # Cached process-wide, same pattern as asr._get_model / vad._get_model —
    # model load is the expensive part, reused across every clip in a batch.
    extractor = AutoFeatureExtractor.from_pretrained(SER_MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(SER_MODEL_ID)
    model.eval()
    return extractor, model


@lru_cache(maxsize=1)
def _get_text_model():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_ID)
    model.eval()
    return tokenizer, model


def run_speech_emotion(speech_audio: np.ndarray) -> dict[str, float]:
    """SER probabilities over the speech-only waveform (16kHz mono float32)."""
    extractor, model = _get_ser_model()
    labels = model.config.id2label
    if len(speech_audio) == 0:
        return {label: 1.0 / len(labels) for label in labels.values()}
    inputs = extractor(speech_audio, sampling_rate=SER_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = F.softmax(logits, dim=-1).tolist()
    return {labels[i]: p for i, p in enumerate(probs)}


def run_text_emotion(text: str) -> dict[str, float]:
    """Text-emotion probabilities over the ASR transcript."""
    tokenizer, model = _get_text_model()
    labels = model.config.id2label
    text = (text or "").strip()
    if not text:
        return {label: 1.0 / len(labels) for label in labels.values()}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = F.softmax(logits, dim=-1).tolist()
    return {labels[i]: p for i, p in enumerate(probs)}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _valence_buckets(probs: dict[str, float], negative_labels, positive_labels) -> dict[str, float]:
    negative = sum(probs.get(l, 0.0) for l in negative_labels)
    positive = sum(probs.get(l, 0.0) for l in positive_labels)
    neutral = _clamp01(1.0 - negative - positive)
    return {"negative": negative, "positive": positive, "neutral": neutral}


def _agitation_score(prosody: ProsodyFeatures) -> float:
    """0-1 heuristic 'how much acoustic energy/pitch movement' — sets
    intensity/severity, never valence (direction) on its own.
    """
    pitch_component = _clamp01(
        (prosody.pitch_range_hz - PITCH_RANGE_LOW_HZ) / (PITCH_RANGE_HIGH_HZ - PITCH_RANGE_LOW_HZ)
    )
    energy_component = _clamp01(
        (prosody.energy_rms_std - ENERGY_STD_LOW) / (ENERGY_STD_HIGH - ENERGY_STD_LOW)
    )
    return 0.5 * pitch_component + 0.5 * energy_component


def _intensity_from_agitation(agitation: float) -> EmotionalIntensity:
    if agitation > 0.66:
        return EmotionalIntensity.high
    if agitation > 0.33:
        return EmotionalIntensity.medium
    return EmotionalIntensity.low


def classify_emotion(
    prosody: ProsodyFeatures,
    speech_emotion_probs: dict[str, float],
    text_emotion_probs: dict[str, float],
    asr_confidence: float = 1.0,
) -> EmotionResult:
    ser = _valence_buckets(speech_emotion_probs, SER_NEGATIVE, SER_POSITIVE)
    text = _valence_buckets(text_emotion_probs, TEXT_NEGATIVE, TEXT_POSITIVE)

    # Lexical content is a more reliable dissatisfaction signal than a
    # wav2vec2-base SER model run alone — weight text higher, and more so
    # the more we trust the transcript (asr_confidence).
    text_weight = 0.55 + 0.2 * _clamp01(asr_confidence)
    ser_weight = 1.0 - text_weight

    combined = {
        bucket: ser_weight * ser[bucket] + text_weight * text[bucket]
        for bucket in ("negative", "positive", "neutral")
    }
    final_bucket = max(combined, key=combined.get)
    agitation = _agitation_score(prosody)

    if final_bucket == "neutral":
        tone = EmotionalTone.neutral
        intensity = EmotionalIntensity.low
    elif final_bucket == "positive":
        tone = EmotionalTone.satisfied
        intensity = _intensity_from_agitation(agitation)
    else:
        severity = 0.6 * combined["negative"] + 0.4 * agitation
        if severity < NEGATIVE_SEVERITY_FRUSTRATED_MAX:
            tone = EmotionalTone.frustrated
        elif severity < NEGATIVE_SEVERITY_UPSET_MAX:
            tone = EmotionalTone.upset
        else:
            tone = EmotionalTone.distressed
        intensity = _intensity_from_agitation(agitation)

    # Confidence = agreement across the emotion sub-signals (prosody has no
    # valence axis of its own, so this checks SER vs. text vs. the fused
    # decision — a calm-voiced-but-dissatisfied call is a legitimate case,
    # not something prosody disagreement should punish).
    ser_vote = max(ser, key=ser.get)
    text_vote = max(text, key=text.get)
    if ser_vote == text_vote == final_bucket:
        tone_confidence = 0.9
    elif final_bucket in (ser_vote, text_vote):
        tone_confidence = 0.65
    else:
        tone_confidence = 0.4

    return EmotionResult(tone=tone, intensity=intensity, tone_confidence=tone_confidence)


def analyze_emotion(
    speech_audio: np.ndarray,
    transcript_text: str,
    prosody: ProsodyFeatures,
    asr_confidence: float = 1.0,
) -> EmotionResult:
    """Full Milestone-3 path: run both models on real inputs, then fuse."""
    speech_emotion_probs = run_speech_emotion(speech_audio)
    text_emotion_probs = run_text_emotion(transcript_text)
    return classify_emotion(prosody, speech_emotion_probs, text_emotion_probs, asr_confidence)
