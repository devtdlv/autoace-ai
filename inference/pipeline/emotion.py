"""Emotional tone + intensity.

Milestone 3. Fusion of three independent signals:
  1. Prosody features (prosody_features.py) — pitch variance/range, speaking
     rate, pause pattern. Loudness/energy is used only in relative/contour
     form (e.g. variance), never as an absolute level, to avoid the
     loudness-implies-frustration confound the trial spec warns against.
  2. A small pretrained speech-emotion-recognition model (wav2vec2-base
     scale) run on the speech-only audio.
  3. A small pretrained text-emotion classifier run on the ASR transcript
     (lexical/semantic signal — catches cases with no prosodic marker,
     e.g. calm-voiced but clearly dissatisfied wording).

Fusion is a calibrated rule/logistic layer, not a model trained on the 3
labeled samples (n=3 is not enough to fit parameters without overfitting —
see docs/validation_results.md for the full rationale). The 3 labeled calls
are used only to sanity-check thresholds after the fact.
"""

from dataclasses import dataclass

from inference.pipeline.schema import EmotionalIntensity, EmotionalTone


@dataclass
class EmotionResult:
    tone: EmotionalTone
    intensity: EmotionalIntensity
    tone_confidence: float  # component confidence, feeds aggregate.py


def classify_emotion(prosody, speech_emotion_probs, text_emotion_probs) -> EmotionResult:
    raise NotImplementedError("Milestone 3: emotion fusion")
