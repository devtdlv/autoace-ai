from inference.pipeline import asr, vad
from inference.pipeline.emotion import analyze_emotion, classify_emotion
from inference.pipeline.prosody_features import (
    ProsodyFeatures,
    extract_prosody_features,
    load_speech_only_audio,
)
from inference.pipeline.schema import EmotionalIntensity, EmotionalTone
from inference.tests.conftest import REQUIRES_ESPEAK


def _prosody(pitch_range_hz=30.0, energy_rms_std=0.01, **overrides):
    defaults = dict(
        pitch_mean_hz=150.0,
        pitch_std_hz=10.0,
        pitch_range_hz=pitch_range_hz,
        energy_rms_mean=0.05,
        energy_rms_std=energy_rms_std,
        speaking_rate_wps=2.5,
        pause_ratio=0.2,
        spectral_centroid_mean=1500.0,
    )
    defaults.update(overrides)
    return ProsodyFeatures(**defaults)


def test_clear_positive_from_both_models_is_satisfied():
    ser = {"neu": 0.1, "hap": 0.8, "ang": 0.05, "sad": 0.05}
    text = {
        "joy": 0.85, "neutral": 0.05, "anger": 0.02, "disgust": 0.02,
        "fear": 0.02, "sadness": 0.02, "surprise": 0.02,
    }
    result = classify_emotion(_prosody(), ser, text, asr_confidence=0.9)
    assert result.tone == EmotionalTone.satisfied
    assert result.tone_confidence > 0.8


def test_clear_neutral_from_both_models_is_neutral():
    ser = {"neu": 0.85, "hap": 0.05, "ang": 0.05, "sad": 0.05}
    text = {
        "neutral": 0.8, "joy": 0.03, "anger": 0.03, "disgust": 0.03,
        "fear": 0.03, "sadness": 0.03, "surprise": 0.05,
    }
    result = classify_emotion(_prosody(), ser, text, asr_confidence=0.9)
    assert result.tone == EmotionalTone.neutral
    assert result.intensity == EmotionalIntensity.low


def test_negative_high_agitation_escalates_severity():
    ser = {"neu": 0.05, "hap": 0.0, "ang": 0.7, "sad": 0.25}
    text = {
        "anger": 0.6, "disgust": 0.15, "fear": 0.05, "joy": 0.0,
        "neutral": 0.05, "sadness": 0.1, "surprise": 0.05,
    }
    high_agitation = _prosody(pitch_range_hz=180.0, energy_rms_std=0.1)
    result = classify_emotion(high_agitation, ser, text, asr_confidence=0.9)
    assert result.tone in (EmotionalTone.upset, EmotionalTone.distressed)
    assert result.intensity == EmotionalIntensity.high


def test_calm_voiced_but_dissatisfied_wording_is_still_negative():
    # Per the module docstring's explicit design goal: lexical dissatisfaction
    # with no prosodic marker must still register as negative, not be masked
    # by low agitation.
    ser = {"neu": 0.6, "hap": 0.05, "ang": 0.2, "sad": 0.15}
    text = {
        "anger": 0.5, "disgust": 0.2, "fear": 0.05, "joy": 0.0,
        "neutral": 0.1, "sadness": 0.1, "surprise": 0.05,
    }
    calm = _prosody(pitch_range_hz=20.0, energy_rms_std=0.01)
    result = classify_emotion(calm, ser, text, asr_confidence=0.95)
    assert result.tone in (
        EmotionalTone.frustrated, EmotionalTone.upset, EmotionalTone.distressed,
    )
    assert result.intensity == EmotionalIntensity.low


def test_model_disagreement_lowers_confidence():
    disagreeing_ser = {"neu": 0.1, "hap": 0.8, "ang": 0.05, "sad": 0.05}  # positive
    disagreeing_text = {
        "anger": 0.6, "disgust": 0.1, "fear": 0.05, "joy": 0.05,
        "neutral": 0.1, "sadness": 0.05, "surprise": 0.05,
    }  # negative
    agreeing_ser = {"neu": 0.05, "hap": 0.0, "ang": 0.8, "sad": 0.15}
    agreeing_text = {
        "anger": 0.7, "disgust": 0.1, "fear": 0.05, "joy": 0.0,
        "neutral": 0.05, "sadness": 0.05, "surprise": 0.05,
    }

    disagreeing = classify_emotion(_prosody(), disagreeing_ser, disagreeing_text, asr_confidence=0.9)
    agreeing = classify_emotion(_prosody(), agreeing_ser, agreeing_text, asr_confidence=0.9)
    assert disagreeing.tone_confidence < agreeing.tone_confidence


@REQUIRES_ESPEAK
def test_prosody_extraction_runs_end_to_end_on_synthetic_call(synthetic_call):
    path = synthetic_call["path"]
    segments = vad.detect_speech_segments(path)
    transcript = asr.transcribe(path)

    features = extract_prosody_features(path, segments, transcript.words)

    assert features.pitch_mean_hz > 0
    assert features.speaking_rate_wps > 0
    assert 0.0 <= features.pause_ratio <= 1.0


@REQUIRES_ESPEAK
def test_full_emotion_path_runs_end_to_end_on_synthetic_call(synthetic_call):
    # espeak-ng TTS is flat/robotic — this asserts the real models load and
    # produce a schema-valid result, not any particular tone label.
    path = synthetic_call["path"]
    segments = vad.detect_speech_segments(path)
    transcript = asr.transcribe(path)
    prosody = extract_prosody_features(path, segments, transcript.words)
    speech_audio = load_speech_only_audio(path, segments)

    result = analyze_emotion(speech_audio, transcript.text, prosody, transcript.asr_confidence)

    assert isinstance(result.tone, EmotionalTone)
    assert isinstance(result.intensity, EmotionalIntensity)
    assert 0.0 <= result.tone_confidence <= 1.0
