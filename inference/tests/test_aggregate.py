from inference.pipeline import asr, noise, overlap, quality, silence, vad
from inference.pipeline.aggregate import aggregate
from inference.pipeline.emotion import EmotionResult, analyze_emotion
from inference.pipeline.noise import NoiseResult
from inference.pipeline.overlap import OverlapResult
from inference.pipeline.prosody_features import extract_prosody_features, load_speech_only_audio
from inference.pipeline.quality import QualityResult
from inference.pipeline.schema import (
    AudioQuality,
    BackgroundNoiseSeverity,
    EmotionalIntensity,
    EmotionalTone,
    Prediction,
    fallback_prediction,
)
from inference.pipeline.silence import SilenceResult
from inference.tests.conftest import REQUIRES_ESPEAK


def _results(quality_enum=AudioQuality.clear):
    emotion = EmotionResult(tone=EmotionalTone.frustrated, intensity=EmotionalIntensity.medium, tone_confidence=0.8)
    noise_result = NoiseResult(present=True, noise_type="engine / vehicle noise", severity=BackgroundNoiseSeverity.low, confidence=0.7)
    quality_result = QualityResult(quality=quality_enum, confidence=0.9)
    overlap_result = OverlapResult(present=False, confidence=0.6)
    silence_result = SilenceResult(present=False, longest_gap_s=1.5)
    return emotion, noise_result, quality_result, overlap_result, silence_result


def test_aggregate_maps_every_field():
    emotion, noise_result, quality_result, overlap_result, silence_result = _results()
    prediction = aggregate(emotion, noise_result, quality_result, overlap_result, silence_result, asr_confidence=0.85)

    assert isinstance(prediction, Prediction)
    assert prediction.emotional_tone == EmotionalTone.frustrated
    assert prediction.emotional_intensity == EmotionalIntensity.medium
    assert prediction.background_noise_present is True
    assert prediction.background_noise_type == "engine / vehicle noise"
    assert prediction.background_noise_severity == BackgroundNoiseSeverity.low
    assert prediction.audio_quality == AudioQuality.clear
    assert prediction.speaker_overlap_present is False
    assert prediction.long_silence_present is False
    assert 0.0 <= prediction.confidence <= 1.0


def test_severely_impaired_quality_lowers_confidence():
    clear_inputs = _results(quality_enum=AudioQuality.clear)
    impaired_inputs = _results(quality_enum=AudioQuality.severely_impaired)

    clear_prediction = aggregate(*clear_inputs, asr_confidence=0.85)
    impaired_prediction = aggregate(*impaired_inputs, asr_confidence=0.85)

    assert impaired_prediction.confidence < clear_prediction.confidence


def test_low_asr_confidence_lowers_overall_confidence():
    inputs = _results()
    high_asr = aggregate(*inputs, asr_confidence=0.95)
    low_asr = aggregate(*inputs, asr_confidence=0.2)
    assert low_asr.confidence < high_asr.confidence


@REQUIRES_ESPEAK
def test_full_pipeline_produces_a_valid_prediction(synthetic_call):
    path = synthetic_call["path"]

    segments = vad.detect_speech_segments(path)
    total_duration = vad.total_duration_s(path)
    transcript = asr.transcribe(path)
    prosody = extract_prosody_features(path, segments, transcript.words)
    speech_audio = load_speech_only_audio(path, segments)

    emotion_result = analyze_emotion(speech_audio, transcript.text, prosody, transcript.asr_confidence)
    noise_result = noise.analyze_noise(path, segments)
    quality_result = quality.analyze_quality(path)
    overlap_result = overlap.detect_overlap(path, segments)
    silence_result = silence.detect_long_silence(segments, total_duration)

    prediction = aggregate(
        emotion_result, noise_result, quality_result, overlap_result, silence_result,
        transcript.asr_confidence,
    )

    assert isinstance(prediction, Prediction)
    # Round-trips through the pydantic model / JSON schema cleanly.
    assert Prediction.model_validate(prediction.model_dump())


def test_fallback_prediction_is_schema_valid_with_zero_confidence():
    prediction = fallback_prediction()
    assert isinstance(prediction, Prediction)
    assert prediction.confidence == 0.0
    assert Prediction.model_validate(prediction.model_dump())
