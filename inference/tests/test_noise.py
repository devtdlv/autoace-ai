from inference.pipeline import noise, vad
from inference.pipeline.schema import BackgroundNoiseSeverity
from inference.tests.conftest import REQUIRES_ESPEAK


def test_reference_bank_loads():
    bank = noise._load_reference_bank()
    assert bank is not None
    names, vectors = bank
    assert len(names) > 0
    assert vectors.shape[0] == len(names)


@REQUIRES_ESPEAK
def test_clean_call_reports_no_noise(clean_call):
    segments = vad.detect_speech_segments(clean_call)
    result = noise.analyze_noise(clean_call, segments)
    assert result.present is False
    assert result.severity == BackgroundNoiseSeverity.none


@REQUIRES_ESPEAK
def test_noisy_call_reports_noise_present(noisy_call):
    segments = vad.detect_speech_segments(noisy_call)
    result = noise.analyze_noise(noisy_call, segments)
    assert result.present is True
    assert result.severity != BackgroundNoiseSeverity.none
