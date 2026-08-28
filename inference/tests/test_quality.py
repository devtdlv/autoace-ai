import soundfile as sf

from inference.pipeline import quality
from inference.pipeline.schema import AudioQuality
from inference.tests.conftest import REQUIRES_ESPEAK

_ORDER = {AudioQuality.clear: 0, AudioQuality.slightly_impaired: 1, AudioQuality.severely_impaired: 2}


def _load(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


@REQUIRES_ESPEAK
def test_clean_call_is_not_severely_impaired(clean_call):
    result = quality.analyze_quality(clean_call)
    assert result.quality != AudioQuality.severely_impaired


@REQUIRES_ESPEAK
def test_clipping_is_detected_directly(clean_call, clipped_call):
    clean_audio, _ = _load(clean_call)
    clipped_audio, _ = _load(clipped_call)
    assert quality._clipping_ratio(clipped_audio) > quality._clipping_ratio(clean_audio)


@REQUIRES_ESPEAK
def test_clean_call_scores_better_than_clipped(clean_call, clipped_call):
    clean = quality.analyze_quality(clean_call)
    clipped = quality.analyze_quality(clipped_call)
    assert _ORDER[clean.quality] < _ORDER[clipped.quality]


@REQUIRES_ESPEAK
def test_muffling_reduces_high_frequency_energy(clean_call, muffled_call):
    clean_audio, sr = _load(clean_call)
    muffled_audio, _ = _load(muffled_call)
    assert quality._high_freq_energy_ratio(muffled_audio, sr) < quality._high_freq_energy_ratio(clean_audio, sr)


@REQUIRES_ESPEAK
def test_clean_call_scores_better_than_muffled(clean_call, muffled_call):
    clean = quality.analyze_quality(clean_call)
    muffled = quality.analyze_quality(muffled_call)
    assert _ORDER[clean.quality] <= _ORDER[muffled.quality]


@REQUIRES_ESPEAK
def test_echo_filter_increases_echo_score(clean_call, echoey_call):
    clean_audio, sr = _load(clean_call)
    echoey_audio, _ = _load(echoey_call)
    assert quality._echo_score(echoey_audio, sr) > quality._echo_score(clean_audio, sr)
