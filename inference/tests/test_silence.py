from inference.pipeline import vad
from inference.pipeline.silence import LONG_SILENCE_THRESHOLD_S, detect_long_silence
from inference.tests.conftest import REQUIRES_ESPEAK


class _Seg:
    def __init__(self, start_s, end_s):
        self.start_s = start_s
        self.end_s = end_s


def test_no_speech_segments_reports_full_duration_as_gap():
    result = detect_long_silence([], total_duration_s=20.0)
    assert result.present is True
    assert result.longest_gap_s == 20.0


def test_short_gaps_are_not_flagged():
    segments = [_Seg(0.0, 5.0), _Seg(6.0, 10.0), _Seg(11.0, 15.0)]
    result = detect_long_silence(segments, total_duration_s=15.0)
    assert result.present is False
    assert result.longest_gap_s < LONG_SILENCE_THRESHOLD_S


def test_long_interior_gap_is_flagged():
    segments = [_Seg(0.0, 5.0), _Seg(15.0, 20.0)]
    result = detect_long_silence(segments, total_duration_s=20.0)
    assert result.present is True
    assert result.longest_gap_s == 10.0


def test_long_trailing_gap_is_flagged():
    segments = [_Seg(0.0, 5.0)]
    result = detect_long_silence(segments, total_duration_s=20.0)
    assert result.present is True
    assert result.longest_gap_s == 15.0


@REQUIRES_ESPEAK
def test_real_long_silence_call_is_flagged(long_silence_call):
    path = long_silence_call["path"]
    segments = vad.detect_speech_segments(path)
    total = vad.total_duration_s(path)
    result = detect_long_silence(segments, total)
    assert result.present is True


@REQUIRES_ESPEAK
def test_real_clean_call_is_not_flagged(clean_call):
    segments = vad.detect_speech_segments(clean_call)
    total = vad.total_duration_s(clean_call)
    result = detect_long_silence(segments, total)
    assert result.present is False
