from inference.pipeline import overlap, vad
from inference.tests.conftest import REQUIRES_ESPEAK


@REQUIRES_ESPEAK
def test_single_voice_call_has_low_candidate_ratio(clean_call):
    segments = vad.detect_speech_segments(clean_call)
    ratio, voiced_frames = overlap.candidate_frame_ratio(clean_call, segments)
    assert voiced_frames > 0
    assert ratio < overlap.CANDIDATE_FRAME_RATIO_THRESHOLD


@REQUIRES_ESPEAK
def test_two_simultaneous_voices_raise_the_candidate_ratio(clean_call, overlapping_call):
    clean_segments = vad.detect_speech_segments(clean_call)
    overlap_segments = vad.detect_speech_segments(overlapping_call)

    clean_ratio, _ = overlap.candidate_frame_ratio(clean_call, clean_segments)
    overlap_ratio, overlap_voiced_frames = overlap.candidate_frame_ratio(overlapping_call, overlap_segments)

    assert overlap_voiced_frames > 0
    # Heuristic fallback (see module docstring: pyannote unavailable without
    # an HF token) — directional only, not a hard present/absent claim.
    assert overlap_ratio > clean_ratio
