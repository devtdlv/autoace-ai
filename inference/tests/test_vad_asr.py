from inference.pipeline import asr, vad
from inference.tests.conftest import REQUIRES_ESPEAK


@REQUIRES_ESPEAK
def test_vad_detects_the_injected_silence_gap(synthetic_call):
    segments = vad.detect_speech_segments(synthetic_call["path"])
    assert len(segments) >= 2

    gaps = [
        segments[i + 1].start_s - segments[i].end_s
        for i in range(len(segments) - 1)
    ]
    lo, hi = synthetic_call["expected_silence_gap_s"]
    assert any(lo <= gap <= hi for gap in gaps), (
        f"expected a silence gap in [{lo}, {hi}]s, got gaps={gaps}"
    )


@REQUIRES_ESPEAK
def test_asr_transcribes_expected_content(synthetic_call):
    transcript = asr.transcribe(synthetic_call["path"])
    lowered = transcript.text.lower()
    for fragment in synthetic_call["expected_text_fragments"]:
        assert fragment in lowered, f"'{fragment}' not found in: {transcript.text}"
    assert transcript.language == "en"
    assert 0.0 <= transcript.asr_confidence <= 1.0
