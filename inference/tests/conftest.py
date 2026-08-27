"""Synthetic test-audio fixture: two espeak-ng speech clips separated by a
silence gap, concatenated and run through our own preprocess.normalize_audio.

This is a stand-in for real production call audio until labeled samples are
available — it exercises the same code path (ffmpeg normalize -> VAD -> ASR)
without depending on any external/confidential recordings. Requires
espeak-ng and ffmpeg on PATH.
"""

import shutil
import subprocess

import pytest

from inference.pipeline.preprocess import normalize_audio

REQUIRES_ESPEAK = pytest.mark.skipif(
    shutil.which("espeak-ng") is None, reason="espeak-ng not installed"
)


@pytest.fixture(scope="session")
def synthetic_call(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synthetic_call")
    part1 = tmp / "part1.wav"
    part2 = tmp / "part2.wav"
    silence = tmp / "silence.wav"
    raw = tmp / "raw.wav"
    normalized = tmp / "normalized.wav"

    subprocess.run(
        ["espeak-ng", "-s", "150", "-w", str(part1),
         "Hi, thanks for calling support, how can I help you today."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["espeak-ng", "-s", "150", "-w", str(part2),
         "I have been on hold for twenty minutes and this is honestly ridiculous."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
         "-t", "4", "-loglevel", "error", str(silence)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(part1), "-i", str(silence), "-i", str(part2),
         "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
         "-map", "[out]", "-loglevel", "error", str(raw)],
        check=True,
    )
    normalize_audio(raw, normalized)
    return {
        "path": str(normalized),
        "expected_silence_gap_s": (4.0, 5.5),  # (min, max) tolerance band
        "expected_text_fragments": ["thanks for calling", "ridiculous"],
    }
