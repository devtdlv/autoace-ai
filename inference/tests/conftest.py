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

API_TEST_USERNAME = "admin"
API_TEST_PASSWORD = "test-password-123"


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """FastAPI TestClient wired to an isolated SQLite DB + uploads dir per
    test, and a fixed admin credential, so API tests never touch the real
    inference/data/autoace.db.
    """
    monkeypatch.setenv("AUTOACE_ADMIN_USERNAME", API_TEST_USERNAME)
    monkeypatch.setenv("AUTOACE_ADMIN_PASSWORD", API_TEST_PASSWORD)
    monkeypatch.setenv("AUTOACE_SECRET_KEY", "test-secret-key")

    from inference.api import auth as auth_module
    from inference.api import batches as batches_module
    from inference.api import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(batches_module, "UPLOADS_DIR", tmp_path / "uploads")
    auth_module._admin_password_hash.cache_clear()
    auth_module._serializer.cache_clear()

    from fastapi.testclient import TestClient

    from inference.api.main import app

    with TestClient(app) as client:
        yield client

    auth_module._admin_password_hash.cache_clear()
    auth_module._serializer.cache_clear()


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


# --- Milestone 4 fixtures --------------------------------------------
# Degraded variants of a single base utterance, built with ffmpeg filters so
# each condition (clipping, muffling, echo, noise, long silence, overlap)
# is deterministic and depends on no external/confidential recordings —
# same philosophy as `synthetic_call` above.


@pytest.fixture(scope="session")
def _base_speech(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("base_speech")
    raw = tmp / "raw.wav"
    subprocess.run(
        ["espeak-ng", "-s", "150", "-w", str(raw),
         "Thanks for calling support. I wanted to follow up on my recent "
         "order and check the delivery date."],
        check=True, capture_output=True,
    )
    return str(raw)


def _render(tmp_path, name, raw_path, af_filter=None):
    out = tmp_path / f"{name}.wav"
    cmd = ["ffmpeg", "-y", "-i", raw_path]
    if af_filter:
        cmd += ["-af", af_filter]
    cmd += ["-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-loglevel", "error", str(out)]
    subprocess.run(cmd, check=True)
    return str(out)


@pytest.fixture(scope="session")
def clean_call(tmp_path_factory, _base_speech):
    tmp = tmp_path_factory.mktemp("clean_call")
    return _render(tmp, "clean", _base_speech)


@pytest.fixture(scope="session")
def clipped_call(tmp_path_factory, _base_speech):
    tmp = tmp_path_factory.mktemp("clipped_call")
    return _render(tmp, "clipped", _base_speech, af_filter="volume=25")


@pytest.fixture(scope="session")
def muffled_call(tmp_path_factory, _base_speech):
    tmp = tmp_path_factory.mktemp("muffled_call")
    return _render(tmp, "muffled", _base_speech, af_filter="lowpass=f=500")


@pytest.fixture(scope="session")
def echoey_call(tmp_path_factory, _base_speech):
    tmp = tmp_path_factory.mktemp("echoey_call")
    return _render(tmp, "echoey", _base_speech, af_filter="aecho=0.8:0.9:120:0.6")


@pytest.fixture(scope="session")
def noisy_call(tmp_path_factory, _base_speech):
    tmp = tmp_path_factory.mktemp("noisy_call")
    out = tmp / "noisy.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", _base_speech,
         "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.25:seed=42",
         "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[out]",
         "-map", "[out]", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
         "-loglevel", "error", str(out)],
        check=True,
    )
    return str(out)


@pytest.fixture(scope="session")
def long_silence_call(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("long_silence_call")
    part1, part2 = tmp / "part1.wav", tmp / "part2.wav"
    silence, raw, normalized = tmp / "silence.wav", tmp / "raw.wav", tmp / "normalized.wav"
    subprocess.run(
        ["espeak-ng", "-s", "150", "-w", str(part1), "Please hold for a moment."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["espeak-ng", "-s", "150", "-w", str(part2), "Thanks for waiting, I am back."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
         "-t", "12", "-loglevel", "error", str(silence)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(part1), "-i", str(silence), "-i", str(part2),
         "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
         "-map", "[out]", "-loglevel", "error", str(raw)],
        check=True,
    )
    normalize_audio(raw, normalized)
    return {"path": str(normalized)}


@pytest.fixture(scope="session")
def overlapping_call(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("overlapping_call")
    voice_a, voice_b, mixed = tmp / "a.wav", tmp / "b.wav", tmp / "mixed.wav"
    subprocess.run(
        ["espeak-ng", "-v", "en+m3", "-s", "150", "-w", str(voice_a),
         "I really need to speak to someone about my account right now."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["espeak-ng", "-v", "en+f3", "-s", "160", "-w", str(voice_b),
         "Sir I understand your frustration let me pull that up for you."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(voice_a), "-i", str(voice_b),
         "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0[out]",
         "-map", "[out]", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
         "-loglevel", "error", str(mixed)],
        check=True,
    )
    return str(mixed)
