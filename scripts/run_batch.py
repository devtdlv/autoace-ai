"""CLI batch runner — runs the full analysis pipeline over a folder of call
recordings without the dashboard. Exists as a reproducibility path: the
same `inference.pipeline` modules the dashboard (Milestone 6) calls through
its API, invoked directly, for scripted/offline batch runs or debugging a
single call outside the web flow.

Usage:
    inference/.venv/bin/python scripts/run_batch.py <folder-of-audio-files> --output results.json

Each result is the same Prediction schema (inference/pipeline/schema.py)
the API returns, keyed by input filename.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.pipeline import asr, noise, overlap, preprocess, quality, silence, vad
from inference.pipeline.aggregate import aggregate
from inference.pipeline.emotion import analyze_emotion
from inference.pipeline.prosody_features import extract_prosody_features, load_speech_only_audio
from inference.pipeline.schema import fallback_prediction

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus",
    ".webm", ".3gp", ".3gpp", ".amr", ".aiff", ".aif", ".au", ".mp4", ".mov",
}


def analyze_file(input_path: Path, tmp_dir: Path) -> dict:
    normalized = tmp_dir / f"{input_path.stem}.normalized.wav"
    preprocess.normalize_audio(input_path, normalized)
    normalized = str(normalized)

    segments = vad.detect_speech_segments(normalized)
    total_duration = vad.total_duration_s(normalized)
    transcript = asr.transcribe(normalized)
    prosody = extract_prosody_features(normalized, segments, transcript.words)
    speech_audio = load_speech_only_audio(normalized, segments)

    emotion_result = analyze_emotion(speech_audio, transcript.text, prosody, transcript.asr_confidence)
    noise_result = noise.analyze_noise(normalized, segments)
    quality_result = quality.analyze_quality(normalized)
    overlap_result = overlap.detect_overlap(normalized, segments)
    silence_result = silence.detect_long_silence(segments, total_duration)

    prediction = aggregate(
        emotion_result, noise_result, quality_result, overlap_result, silence_result,
        transcript.asr_confidence,
    )
    return prediction.model_dump()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder of call recordings")
    parser.add_argument("--output", type=Path, default=Path("batch_results.json"))
    args = parser.parse_args()

    files = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        print(f"No audio files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, path in enumerate(files, start=1):
            t0 = time.perf_counter()
            try:
                results[path.name] = analyze_file(path, tmp_dir)
                status = "ok"
            except Exception as exc:  # noqa: BLE001 — one bad file shouldn't abort the batch
                # Still a schema-valid (confidence=0.0) guess, not just an
                # error blob — see schema.fallback_prediction.
                results[path.name] = {**fallback_prediction().model_dump(), "error": str(exc)}
                status = "FAILED"
            dt = time.perf_counter() - t0
            print(f"[{i}/{len(files)}] {path.name}: {status} ({dt:.1f}s)")

    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
