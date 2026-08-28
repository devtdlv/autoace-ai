"""Batch upload + processing endpoints.

Milestone 6. Upload a manifest (CSV, one `filename` column) plus either a
ZIP archive or a set of individual audio files; validates the manifest
against what was actually uploaded, stores everything under
inference/data/uploads/<batch_id>/, and kicks off a sequential background
job that runs each call through the full Milestone 1-5 pipeline, writing
progress to SQLite (db.py) as it goes.

Sequential, not parallel, by design — see docs/cost_analysis.md: more
workers would speed up wall-clock batch completion but not $/audio-minute,
and this is a 2 vCPU box, not a compute cluster.
"""

import csv
import io
import shutil
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, UploadFile

from inference.api import db
from inference.api.auth import require_session
from inference.pipeline import asr, noise, overlap, preprocess, quality, silence, vad
from inference.pipeline.aggregate import aggregate
from inference.pipeline.emotion import analyze_emotion
from inference.pipeline.prosody_features import extract_prosody_features, load_speech_only_audio

router = APIRouter()

UPLOADS_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

EXPORT_FIELDNAMES = [
    "filename", "status", "emotional_tone", "emotional_intensity",
    "background_noise_present", "background_noise_type", "background_noise_severity",
    "audio_quality", "speaker_overlap_present", "long_silence_present", "confidence", "error",
]


def _parse_manifest(raw_csv: bytes) -> list[str]:
    text = raw_csv.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "filename" not in reader.fieldnames:
        raise HTTPException(400, "Manifest CSV must have a 'filename' column")
    names = [row["filename"].strip() for row in reader if row.get("filename", "").strip()]
    if not names:
        raise HTTPException(400, "Manifest lists no files")
    return names


def _process_batch(batch_id: str, batch_dir: Path) -> None:
    db.set_batch_status(batch_id, "processing")
    for call in db.list_calls(batch_id):
        db.set_call_result(call.id, "processing")
        try:
            input_path = batch_dir / call.filename
            normalized = batch_dir / f"{input_path.stem}.normalized.wav"
            preprocess.normalize_audio(input_path, normalized)
            normalized_str = str(normalized)

            segments = vad.detect_speech_segments(normalized_str)
            total_duration = vad.total_duration_s(normalized_str)
            transcript = asr.transcribe(normalized_str)
            prosody = extract_prosody_features(normalized_str, segments, transcript.words)
            speech_audio = load_speech_only_audio(normalized_str, segments)

            emotion_result = analyze_emotion(speech_audio, transcript.text, prosody, transcript.asr_confidence)
            noise_result = noise.analyze_noise(normalized_str, segments)
            quality_result = quality.analyze_quality(normalized_str)
            overlap_result = overlap.detect_overlap(normalized_str, segments)
            silence_result = silence.detect_long_silence(segments, total_duration)

            prediction = aggregate(
                emotion_result, noise_result, quality_result, overlap_result, silence_result,
                transcript.asr_confidence,
            )
            db.set_call_result(call.id, "done", result=prediction.model_dump())
        except Exception as exc:  # noqa: BLE001 — one bad file must not abort the batch
            db.set_call_result(call.id, "failed", error=str(exc))

    db.set_batch_status(batch_id, "done")


@router.post("/batches")
async def create_batch(
    background_tasks: BackgroundTasks,
    manifest: UploadFile,
    archive: UploadFile | None = None,
    files: list[UploadFile] | None = None,
    user: str = Depends(require_session),
):
    manifest_filenames = _parse_manifest(await manifest.read())

    batch_id = uuid.uuid4().hex
    batch_dir = UPLOADS_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    uploaded_names: set[str] = set()
    if archive is not None:
        content = await archive.read()
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name or Path(member).suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                with zf.open(member) as src:
                    (batch_dir / name).write_bytes(src.read())
                uploaded_names.add(name)
    if files:
        for f in files:
            (batch_dir / f.filename).write_bytes(await f.read())
            uploaded_names.add(f.filename)

    missing = [name for name in manifest_filenames if name not in uploaded_names]
    if missing:
        raise HTTPException(400, f"Manifest references files that were not uploaded: {missing}")

    db.create_batch(
        batch_id, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), manifest.filename, manifest_filenames,
    )
    background_tasks.add_task(_process_batch, batch_id, batch_dir)

    return {"batch_id": batch_id, "total_calls": len(manifest_filenames)}


@router.get("/batches")
def get_batches(user: str = Depends(require_session)):
    return [vars(b) for b in db.list_batches()]


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str, user: str = Depends(require_session)):
    batch = db.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    return {"batch": vars(batch), "calls": [vars(c) for c in db.list_calls(batch_id)]}


@router.delete("/batches/{batch_id}")
def remove_batch(batch_id: str, user: str = Depends(require_session)):
    batch = db.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    db.delete_batch(batch_id)
    shutil.rmtree(UPLOADS_DIR / batch_id, ignore_errors=True)
    return {"ok": True}


@router.get("/batches/{batch_id}/export")
def export_batch(batch_id: str, format: str = "json", user: str = Depends(require_session)):
    batch = db.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    calls = db.list_calls(batch_id)

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=EXPORT_FIELDNAMES)
        writer.writeheader()
        for call in calls:
            row = {"filename": call.filename, "status": call.status, "error": call.error or ""}
            if call.result:
                row.update(call.result)
            writer.writerow(row)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{batch_id}.csv"'},
        )

    return {"batch": vars(batch), "calls": [vars(c) for c in calls]}
