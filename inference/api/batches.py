"""Batch upload + processing endpoints.

Milestone 6, per the AutoAce trial's Section 7 spec: a batch is a folder or
ZIP containing audio files at the root plus one CSV manifest with `name`
(exact filename) and optional `result_json` (ground-truth, for labeled
batches) columns. Validates the batch — reporting both files the manifest
references but that weren't uploaded, and files that were uploaded but
aren't in the manifest — then kicks off a sequential background job that
runs each call through the full Milestone 1-5 pipeline, writing progress
to SQLite (db.py) as it goes.

A manifest CSV can be uploaded as its own field, OR embedded among the
audio files/inside the ZIP (matching the spec's single-upload shape) — the
first .csv found either way is used automatically. A manifest is not
strictly required outside of matching the graded workflow: with none at
all, every uploaded audio file is processed.

Sequential, not parallel, by design — see docs/cost_analysis.md: more
workers would speed up wall-clock batch completion but not $/audio-minute,
and this is a 2 vCPU box, not a compute cluster.
"""

import csv
import io
import json
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
    "audio_quality", "speaker_overlap_present", "long_silence_present", "confidence",
    "expected_json", "error",
]


def _parse_manifest(raw_csv: bytes) -> list[tuple[str, str | None]]:
    """Returns (name, expected_result_json_or_None) per manifest row.

    `name` is the spec's required column; `filename` is accepted too since
    that's what this dashboard itself used before this fix, and existing
    manifests built against it shouldn't break. `result_json`, if present,
    is validated as real JSON at upload time (fail loudly here, not later).
    """
    text = raw_csv.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    name_column = "name" if "name" in fieldnames else ("filename" if "filename" in fieldnames else None)
    if name_column is None:
        raise HTTPException(400, "Manifest CSV must have a 'name' column")

    entries: list[tuple[str, str | None]] = []
    for row in reader:
        name = (row.get(name_column) or "").strip()
        if not name:
            continue
        expected = (row.get("result_json") or "").strip() or None
        if expected is not None:
            try:
                json.loads(expected)
            except json.JSONDecodeError as exc:
                raise HTTPException(400, f"Manifest row for '{name}' has invalid result_json: {exc}") from exc
        entries.append((name, expected))

    if not entries:
        raise HTTPException(400, "Manifest lists no files")
    return entries


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
    manifest: UploadFile | None = None,
    archive: UploadFile | None = None,
    files: list[UploadFile] | None = None,
    user: str = Depends(require_session),
):
    batch_id = uuid.uuid4().hex
    batch_dir = UPLOADS_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    uploaded_names: set[str] = set()
    embedded_manifest_bytes: bytes | None = None

    def _ingest(name: str, content: bytes) -> None:
        nonlocal embedded_manifest_bytes
        suffix = Path(name).suffix.lower()
        if suffix == ".csv":
            # Matches the spec's single-upload shape: a folder/ZIP with the
            # manifest sitting alongside the audio files, not a separate
            # upload. First .csv found wins; an explicit `manifest` field
            # (checked below) still takes priority over this.
            if embedded_manifest_bytes is None:
                embedded_manifest_bytes = content
            return
        if suffix not in AUDIO_EXTENSIONS:
            return
        (batch_dir / name).write_bytes(content)
        uploaded_names.add(name)

    if archive is not None:
        content = await archive.read()
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name:
                    continue
                with zf.open(member) as src:
                    _ingest(name, src.read())
    if files:
        for f in files:
            _ingest(f.filename, await f.read())

    if not uploaded_names:
        raise HTTPException(400, "No audio files uploaded")

    manifest_bytes: bytes | None = None
    manifest_label = ""
    if manifest is not None and manifest.filename:
        manifest_bytes = await manifest.read()
        manifest_label = manifest.filename
    elif embedded_manifest_bytes is not None:
        manifest_bytes = embedded_manifest_bytes
        manifest_label = "manifest.csv"

    if manifest_bytes is not None:
        entries = _parse_manifest(manifest_bytes)
        manifest_names = {name for name, _ in entries}
        missing = sorted(manifest_names - uploaded_names)
        unmatched = sorted(uploaded_names - manifest_names)
        if missing or unmatched:
            parts = []
            if missing:
                parts.append(f"listed in the manifest but not uploaded: {missing}")
            if unmatched:
                parts.append(f"uploaded but not listed in the manifest: {unmatched}")
            raise HTTPException(400, "Batch validation failed — " + "; ".join(parts))
        batch_label = manifest_label
    else:
        entries = [(name, None) for name in sorted(uploaded_names)]
        batch_label = f"{len(entries)} file(s), no manifest"

    db.create_batch(
        batch_id, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), batch_label, entries,
    )
    background_tasks.add_task(_process_batch, batch_id, batch_dir)

    return {"batch_id": batch_id, "total_calls": len(entries)}


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
            if call.expected:
                row["expected_json"] = json.dumps(call.expected)
            writer.writerow(row)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{batch_id}.csv"'},
        )

    return {"batch": vars(batch), "calls": [vars(c) for c in calls]}
