"""FastAPI inference service entrypoint.

Milestone 1: health check + preprocessing smoke-test endpoint only.
Batch upload/validate/process endpoints arrive in Milestone 6, once the
pipeline modules (Milestones 2-5) are real.
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from inference.pipeline.preprocess import PreprocessError, normalize_audio

app = FastAPI(title="AutoAce AI Inference Service")


class HealthResponse(BaseModel):
    status: str
    ffmpeg_available: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    import shutil

    return HealthResponse(status="ok", ffmpeg_available=shutil.which("ffmpeg") is not None)


@app.post("/debug/normalize")
async def debug_normalize(file: UploadFile):
    """Milestone-1 smoke test only: proves the ffmpeg normalize path works
    end-to-end through an HTTP upload. Not part of the batch workflow.
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / file.filename
        out_path = Path(tmp) / "normalized.wav"
        in_path.write_bytes(await file.read())
        try:
            normalize_audio(in_path, out_path)
        except PreprocessError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "normalized_size_bytes": out_path.stat().st_size}
