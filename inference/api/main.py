"""FastAPI inference service entrypoint.

/health and /debug/normalize are Milestone-1 smoke-test endpoints. The real
batch workflow (Milestone 6) lives in auth.py (login/session) and
batches.py (upload/process/review/export), both wired in below.
"""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference.api import batches, db
from inference.api.auth import (
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_S,
    create_session_cookie,
    require_session,
    verify_credentials,
)
from inference.pipeline.preprocess import PreprocessError, normalize_audio


@asynccontextmanager
async def _lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="AutoAce AI Inference Service", lifespan=_lifespan)

# Only needed for local dev, where the dashboard (localhost:3001) and API
# (localhost:8001) are different origins. In production, nginx serves both
# under autoace.tdlv.dev, so cookies work same-origin without CORS.
_dev_origins = os.environ.get("AUTOACE_DEV_CORS_ORIGINS", "http://localhost:3001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_dev_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(batches.router)


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


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def login(body: LoginRequest, response: Response):
    if not verify_credentials(body.username, body.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    cookie_secure = os.environ.get("AUTOACE_COOKIE_SECURE", "false").lower() == "true"
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=create_session_cookie(body.username),
        max_age=SESSION_MAX_AGE_S,
        httponly=True,
        samesite="lax",
        secure=cookie_secure,
    )
    return {"ok": True, "username": body.username}


@app.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@app.get("/auth/me")
def me(user: str = Depends(require_session)):
    return {"username": user}
