# AutoAce AI — Voice Tone & Background Noise Analysis

Analyzes production call audio for emotional tone, emotional intensity,
background noise (presence/type/severity), audio quality, speaker overlap,
long silence, and an overall confidence score — via a fully local,
zero-external-API pipeline (no customer audio or derived text ever leaves
this infrastructure).

**Status: under active development.** This README is updated as each
milestone lands; see `docs/` for the memo, validation results, cost/latency
analysis, and limitations once written.

## Repository layout

```
inference/          Python inference service (FastAPI + pipeline modules)
  pipeline/          Preprocessing, VAD, ASR, feature extraction, per-field
                     analysis modules, schema, aggregation
  api/               FastAPI app (batch upload/validate/process/status)
  validation/        Synthetic benchmark generation + validation runner
  tests/
web/                 Next.js dashboard (login, batch upload, review, export)
data/labeled_samples/  The 3 provided labeled calls (gitignored — confidential)
deploy/              systemd units + nginx config for autoace.tdlv.dev
scripts/             CLI batch runner (reproducibility path without the UI)
docs/                Technical memo, validation results, cost analysis,
                     latency analysis, failure modes & next steps
```

## Architecture (summary)

Local hybrid pipeline: ffmpeg normalization → VAD (speech/non-speech
segmentation) → local ASR (faster-whisper) → prosody feature extraction →
emotion fusion (prosody + small pretrained SER model + small pretrained
text-emotion model) → deterministic noise/quality/overlap/silence analysis
→ schema assembly with derived (not self-reported) confidence.

Emotional tone and background noise are deliberately computed from
non-overlapping feature sets: loudness/energy never directly drives the
emotion decision, and background-noise detection never uses the
audio-quality (distortion/clipping) signal as evidence, per the trial's
explicit requirement that these not be conflated.

Full rationale and the alternatives considered (including why a cloud
audio-native LLM was evaluated and rejected for production use) are in
`docs/technical_memo.md`.

## Setup — inference service

```bash
cd inference
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn inference.api.main:app --host 127.0.0.1 --port 8001
```

Verify:
```bash
curl http://127.0.0.1:8001/health
# {"status":"ok","ffmpeg_available":true}
```

Note: run uvicorn from the **repository root** (not from inside `inference/`)
so the `inference.*` package imports resolve — i.e.
`inference/.venv/bin/uvicorn inference.api.main:app ...` from `/var/www/autoace.tdlv.dev`.

Heavier ML dependencies (faster-whisper, librosa, transformers, pyannote)
are introduced in `inference/requirements-ml.txt` starting at Milestone 2,
kept separate from the fast-installing core service dependencies.

## Setup — dashboard

```bash
cd web
npm install
npm run dev -- -p 3001
```

## Development plan / milestones

1. ✅ Repo scaffold, ffmpeg preprocessing, FastAPI + Next.js skeletons
2. VAD (silero-vad) + local ASR (faster-whisper)
3. Emotion path: prosody features + SER model + text-emotion model + fusion
4. Noise / audio-quality / overlap / silence modules + synthetic validation set
5. Schema aggregation + confidence calibration + batch CLI
6. Dashboard: auth, upload/validate, batch processing, review, CSV/JSON export
7. Deployment (nginx + systemd + TLS) to autoace.tdlv.dev

## Deliverables index

| Deliverable | Location |
|---|---|
| Hosted dashboard | https://autoace.tdlv.dev (pending deployment, Milestone 7) |
| Technical memo | `docs/technical_memo.md` |
| Validation results + confusion matrix | `docs/validation_results.md` |
| Cost analysis | `docs/cost_analysis.md` |
| Latency analysis | `docs/latency_analysis.md` |
| Failure modes / next steps | `docs/failure_modes_and_next_steps.md` |
| CLI reproducibility path | `scripts/batch_cli.py` |
