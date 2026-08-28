# AutoAce AI — Voice Tone & Background Noise Analysis

Analyzes production call audio for emotional tone, emotional intensity,
background noise (presence/type/severity), audio quality, speaker overlap,
long silence, and an overall confidence score — via a fully local,
zero-external-API pipeline (no customer audio or derived text ever leaves
this infrastructure).

**Status: all 6 milestones implemented and tested** (backend: 41 automated
tests passing; dashboard: lint + production build clean, manually verified
end-to-end — login, upload, batch processing, review, export). See `docs/`
for the technical memo, validation results, cost/latency analysis, and
known limitations.

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

System dependencies: `ffmpeg` (audio normalization) and `espeak-ng` (only
needed to generate synthetic test audio for the test suite — not used at
inference time).

```bash
apt-get install -y ffmpeg espeak-ng

cd inference
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -r requirements-ml.txt   # faster-whisper, silero-vad, librosa, transformers
./.venv/bin/pip install -r requirements-dev.txt  # pytest, for running tests
cd ..  # back to repo root

# One-time: builds inference/models/noise_reference_bank.npz (spectral
# signatures for noise.py's noise-type matcher) from a curated ESC-50
# subset. Needs internet access; not needed again unless the bank is
# deleted or scripts/build_noise_reference_bank.py's category list changes.
inference/.venv/bin/python scripts/build_noise_reference_bank.py

inference/.venv/bin/uvicorn inference.api.main:app --host 127.0.0.1 --port 8001
```

Verify:
```bash
curl http://127.0.0.1:8001/health
# {"status":"ok","ffmpeg_available":true}
```

Run the pipeline test suite (generates synthetic speech via espeak-ng, runs
it through ffmpeg normalization -> VAD -> ASR, and checks the results):
```bash
PYTHONPATH=. inference/.venv/bin/pytest inference/tests/ -v
```

Note: run uvicorn and pytest from the **repository root** (not from inside
`inference/`) so the `inference.*` package imports resolve.

Heavier ML dependencies (faster-whisper, silero-vad; later librosa,
transformers, pyannote) live in `inference/requirements-ml.txt`, kept
separate from the fast-installing core service dependencies in
`requirements.txt`.

## Setup — dashboard

```bash
cd web
npm install
cp ../.env.example ../.env   # then edit — see below
npm run dev -- -p 3001
```

The dashboard (Next.js, port 3001 in dev) and the inference API (FastAPI,
port 8001) are separate processes. `web/next.config.ts` proxies `/api/*`
to the API for local dev (`AUTOACE_API_ORIGIN`, default
`http://127.0.0.1:8001`) so the browser sees them as one origin — in
production, nginx does this instead (`deploy/nginx.conf`) and the rewrite
never fires.

Login uses a single admin credential from env vars, not multi-user
accounts — see `.env.example` for `AUTOACE_ADMIN_USERNAME`,
`AUTOACE_ADMIN_PASSWORD` (dev) / `AUTOACE_ADMIN_PASSWORD_HASH` (prod), and
`AUTOACE_SECRET_KEY` (signs session cookies — set it in production so
sessions survive a restart). Source `.env` (or export the vars) in both
the `uvicorn` and `npm run dev` shells before starting them.

## Development plan / milestones

1. ✅ Repo scaffold, ffmpeg preprocessing, FastAPI + Next.js skeletons
2. ✅ VAD (silero-vad) + local ASR (faster-whisper)
3. ✅ Emotion path: prosody features + SER model + text-emotion model + fusion
4. ✅ Noise / audio-quality / overlap / silence modules + synthetic validation tests
5. ✅ Schema aggregation + confidence calibration + batch CLI (`scripts/run_batch.py`)
6. ✅ Dashboard: auth, upload/validate, batch processing, review, CSV/JSON export
7. ✅ Deployment (nginx + systemd + TLS) to autoace.tdlv.dev — **the running
   systemd services (`autoace-api`, `autoace-web`) still serve the code from
   before Milestones 3-6 landed; restarting them to deploy this work is a
   deliberate step that hasn't been taken yet** (see the note in this repo's
   latest session summary / ask before restarting production).

### Measured latency (2 vCPU / 3.8GB, no GPU — the actual deployment target)

See `docs/latency_analysis.md` for the full per-stage breakdown. Headline
numbers: ~1.14x real-time end-to-end (warm), ~2.3GB peak RSS with every
model loaded — both comfortably within budget.

## Deliverables index

| Deliverable | Location |
|---|---|
| Dashboard code (not yet deployed live — see Milestone 7 note above) | `web/`, proxied via `deploy/nginx.conf` once restarted |
| Technical memo | `docs/technical_memo.md` |
| Validation results | `docs/validation_results.md` |
| Cost analysis | `docs/cost_analysis.md` |
| Latency analysis | `docs/latency_analysis.md` |
| Failure modes / next steps | `docs/failure_modes_and_next_steps.md` |
| CLI reproducibility path | `scripts/run_batch.py` |
