# Latency Analysis

Measured on the actual target deployment hardware: 2 vCPU, 3.8GB RAM, no
GPU. All models run CPU-only (faster-whisper int8, wav2vec2-base SER,
distilroberta text-emotion, silero-vad ONNX).

## Method

A ~19.4s synthetic espeak-ng utterance (same fixture family as
`inference/tests/conftest.py`) run through the full pipeline
(`preprocess → vad → asr → prosody_features → emotion → noise → quality →
overlap → silence → aggregate`), with all models already loaded (warm —
the realistic steady-state for batch processing, where model load happens
once per worker process, not once per call).

## Per-stage latency (warm, 19.4s clip)

| Stage              | Latency  | Notes |
|---------------------|---------:|-------|
| VAD                 |   295 ms | silero-vad, ONNX backend |
| ASR                 | 10,567 ms | faster-whisper "small", int8 — the single largest cost |
| Prosody features    |  2,291 ms | librosa pyin (F0) dominates this stage |
| Emotion fusion       |  7,720 ms | 2 transformer forward passes (SER + text-emotion) + fusion |
| Noise               |     3 ms | non-speech-region MFCC + cosine match |
| Audio quality       |    53 ms | clipping/spectral/dropout/echo checks |
| Speaker overlap     |  1,257 ms | per-frame pyin + harmonic-collision scan |
| Long silence        |    <1 ms | pure VAD-gap arithmetic |
| Aggregate           |    <1 ms | dataclass → schema assembly |
| **Total**           | **22,186 ms** | for 19.4s of audio |

**Cold-start note**: loading all 4 models (silero-vad, faster-whisper,
wav2vec2 SER, distilroberta text-emotion) the first time a process handles
a clip adds roughly another 12s, one-time per worker process — not
per-call. A long-running batch worker pays this once.

## Real-time factor

22.19s of processing for 19.43s of audio → **~1.14x real-time**, i.e.
~68.6 compute-seconds per audio-minute processed. This is *not* live/
streaming-viable on this hardware, but is comfortably batch-viable: the
deployment's nginx config already anticipates this
(`proxy_read_timeout 300s` on `/api/`), and the dashboard's batch model
(Milestone 6) processes calls sequentially in a background job, not
inline with the HTTP request.

## Peak memory

2.27GB RSS with every model loaded simultaneously (measured via
`/usr/bin/time -v`) — comfortably within the 3.8GB budget, leaving ~1.5GB
headroom for the FastAPI/uvicorn process itself, the OS, and the Next.js
dashboard process running alongside it on the same box.

## Where the time goes

ASR (48%) and emotion fusion (35%) account for 83% of total processing
time. If this pipeline needs to get faster, those are the two places to
optimize first — e.g. a smaller Whisper variant (accuracy tradeoff,
already the smallest viable per the model-selection rationale in
`docs/technical_memo.md`) or batching multiple calls' SER/text-emotion
forward passes together rather than one clip at a time (not implemented —
the batch dashboard currently processes calls sequentially; see
`docs/failure_modes_and_next_steps.md`).
