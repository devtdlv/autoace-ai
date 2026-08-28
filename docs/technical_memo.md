# Technical Memo

## Approaches compared

Two architectures were considered for the analysis pipeline:

1. **A cloud audio-native LLM** (send call audio to a hosted multimodal
   model, ask it to return the structured fields directly). Rejected for
   production use: the trial spec requires that customer audio and
   derived text never leave this infrastructure, which a cloud API
   categorically cannot satisfy regardless of that provider's own privacy
   terms. It would also make cost proportional to a third party's per-
   minute pricing rather than this server's own compute, and would remove
   the ability to independently verify *why* a given field was predicted
   (no per-field feature control, no way to enforce the confound-
   decorrelation requirement below).
2. **A local hybrid pipeline** (chosen): ffmpeg normalization → VAD
   (silero-vad) → local ASR (faster-whisper) → deterministic prosody
   feature extraction → emotion fusion (prosody + a small pretrained SER
   model + a small pretrained text-emotion model) → independent
   deterministic noise/quality/overlap/silence analysis → schema assembly
   with a derived (not self-reported) confidence score. Every model is
   small enough to run CPU-only on the 2 vCPU / 3.8GB deployment target
   (measured: ~2.3GB peak RSS with every model loaded — see
   `docs/latency_analysis.md`) and nothing ever leaves the box.

## Baseline and approach comparison (per the trial's experimental-process guidance)

**Baseline**: a trivial majority-class predictor — `neutral`/`low` tone,
no noise, `severely_impaired` quality, `confidence=0.0`
(`schema.fallback_prediction`) — is the floor every real prediction is
implicitly compared against. It's also what the system falls back to for
a file it genuinely cannot process (see `docs/failure_modes_and_next_steps.md`),
so "no worse than the baseline" holds even in that failure case, not just
as a conceptual reference point.

**Two materially different approaches compared for `emotional_tone`**, as
requested (e.g. "audio foundation model vs. acoustic features + lightweight
classifier"):

1. **Audio-foundation-model-only**: the wav2vec2 SER model on the waveform
   alone, no transcript.
2. **Fused (chosen)**: SER + a text-emotion model on the ASR transcript +
   prosody-derived intensity, per `emotion.py`.

These are not just described — `inference/tests/test_emotion.py::test_audio_only_approach_misses_what_fusion_catches`
runs both against the same calm-voiced-but-lexically-negative input:
approach 1 reports `neutral` (SER hears a flat, unremarkable voice —
there's nothing in the waveform alone to detect), approach 2 correctly
reports a negative tone, because it can read the words. This is the
concrete evidence for fusing rather than shipping the foundation model
alone, beyond the design rationale already documented in `emotion.py`'s
docstring.

A third arm — acoustic features (prosody) *alone*, with no SER model, as
a lightweight-classifier baseline — was not built as a separate
standalone system in the time available; the closest equivalent is that
prosody already runs as an independent input inside the fusion (setting
intensity, per `_agitation_score`), and its known limitation on its own
(it has no valence axis — it can detect agitation, not positive-vs-
negative) is documented in `emotion.py`'s docstring and
`docs/failure_modes_and_next_steps.md`. Building it out as a fully
separate, independently-scorable third approach is the first thing to do
with more time — see `docs/failure_modes_and_next_steps.md`.

## Per-field method summary

| Field | Method | Module |
|---|---|---|
| `emotional_tone` / `emotional_intensity` | Rule-based fusion of prosody features, a wav2vec2-base SER model, and a distilroberta text-emotion model | `emotion.py`, `prosody_features.py` |
| `background_noise_present/type/severity` | Non-speech-region MFCC/spectral-centroid/bandwidth signature, cosine-matched against a curated ESC-50-derived reference bank | `noise.py` |
| `audio_quality` | Deterministic clipping/high-frequency-energy/dropout/echo checks on the full waveform | `quality.py` |
| `speaker_overlap_present` | Multi-band energy/harmonic-collision heuristic (pyannote's OSD was the preferred option but requires an HF-gated token unavailable here — see `docs/failure_modes_and_next_steps.md`) | `overlap.py` |
| `long_silence_present` | Pure VAD-gap arithmetic | `silence.py` |
| `confidence` | Weighted blend of every module's own confidence, scaled down further under measured poor audio quality | `aggregate.py` |

## The confound-decorrelation design

Two conflations the trial spec explicitly warns against, and how this
pipeline avoids each:

1. **Loudness ≠ frustration.** `prosody_features.py` computes energy as a
   *contour* (`energy_rms_std`, the RMS envelope's variance across the
   clip) rather than an absolute level. `emotion.py`'s fusion logic
   consumes only that variance term (via `_agitation_score`) — a call
   that's simply loud throughout, with a flat energy contour, doesn't
   register as agitated; a call with sudden bursts of energy variation
   does, independent of the call's overall volume.
2. **Background noise ≠ audio quality.** `noise.py` reads only non-speech-
   region audio (the gaps between VAD-detected speech segments) and never
   touches `quality.py`'s clipping/dropout/echo signals; `quality.py`
   reads the whole waveform and never touches noise's non-speech energy
   ratio. A quiet call recorded through a bad mic (severely_impaired
   quality, no noise) and a clear-quality call taken next to traffic
   (present noise, clear quality) score independently, as required.

## Why fusion/thresholds are rule-based, not fit

Every heuristic constant in `emotion.py`, `quality.py`, `noise.py`, and
`overlap.py` (agitation bounds, severity cutoffs, confidence-floor values)
is a domain-reasoning starting point, not a value fit on the 3 labeled
production calls. n=3 has no statistical power to fit parameters without
overfitting to those exact 3 recordings — see `docs/validation_results.md`
for what's actually been validated (directional synthetic-condition tests)
versus what hasn't (quantitative accuracy against labeled ground truth).
The 3 labeled calls, when available, are meant to sanity-check these
thresholds after the fact, not train them.
