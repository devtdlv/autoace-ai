# Failure Modes, Limitations, and Next Steps

## Emotion (`emotion.py`, `prosody_features.py`)

- **Fusion thresholds are heuristic starting points, not fit values.**
  `NEGATIVE_SEVERITY_FRUSTRATED_MAX`/`UPSET_MAX`, the prosody agitation
  bounds, and the text/SER fusion weights were chosen from domain
  reasoning, not calibrated against labeled calls (n=3 is not enough to
  fit parameters without overfitting — see `docs/validation_results.md`).
  If/when more labeled data exists, these are the first constants to
  revisit.
- **No non-English support.** The text-emotion model
  (`j-hartmann/emotion-english-distilroberta-base`) and the SER model are
  both English-trained; a non-English call will get an unreliable text
  signal and a SER signal of unknown reliability. `asr.Transcript.language`
  is already surfaced by faster-whisper and could gate this (skip/flag
  text-emotion when `language != "en"`) — not implemented.
- **Purely-prosodic cases with no lexical signal** (e.g. a caller who's
  curt/short and hangs up, or speaks in fragments ASR transcribes poorly)
  lean entirely on the SER model and prosody agitation, which is the
  weakest of the three signals on its own (see `tone_confidence`'s design
  — this is exactly the case it's meant to flag as lower-confidence).
- **wav2vec2-base-superb-er's 4-class label set** (neutral/happy/angry/sad)
  is coarser than the 5-class trial target — mapping `sad` into the
  frustrated/upset/distressed ladder purely via prosody agitation is a
  simplification.
- **Only two of the spec's suggested "materially different approaches"
  were actually compared** (audio-foundation-model-only vs. the fused
  system — see `docs/technical_memo.md`). A third arm — acoustic
  (prosody) features alone driving a lightweight rule/classifier, with no
  learned SER or text model at all — was not built and evaluated as a
  fully separate, independently-scorable approach given the time
  available. This is the highest-value next experiment: it would show
  directly how much the two learned models are actually contributing
  versus a near-zero-cost deterministic approach.

## Noise (`noise.py`)

- **Reference bank is 15 curated ESC-50 categories, not a general acoustic
  scene classifier.** A noise type outside that list (a category ESC-50
  doesn't cover, or a genuinely novel sound) will either weakly match a
  wrong category or fall below `MATCH_CONFIDENCE_FLOOR` and report
  presence/severity with no type label — by design (see module docstring),
  but still a real coverage gap.
- **5 reference clips per category** (not the full ~40 ESC-50 clips per
  category) — a deliberate speed/setup-time tradeoff (see
  `scripts/build_noise_reference_bank.py`), traded against signature
  robustness. Worth revisiting if noise-type accuracy turns out to matter
  more than setup speed.

## Audio quality (`quality.py`)

- **Threshold constants are heuristic**, same caveat as emotion's fusion
  thresholds — not fit on labeled degraded/clean pairs, chosen from signal-
  processing domain reasoning and validated only directionally (see
  `docs/validation_results.md`).
- **Dropout detection is a coarse energy-collapse heuristic**, not a real
  codec-artifact/packet-loss detector — it will miss packet loss that
  doesn't manifest as a full energy dropout (e.g. a corrupted-but-not-
  silent frame).

## Speaker overlap (`overlap.py`)

- **This is the fallback path, not the originally preferred approach.**
  pyannote.audio's purpose-built overlapped-speech-detection pipeline
  requires a Hugging Face account, gated-model license acceptance, and an
  auth token — none available in this environment. The multi-band energy/
  harmonic-collision heuristic implemented instead is real and tested
  (directionally — see `docs/validation_results.md`) but weaker in
  practice than a trained OSD model: expect more false positives on
  audio with heavy background noise or reverb (broadband energy in a
  frame can look "unexplained by one voice's harmonics" without a second
  speaker actually being present), and more false negatives on overlap
  between two similar-pitched voices (harmonics collide, but a single
  merged harmonic series can still look plausible).
- **Next step if this matters more than the setup cost**: provision an HF
  token with the appropriate license acceptance and swap in pyannote's OSD
  pipeline — `overlap.py`'s `detect_overlap` signature doesn't need to
  change, only its implementation.
- **The candidate-frame-ratio threshold was originally miscalibrated below
  the single-speaker noise floor** — measured on real single-voice test
  clips (0.12-0.13) vs. a genuine two-voice mix (0.23), the original 0.08
  threshold flagged every single-speaker clip tested as overlapping.
  Raised to 0.18 (still a small-sample calibration, not fit on labeled
  data — the same caveat as every other heuristic constant here).

## Long silence (`silence.py`)

- Lowest-risk module in the pipeline — purely deterministic VAD-gap
  arithmetic. Its accuracy is entirely bounded by silero-vad's own
  accuracy, not anything specific to this module.
- `LONG_SILENCE_THRESHOLD_S = 8.0` is still a placeholder constant (per the
  original Milestone 2 docstring) — worth calibrating against what
  actually counts as "unacceptable dead air" for this specific use case
  once real call data is available.

## Batch processing (Milestone 6)

- Calls within a batch are processed **sequentially**, not in parallel —
  reasonable at the 2 vCPU deployment scale (see `docs/cost_analysis.md`:
  parallelizing doesn't reduce $/audio-minute, only wall-clock batch
  completion time), but means a large batch takes proportionally long to
  finish. If batch turnaround time becomes a complaint, running multiple
  worker processes (more vCPUs) is the fix, not code changes.

## Validation gap (see `docs/validation_results.md` for full detail)

No quantitative accuracy/F1 numbers exist yet for any field — only
directional synthetic-condition tests. Closing this needs either the 3
labeled production calls (still n=3, still directional-only even then) or
a public labeled speech-emotion corpus (RAVDESS/CREMA-D) integrated as a
proper mapped validation set. Neither has been done in this pass.
