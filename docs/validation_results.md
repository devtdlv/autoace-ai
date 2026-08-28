# Validation Results

Reports what has actually been validated and how, without conflating
directional/synthetic checks with quantitative accuracy metrics on real
labeled data — the trial spec's explicit requirement for validation
reporting.

## 1. What exists: directional synthetic-condition tests (real, run in CI-equivalent locally)

`inference/tests/` (30 tests, all passing — run via
`PYTHONPATH=. inference/.venv/bin/pytest inference/tests/ -v`) validates
that each Milestone 3-5 module moves in the *correct direction* under a
known, deliberately-constructed synthetic condition, generated via
espeak-ng (speech) + ffmpeg filters (degradation), not the 3 confidential
production calls:

| Module | Synthetic conditions tested | What's checked |
|---|---|---|
| `quality.py` | clean vs. clipped (ffmpeg `volume=25`), clean vs. muffled (`lowpass=f=500`), clean vs. echoey (`aecho`) | clipping ratio, high-frequency energy ratio, and echo score each move the correct direction; overall `AudioQuality` bucket for the degraded clip is never better than the clean clip |
| `noise.py` | clean vs. pink-noise-mixed | `present`/`severity` correctly flip on when non-speech-region energy rises; reference-bank load itself is checked (15 ESC-50-derived category signatures) |
| `overlap.py` | single voice vs. two simultaneous espeak voices (different voice IDs, mixed) | the fallback heuristic's raw candidate-frame ratio is higher for the two-voice clip than the single-voice clip |
| `silence.py` | synthetic gap-position and gap-length cases (unit-level, no audio needed) + a real ~12s injected silence vs. a normal-paced clean call | correctly flags/doesn't flag `long_silence_present` |
| `emotion.py` | fabricated prosody/SER-prob/text-prob inputs covering each branch of the fusion's mapping table (clear positive, clear neutral, high-agitation negative, calm-voiced-but-lexically-negative, model disagreement) | `classify_emotion`'s bucket/intensity/confidence logic is correct on inputs with known expected outcomes; a full real-model integration smoke test confirms the two transformer models load and produce schema-valid output |
| `aggregate.py` | fabricated component results at varying quality/confidence levels | `confidence` decreases under severely-impaired quality and low ASR confidence, as designed |

This is real evidence the *mechanism* of each module works as intended. It
is **not** a quantitative accuracy measurement — none of these synthetic
clips have an independently-labeled "correct" emotional-tone answer (TTS
speech has no genuine emotion to recover), and the degradation tests check
relative direction, not absolute precision/recall.

## 2. What does NOT exist yet: quantitative accuracy metrics

Honestly scoped, not glossed over:

- **No accuracy/F1/confusion-matrix numbers exist for `emotional_tone` or
  `emotional_intensity`.** Producing those requires either (a) the 3
  labeled production calls — `data/labeled_samples/` is present but empty
  (gitignored, confidential; this repo/environment has never had the
  actual audio) — or (b) integrating a public labeled speech-emotion
  corpus (e.g. RAVDESS, CREMA-D) as a mapped validation set, which has not
  been done in this pass. n=3 would have no statistical power regardless
  (a single-call leave-one-out check, directional only) — this was true
  before this milestone and remains true now.
- **No formal precision/recall numbers exist for noise-type classification**
  beyond the presence/absence directional test above — the 15-category
  ESC-50-derived reference bank has not been validated against a held-out
  ESC-50 test split.
- **Speaker overlap's fallback heuristic has no measured false-positive
  rate on real multi-speaker call audio** — only the directional
  single-vs-two-voice synthetic comparison above.

## Next steps if more validation data becomes available

1. If the 3 labeled production calls are provided to this environment: run
   them through the full pipeline, report per-field agreement plainly
   labeled "n=3, directional only, no statistical power" — exactly as this
   doc's placeholder already committed to, never presented as a real
   accuracy metric.
2. Integrate a public speech-emotion corpus (RAVDESS/CREMA-D) as a mapped
   validation set for `emotional_tone`/`emotional_intensity` — real actors
   expressing real target emotions, unlike the flat espeak-ng TTS used in
   the unit tests above.
3. Hold out an ESC-50 test split (disjoint from the reference-bank build
   clips) to measure noise-type top-1 accuracy.

See `docs/failure_modes_and_next_steps.md` for the full list of known
limitations this validation gap and the rest of the pipeline currently has.
