# Validation Results

Reports what has actually been validated and how, without conflating
directional/synthetic checks with quantitative accuracy metrics on real
labeled data — the trial spec's explicit requirement for validation
reporting.

## 1. What exists: directional synthetic-condition tests (real, run in CI-equivalent locally)

`inference/tests/` (50 tests, all passing — run via
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
| `emotion.py` | fabricated prosody/SER-prob/text-prob inputs covering each branch of the fusion's mapping table (clear positive, clear neutral, high-agitation negative, calm-voiced-but-lexically-negative, model disagreement, audio-only vs. fused approach comparison) | `classify_emotion`'s bucket/intensity/confidence logic is correct on inputs with known expected outcomes; a full real-model integration smoke test confirms the two transformer models load and produce schema-valid output |
| `aggregate.py` | fabricated component results at varying quality/confidence levels, plus `fallback_prediction()`'s own schema validity | `confidence` decreases under severely-impaired quality and low ASR confidence, as designed; the failure-path guess is always schema-valid |

### Hidden-set robustness stress test (manual, not part of the pytest suite)

Run directly against the pipeline: a normal clip re-encoded as mp3, m4a,
stereo/44.1kHz, and telephone-band 8kHz; a 0.05s clip; 3s of pure silence;
5s of pure noise (no speech); and a genuinely corrupt (non-audio) file.
Every real-audio variant processed without error; only the corrupt file
failed, and it produced the schema-valid `fallback_prediction()` rather
than an empty/missing result. This is what motivated broadening
`AUDIO_EXTENSIONS` (a previously-unrecognized real format was being
silently dropped, not even reported as a failure) and adding the
fallback-prediction safety net — see
`docs/failure_modes_and_next_steps.md` and the commit that introduced
both.

This is real evidence the *mechanism* of each module works as intended. It
is **not** a quantitative accuracy measurement — none of these synthetic
clips have an independently-labeled "correct" emotional-tone answer (TTS
speech has no genuine emotion to recover), and the degradation tests check
relative direction, not absolute precision/recall.

## 2. The 3 labeled production calls (n=3 — directional only, no statistical power)

The 3 provided calls were run through the deployed pipeline. This is a
real result, not a synthetic one — and is reported exactly as it came
out, including where the system is wrong, per Section 11's requirement
not to report accuracy that flatters the system:

| Call | Field | Predicted | Expected | Match? |
|---|---|---|---|---|
| call_001 | emotional_tone | neutral | upset | ✗ |
| call_001 | background_noise_present | true | false | ✗ |
| call_001 | audio_quality | clear | clear | ✓ |
| call_001 | speaker_overlap_present | false | false | ✓ |
| call_002 | emotional_tone | upset | neutral | ✗ |
| call_002 | background_noise_present | true | true | ✓ (type/severity wrong: "car horn / traffic"/high vs. "TV"/medium) |
| call_002 | audio_quality | clear | clear | ✓ |
| call_002 | speaker_overlap_present | false | true | ✗ |
| call_003 | emotional_tone | neutral | satisfied | ✗ |
| call_003 | background_noise_present | true | true | ✓ (type wrong: "car horn / traffic" vs. "sharp static"; severity correct: medium) |
| call_003 | audio_quality | clear | clear | ✓ |
| call_003 | speaker_overlap_present | false | true | ✗ |

**Summary, n=3**: `emotional_tone` 0/3, `background_noise_present` 2/3
(noise *type* 0/3 exact match), `audio_quality` 3/3, `speaker_overlap_present`
1/3, `long_silence_present` 2/3.

**`audio_quality` going 0/3 → 3/3** is because of a real bug this exact
check surfaced: the quality thresholds were calibrated purely on synthetic
TTS audio and treated real telephone-band speech's normal characteristics
(near-zero energy above 3.4kHz, natural-speech-rhythm autocorrelation) as
severe impairment. Fixed — see `docs/failure_modes_and_next_steps.md` and
the commit recalibrating `quality.py`.

**`emotional_tone` at 0/3 was *not* patched to force a better number.**
Forcing these 3 specific answers to come out right by hand-tuning
thresholds would be exactly the overfitting Section 9 warns against —
tuning parameters to the exact evaluation samples produces a number that
looks good here and generalizes worse to the hidden set. Unlike the
quality fix (a specific, generalizable, verifiable fact about telephone
codecs), there's no equivalent "this is objectively wrong" finding for
tone on 3 samples — it may reflect a genuine model limitation (real
call-center speech emotion is a harder problem than the synthetic TTS
this was tuned against) or normal small-sample noise. n=3 cannot
distinguish those, and no attempt is made here to pretend it can. This
result is visible on the live dashboard (login → the "labels.csv" batch)
with per-field expected-vs-predicted coloring.

## 3. What does NOT exist yet: quantitative accuracy metrics beyond n=3

Honestly scoped, not glossed over:

- **No accuracy/F1/confusion-matrix numbers beyond the n=3 table above
  exist for `emotional_tone` or `emotional_intensity`.** Producing a
  statistically meaningful version requires either (a) more labeled
  production calls than the 3 provided, or (b) integrating a public
  labeled speech-emotion corpus (e.g. RAVDESS, CREMA-D) as a mapped
  validation set, which has not
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

1. ~~Run the 3 labeled production calls through the pipeline~~ — done, see
   Section 2 above.
2. Integrate a public speech-emotion corpus (RAVDESS/CREMA-D) as a mapped
   validation set for `emotional_tone`/`emotional_intensity` — real actors
   expressing real target emotions, unlike the flat espeak-ng TTS used in
   the unit tests above.
3. Hold out an ESC-50 test split (disjoint from the reference-bank build
   clips) to measure noise-type top-1 accuracy.

See `docs/failure_modes_and_next_steps.md` for the full list of known
limitations this validation gap and the rest of the pipeline currently has.
