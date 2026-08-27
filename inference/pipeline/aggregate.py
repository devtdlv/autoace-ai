"""Assembles per-module results into the final Prediction schema and derives
the overall `confidence` field.

Milestone 5. `confidence` is NOT self-reported by any single model — it is
derived from: agreement across the emotion sub-signals (prosody vs. SER
model vs. text-emotion model), ASR confidence, and measured audio quality
(low quality -> lower confidence across the board, since every downstream
signal is noisier). Calibration approach documented in
docs/validation_results.md.
"""

from inference.pipeline.schema import Prediction


def aggregate(emotion_result, noise_result, quality_result, overlap_result, silence_result, asr_confidence: float) -> Prediction:
    raise NotImplementedError("Milestone 5: schema assembly + confidence calibration")
