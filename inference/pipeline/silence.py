"""Long silence / dead-air detection — purely deterministic from VAD gaps.

Milestone 4 (but the simplest, highest-confidence module in the pipeline).
"""

from dataclasses import dataclass

LONG_SILENCE_THRESHOLD_S = 8.0  # placeholder; calibrated in Milestone 4


@dataclass
class SilenceResult:
    present: bool
    longest_gap_s: float


def detect_long_silence(speech_segments, total_duration_s: float) -> SilenceResult:
    raise NotImplementedError("Milestone 4: VAD gap analysis")
