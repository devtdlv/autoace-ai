"""Long silence / dead-air detection — purely deterministic from VAD gaps.

Milestone 4 (the simplest, highest-confidence module in the pipeline: no
model, no threshold tuning beyond the one constant below).
"""

from dataclasses import dataclass

LONG_SILENCE_THRESHOLD_S = 8.0  # placeholder; calibrated in Milestone 4


@dataclass
class SilenceResult:
    present: bool
    longest_gap_s: float


def detect_long_silence(speech_segments, total_duration_s: float) -> SilenceResult:
    if not speech_segments:
        return SilenceResult(present=total_duration_s >= LONG_SILENCE_THRESHOLD_S, longest_gap_s=total_duration_s)

    ordered = sorted(speech_segments, key=lambda s: s.start_s)
    gaps = [ordered[0].start_s]  # silence before the first speech segment
    gaps.extend(
        ordered[i + 1].start_s - ordered[i].end_s for i in range(len(ordered) - 1)
    )
    gaps.append(total_duration_s - ordered[-1].end_s)  # silence after the last

    longest_gap_s = max(0.0, max(gaps))
    return SilenceResult(
        present=longest_gap_s >= LONG_SILENCE_THRESHOLD_S,
        longest_gap_s=longest_gap_s,
    )
