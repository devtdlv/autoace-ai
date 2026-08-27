"""Speaker overlap detection — two+ speakers talking simultaneously enough
to affect understanding/analysis.

Milestone 4. Primary candidate: pyannote.audio's overlapped-speech-detection
pipeline (purpose-built, CPU-viable for batch, more reliable than a
hand-rolled energy heuristic on mono call audio). Falls back to a simpler
multi-band energy/harmonic-collision heuristic if pyannote's model gating
(HF auth token + license acceptance) proves impractical to provision — this
tradeoff will be decided and documented in Milestone 4, not silently.
"""

from dataclasses import dataclass


@dataclass
class OverlapResult:
    present: bool
    confidence: float


def detect_overlap(wav_path: str, speech_segments) -> OverlapResult:
    raise NotImplementedError("Milestone 4: pyannote OSD or fallback heuristic")
