"""Background noise: presence, type, severity.

Milestone 4. Deliberately consumes ONLY non-speech regions (per vad.py) and
the noise floor beneath speech — never audio_quality's distortion/clipping
signals — so noise detection cannot be a proxy for poor technical quality
(the trial spec explicitly separates these two).

Noise type is matched against a small canonical vocabulary (office chatter,
music, road noise, television, keyboard typing, wind, mechanical noise, ...)
via spectral-signature similarity to a reference bank built from public
environmental-sound data (ESC-50/UrbanSound8K-style categories), not from
the 3 labeled production calls.
"""

from dataclasses import dataclass

from inference.pipeline.schema import BackgroundNoiseSeverity


@dataclass
class NoiseResult:
    present: bool
    noise_type: str  # "" when not present
    severity: BackgroundNoiseSeverity
    confidence: float


def analyze_noise(wav_path: str, speech_segments) -> NoiseResult:
    raise NotImplementedError("Milestone 4: noise floor + spectral-signature matching")
