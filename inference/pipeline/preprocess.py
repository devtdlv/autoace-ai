"""Audio normalization via ffmpeg. Real implementation (not a stub) — every
downstream module (VAD, ASR, feature extraction) depends on a consistent
sample rate / channel layout, so this runs first and unconditionally.
"""

import subprocess
from pathlib import Path

TARGET_SAMPLE_RATE = 16000  # standard rate for VAD/ASR/SER models used downstream


class PreprocessError(RuntimeError):
    """Raised when ffmpeg fails to decode/normalize an input file."""


def normalize_audio(input_path: str | Path, output_path: str | Path) -> Path:
    """Decode any supported input format and re-encode to mono 16kHz PCM16 WAV.

    Deliberately narrow: this is the one place format-specific quirks
    (VBR mp3, odd sample rates, stereo call recordings, container issues)
    get resolved, so every later pipeline stage can assume a uniform format.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-sample_fmt", "s16",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PreprocessError(
            f"ffmpeg failed on {input_path.name}: {result.stderr.strip()[-2000:]}"
        )
    return output_path
