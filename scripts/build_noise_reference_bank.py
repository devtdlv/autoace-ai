"""Builds inference/models/noise_reference_bank.npz — one spectral
signature per background-noise category, used by noise.py's cosine-
similarity matcher.

Run once at setup time (documented in the top-level README), not at
request time. Downloads a small curated subset of ESC-50 (a public,
CC-BY-licensed environmental-sound dataset) — a handful of clips per
category relevant to call-center/home/office environments, not the full
2000-clip dataset (not needed for a mean-signature reference bank, and
slower to fetch/process for no accuracy benefit at this scale).

Usage:
    inference/.venv/bin/python scripts/build_noise_reference_bank.py
"""

import csv
import io
import sys
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf

ESC50_META_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC50_AUDIO_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{filename}"

# ESC-50 category -> AutoAce noise_type label. Curated to environments a
# support call is plausibly taken from/near; excludes ESC-50's many animal
# and nature categories with no call-center relevance.
CATEGORY_MAP = {
    "keyboard_typing": "keyboard typing",
    "mouse_click": "mouse click",
    "door_wood_knock": "door knock",
    "car_horn": "car horn / traffic",
    "engine": "engine / vehicle noise",
    "train": "train / transit noise",
    "siren": "siren",
    "vacuum_cleaner": "vacuum cleaner",
    "washing_machine": "washing machine / appliance",
    "wind": "wind",
    "rain": "rain",
    "crying_baby": "crying baby",
    "clock_tick": "clock ticking",
    "footsteps": "footsteps",
    "laughing": "background talking / laughter",
}

CLIPS_PER_CATEGORY = 5
TARGET_SAMPLE_RATE = 16000
N_MFCC = 13

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "inference" / "models" / "noise_reference_bank.npz"


def _feature_vector(audio: np.ndarray, sr: int) -> np.ndarray:
    """Same feature family noise.py computes over a call's non-speech
    regions: MFCC means + spectral centroid + spectral bandwidth. Must stay
    in sync with noise.py's `_feature_vector`.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    return np.concatenate([
        mfcc.mean(axis=1),
        [centroid.mean(), bandwidth.mean()],
    ]).astype(np.float32)


def main() -> None:
    print(f"Fetching ESC-50 metadata from {ESC50_META_URL} ...")
    resp = requests.get(ESC50_META_URL, timeout=30)
    resp.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(resp.text)))

    category_names: list[str] = []
    vectors: list[np.ndarray] = []

    for esc_category, noise_type in CATEGORY_MAP.items():
        filenames = [r["filename"] for r in rows if r["category"] == esc_category][:CLIPS_PER_CATEGORY]
        if not filenames:
            print(f"  WARNING: no ESC-50 clips found for category '{esc_category}', skipping", file=sys.stderr)
            continue

        clip_vectors = []
        for filename in filenames:
            url = ESC50_AUDIO_URL.format(filename=filename)
            audio_resp = requests.get(url, timeout=30)
            audio_resp.raise_for_status()
            audio, sr = sf.read(io.BytesIO(audio_resp.content), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            clip_vectors.append(_feature_vector(audio, TARGET_SAMPLE_RATE))

        category_names.append(noise_type)
        vectors.append(np.mean(clip_vectors, axis=0))
        print(f"  {esc_category} -> '{noise_type}' ({len(filenames)} clips)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        category_names=np.array(category_names),
        vectors=np.stack(vectors),
        sample_rate=TARGET_SAMPLE_RATE,
        n_mfcc=N_MFCC,
    )
    print(f"Wrote {len(category_names)} category signatures to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
