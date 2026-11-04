"""Audio feature extraction for vox_cloner.

Turns a mono vocal into a ~60-dim vector capturing *tonal/dynamic character*
(not melody), so two different singers/phrases can be compared. This vector is
the distance metric the knob optimizer minimizes against the target.

ponytail: all features come from librosa + pyloudnorm. No hand-rolled DSP.

Run directly for a self-check:
    python audio_feature_extraction.py
"""
from __future__ import annotations

import numpy as np
import librosa
import pyloudnorm as pyln

N_MFCC = 13
SIBILANCE_BAND = (5000.0, 10000.0)  # Hz, "ess" region for vocals


def load(path: str, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load a file as mono float32 at sr."""
    y, _sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr


def feature_names() -> list[str]:
    names: list[str] = []
    names += [f"mfcc{i}_mean" for i in range(N_MFCC)]
    names += [f"mfcc{i}_std" for i in range(N_MFCC)]
    names += [f"mfcc_delta{i}_mean" for i in range(N_MFCC)]
    for stat in ("mean", "std"):
        names += [f"centroid_{stat}", f"rolloff_{stat}", f"bandwidth_{stat}",
                  f"flatness_{stat}", f"zcr_{stat}", f"rms_{stat}"]
    names += [f"contrast{i}_mean" for i in range(7)]  # 6 bands + 1
    names += ["crest_factor", "lufs", "sibilance_ratio"]
    return names


def extract_features(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """Mono audio -> fixed-length feature vector (see feature_names())."""
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    # guard silence / near-empty
    if y.size < 2048 or not np.any(np.abs(y) > 1e-6):
        return np.zeros(len(feature_names()), dtype=np.float64)

    feats: list[float] = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feats += list(mfcc.mean(axis=1))
    feats += list(mfcc.std(axis=1))
    feats += list(librosa.feature.delta(mfcc).mean(axis=1))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    for arr in (centroid, rolloff, bandwidth, flatness, zcr, rms):
        feats.append(float(arr.mean()))
    for arr in (centroid, rolloff, bandwidth, flatness, zcr, rms):
        feats.append(float(arr.std()))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # (7, frames)
    feats += list(contrast.mean(axis=1))

    # crest factor: peak / rms (dynamics). High = punchy/uncompressed.
    peak = float(np.max(np.abs(y)))
    rms_global = float(np.sqrt(np.mean(y ** 2))) + 1e-9
    feats.append(peak / rms_global)

    # integrated loudness
    try:
        lufs = pyln.Meter(sr).integrated_loudness(y.astype(np.float64))
        if not np.isfinite(lufs):
            lufs = -70.0
    except Exception:
        lufs = -70.0
    feats.append(float(lufs))

    # sibilance ratio: energy in the ess band / total energy (via STFT)
    S = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    band = (freqs >= SIBILANCE_BAND[0]) & (freqs <= SIBILANCE_BAND[1])
    total = float(S.sum()) + 1e-9
    feats.append(float(S[band].sum()) / total)

    out = np.array(feats, dtype=np.float64)
    assert out.shape[0] == len(feature_names()), (out.shape[0], len(feature_names()))
    return out


# --- z-score normalization across a sample set -------------------------------
# The optimizer's distance must not be dominated by big-magnitude dims (Hz, LUFS).
# Fit stats once over your corpus, then normalize every vector before comparing.

def fit_norm(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """matrix: (n_samples, n_features) -> {'mean','std'} for z-scoring."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-9] = 1.0
    return {"mean": mean, "std": std}


def apply_norm(vec: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (vec - stats["mean"]) / stats["std"]


def distance(a: np.ndarray, b: np.ndarray, stats: dict[str, np.ndarray] | None = None) -> float:
    """Euclidean distance between two feature vectors (z-scored if stats given)."""
    if stats is not None:
        a, b = apply_norm(a, stats), apply_norm(b, stats)
    return float(np.linalg.norm(a - b))


def _demo() -> None:
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    bright = (0.3 * np.sin(2 * np.pi * 220 * t)
              + 0.2 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32)

    f_tone = extract_features(tone, sr)
    f_bright = extract_features(bright, sr)
    assert f_tone.shape == (len(feature_names()),)
    assert np.all(np.isfinite(f_tone))

    # a brighter signal must have a higher sibilance ratio -> features differ
    sib_idx = feature_names().index("sibilance_ratio")
    assert f_bright[sib_idx] > f_tone[sib_idx]

    # z-score + distance sanity: a signal is zero distance from itself
    stats = fit_norm(np.stack([f_tone, f_bright]))
    assert distance(f_tone, f_tone, stats) < 1e-9
    assert distance(f_tone, f_bright, stats) > 0.0

    print(f"OK: {len(feature_names())} features, finite, brightness + distance checks passed")


if __name__ == "__main__":
    _demo()
