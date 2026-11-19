"""Pedalboard knob chain for vox_cloner.

Defines a vocal processing chain as a flat set of named knobs, and converts
between a parameter dict / optimizer vector and a renderable pedalboard.Pedalboard.

The feature-matching optimizer treats these knobs as the search space:
    render(base_audio, sr, params) -> processed audio -> features -> distance to target.

Run directly for a self-check:
    python knobs.py
"""
from __future__ import annotations

import numpy as np
from pedalboard import (
    Pedalboard, HighpassFilter, LowpassFilter, LowShelfFilter, PeakFilter,
    HighShelfFilter, Compressor, Distortion, Chorus, Delay, Reverb, Gain,
)

# Each knob: name -> (min, max, default). Order here == optimizer vector order.
# ponytail: generous but curated set (~30 knobs). If the optimizer struggles with
# the dimensionality, freeze a subset via FROZEN rather than growing this list.
KNOBS: dict[str, tuple[float, float, float]] = {
    # --- cleanup ---
    "highpass_hz":        (20.0,   300.0,  80.0),
    "lowpass_hz":         (4000.0, 20000.0, 20000.0),
    # --- EQ: low shelf ---
    "lowshelf_hz":        (40.0,   300.0,  120.0),
    "lowshelf_db":        (-12.0,  12.0,   0.0),
    "lowshelf_q":         (0.3,    2.0,    0.707),
    # --- EQ: peak band 1 (low-mid) ---
    "peak1_hz":           (150.0,  800.0,  350.0),
    "peak1_db":           (-12.0,  12.0,   0.0),
    "peak1_q":            (0.3,    5.0,    1.0),
    # --- EQ: peak band 2 (mid) ---
    "peak2_hz":           (800.0,  4000.0, 1800.0),
    "peak2_db":           (-12.0,  12.0,   0.0),
    "peak2_q":            (0.3,    5.0,    1.0),
    # --- EQ: peak band 3 (presence) ---
    "peak3_hz":           (4000.0, 10000.0, 6000.0),
    "peak3_db":           (-12.0,  12.0,   0.0),
    "peak3_q":            (0.3,    5.0,    1.0),
    # --- EQ: high shelf (air) ---
    "highshelf_hz":       (6000.0, 16000.0, 10000.0),
    "highshelf_db":       (-12.0,  12.0,   0.0),
    "highshelf_q":        (0.3,    2.0,    0.707),
    # --- compressor ---
    "comp_threshold_db":  (-40.0,  0.0,    -18.0),
    "comp_ratio":         (1.0,    10.0,   2.0),
    "comp_attack_ms":     (0.5,    50.0,   10.0),
    "comp_release_ms":    (10.0,   500.0,  100.0),
    # --- saturation ---
    "drive_db":           (0.0,    20.0,   0.0),
    # --- chorus ---  # ponytail: mix/depth capped so optimizer can't warble-smear the vocal to game the metric
    "chorus_rate_hz":     (0.1,    5.0,    1.0),
    "chorus_depth":       (0.0,    0.3,    0.0),
    "chorus_mix":         (0.0,    0.2,    0.0),
    # --- delay ---  # ponytail: feedback/mix capped; high-feedback short delay = metallic comb ringing
    "delay_seconds":      (0.0,    0.8,    0.0),
    "delay_feedback":     (0.0,    0.35,   0.0),
    "delay_mix":          (0.0,    0.25,   0.0),
    # --- reverb ---  # ponytail: wet capped so the vocal isn't drowned
    "reverb_room_size":   (0.0,    1.0,    0.2),
    "reverb_damping":     (0.0,    1.0,    0.5),
    "reverb_wet":         (0.0,    0.35,   0.0),
    # --- output ---
    "output_gain_db":     (-12.0,  12.0,   0.0),
}

KNOB_NAMES: list[str] = list(KNOBS.keys())


def defaults() -> dict[str, float]:
    """Neutral-ish starting params (effects mostly transparent)."""
    return {k: v[2] for k, v in KNOBS.items()}


def bounds() -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) arrays in KNOB_NAMES order, for the optimizer."""
    lo = np.array([KNOBS[k][0] for k in KNOB_NAMES], dtype=float)
    hi = np.array([KNOBS[k][1] for k in KNOB_NAMES], dtype=float)
    return lo, hi


def vector_to_params(x: np.ndarray) -> dict[str, float]:
    """Optimizer vector -> param dict, clipped to each knob's range."""
    lo, hi = bounds()
    x = np.clip(np.asarray(x, dtype=float), lo, hi)
    return {name: float(v) for name, v in zip(KNOB_NAMES, x)}


def params_to_vector(params: dict[str, float]) -> np.ndarray:
    return np.array([params[k] for k in KNOB_NAMES], dtype=float)


def params_to_unit(params: dict[str, float]) -> np.ndarray:
    """Param dict -> vector scaled to [0,1] per knob range (model target space)."""
    lo, hi = bounds()
    return (params_to_vector(params) - lo) / (hi - lo)


def unit_to_params(u: np.ndarray) -> dict[str, float]:
    """[0,1] vector -> param dict (inverse of params_to_unit), clipped."""
    lo, hi = bounds()
    return vector_to_params(lo + np.clip(np.asarray(u, dtype=float), 0.0, 1.0) * (hi - lo))


def random_params(rng: np.random.Generator | None = None) -> dict[str, float]:
    """Uniform-random knob settings within bounds (for dataset generation).

    ponytail: uniform over the full range is fine for v0. Many draws will be
    heavy-handed (max reverb + delay + drive at once); if the model learns garbage
    from unrealistic combos, bias the wet/drive knobs toward 0 here, not elsewhere.
    """
    rng = rng or np.random.default_rng()
    lo, hi = bounds()
    return vector_to_params(lo + rng.random(len(KNOB_NAMES)) * (hi - lo))


def build_board(params: dict[str, float]) -> Pedalboard:
    """Construct the pedalboard chain from a full param dict."""
    p = params
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=p["highpass_hz"]),
        LowpassFilter(cutoff_frequency_hz=p["lowpass_hz"]),
        LowShelfFilter(cutoff_frequency_hz=p["lowshelf_hz"], gain_db=p["lowshelf_db"], q=p["lowshelf_q"]),
        PeakFilter(cutoff_frequency_hz=p["peak1_hz"], gain_db=p["peak1_db"], q=p["peak1_q"]),
        PeakFilter(cutoff_frequency_hz=p["peak2_hz"], gain_db=p["peak2_db"], q=p["peak2_q"]),
        PeakFilter(cutoff_frequency_hz=p["peak3_hz"], gain_db=p["peak3_db"], q=p["peak3_q"]),
        HighShelfFilter(cutoff_frequency_hz=p["highshelf_hz"], gain_db=p["highshelf_db"], q=p["highshelf_q"]),
        Compressor(threshold_db=p["comp_threshold_db"], ratio=p["comp_ratio"],
                   attack_ms=p["comp_attack_ms"], release_ms=p["comp_release_ms"]),
        Distortion(drive_db=p["drive_db"]),
        Chorus(rate_hz=p["chorus_rate_hz"], depth=p["chorus_depth"], mix=p["chorus_mix"]),
        Delay(delay_seconds=p["delay_seconds"], feedback=p["delay_feedback"], mix=p["delay_mix"]),
        Reverb(room_size=p["reverb_room_size"], damping=p["reverb_damping"],
               wet_level=p["reverb_wet"], dry_level=1.0 - p["reverb_wet"]),
        Gain(gain_db=p["output_gain_db"]),
    ])


def render(audio: np.ndarray, sr: int, params: dict[str, float] | np.ndarray) -> np.ndarray:
    """Apply the chain to mono audio. Accepts a param dict or an optimizer vector.

    audio: float32 mono, shape (num_samples,). Returns same shape/dtype.
    """
    if not isinstance(params, dict):
        params = vector_to_params(params)
    audio = np.asarray(audio, dtype=np.float32)
    board = build_board(params)
    out = board(audio, sr)
    return np.asarray(out, dtype=np.float32)


def _demo() -> None:
    sr = 44100
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(sr) * 0.1).astype(np.float32)  # 1s of quiet noise

    assert len(KNOB_NAMES) == len(KNOBS)
    lo, hi = bounds()
    assert np.all(lo < hi), "every knob needs min < max"

    # round-trip vector <-> params
    v = params_to_vector(defaults())
    assert np.allclose(v, params_to_vector(vector_to_params(v)))

    # out-of-range vector is clipped, not crashed
    wild = vector_to_params(hi * 10)
    assert all(lo[i] <= wild[n] <= hi[i] for i, n in enumerate(KNOB_NAMES))

    # render runs, keeps shape, and actually changes the audio with a non-default setting
    y_def = render(x, sr, defaults())
    assert y_def.shape == x.shape and np.all(np.isfinite(y_def))
    loud = defaults() | {"drive_db": 15.0, "output_gain_db": 6.0, "reverb_wet": 0.5}
    y_fx = render(x, sr, loud)
    assert not np.allclose(y_fx, x), "effects should change the signal"

    print(f"OK: {len(KNOB_NAMES)} knobs, render {x.shape} -> {y_fx.shape}, all checks passed")


if __name__ == "__main__":
    _demo()
