"""Per-pair knob optimizer -- the v0 that actually reaches the goal.

Given MY dry vocal (base) and SOMEONE ELSE'S isolated vocal (target), search knob
settings so that features(render(base, knobs)) is as close as possible to
features(target). No training, no model: it dials the knobs by guided trial-and-error
for THIS pair, using the feature distance as the score.

    poetry run python scripts/optimize_match.py <base.wav> <target.wav> [--out match.wav]
    poetry run python scripts/optimize_match.py --demo        # self-check

Speed: the search renders+scores the base many times, so it analyzes only the first
--seconds of audio (features are statistical, a few seconds characterize the tone),
then applies the winning knobs to the FULL base vocal for the saved output.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from knobs import render, unit_to_params, KNOB_NAMES  # noqa: E402
from audio_feature_extraction import (  # noqa: E402
    load, extract_features, fit_norm, apply_norm,
)

SR = 44100


def get_stats(base: np.ndarray, sr: int = SR, n: int = 24, seed: int = 0) -> dict:
    """Per-dimension mean/std to z-score features so no dim dominates the distance.

    Self-contained: builds the scaling from the base vocal and a pool of random
    renders of it -- i.e. the spread of what THIS base can actually produce. No
    dataset or cache needed.
    """
    from knobs import random_params
    rng = np.random.default_rng(seed)
    rows = [extract_features(base, sr)]
    for _ in range(n):
        rows.append(extract_features(render(base, sr, random_params(rng)), sr))
    return fit_norm(np.asarray(rows, dtype=float))


# Per-dimension z-difference is clipped to this many std devs before summing, so a
# few features the base simply cannot reach don't dominate the score and starve the
# many reachable ones. Keeps the optimizer working on what it can actually change.
CLIP_Z = 4.0


def _clipped_dist(feats_n: np.ndarray, tgt_n: np.ndarray) -> float:
    return float(np.linalg.norm(np.clip(feats_n - tgt_n, -CLIP_Z, CLIP_Z)))


def optimize(base: np.ndarray, sr: int, target_feats: np.ndarray, stats: dict,
             maxfevals: int = 1250, seed: int = 0) -> tuple[dict, float, float, float]:
    """Search knob settings minimizing clipped z-scored feature distance to target.

    Returns (best_params, dist_raw, dist_best, dist_neutral).
    """
    import cma

    tgt_n = apply_norm(target_feats, stats)

    def objective(u: np.ndarray) -> float:
        p = unit_to_params(u)
        out = render(base, sr, p)
        f = apply_norm(extract_features(out, sr), stats)
        return _clipped_dist(f, tgt_n)

    # reference distances: unprocessed base, and neutral-knob render
    dist_raw = _clipped_dist(apply_norm(extract_features(base, sr), stats), tgt_n)
    x0 = np.full(len(KNOB_NAMES), 0.5)
    dist_neutral = objective(x0)

    es = cma.CMAEvolutionStrategy(x0, 0.25, {
        "bounds": [0.0, 1.0], "maxfevals": maxfevals,
        "seed": seed, "verbose": -9, "verb_disp": 0,
    })
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [objective(x) for x in sols])
    best = es.result.xbest
    return unit_to_params(best), dist_raw, float(es.result.fbest), dist_neutral


def run(base_path: str, target_path: str, out_path: str | None,
        seconds: float, maxfevals: int) -> dict:
    base_full, sr = load(base_path, sr=SR)
    target_full, _ = load(target_path, sr=SR)

    n = int(seconds * sr)
    base_opt = base_full[:n] if n and len(base_full) > n else base_full
    target_feats = extract_features(target_full, sr)   # characterize full target
    stats = get_stats(base_opt, sr)

    params, d_raw, d_best, d_neutral = optimize(base_opt, sr, target_feats, stats, maxfevals)

    # apply winning knobs to the FULL base vocal and save for listening
    if out_path is None:
        bstem = os.path.splitext(os.path.basename(base_path))[0].replace(" ", "_")
        tstem = os.path.splitext(os.path.basename(target_path))[0].replace(" ", "_")
        out_dir = os.path.join(os.path.dirname(__file__), "..", "matches")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{bstem}__matched_to__{tstem}.wav")
    matched = render(base_full, sr, params)
    sf.write(out_path, matched, sr)
    json.dump(params, open(out_path.replace(".wav", "_knobs.json"), "w", encoding="utf-8"), indent=2)

    reduction = 100.0 * (d_raw - d_best) / d_raw if d_raw else 0.0
    return {
        "dist_raw_base_vs_target": round(d_raw, 3),
        "dist_neutral_knobs": round(d_neutral, 3),
        "dist_best_match": round(d_best, 3),
        "pct_closer_than_raw": round(reduction, 1),
        "out_wav": os.path.abspath(out_path),
        "params": params,
    }


def _demo() -> None:
    """Make a target by applying KNOWN knobs to a base, then confirm the optimizer
    drives the distance far down (the loop works)."""
    from knobs import render as _r, random_params
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..",
                                           "data", "raw_vocal_samples", "*.wav")))
    sr = SR
    if files:
        base, _ = load(files[0], sr=sr)
        base = base[: 6 * sr]
    else:
        t = np.linspace(0, 6, 6 * sr, endpoint=False)
        base = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    known = random_params(np.random.default_rng(3))
    target = _r(base, sr, known)
    target_feats = extract_features(target, sr)
    stats = get_stats(base, sr)

    _, d_raw, d_best, d_neutral = optimize(base, sr, target_feats, stats, maxfevals=900)
    print(f"demo: dist raw={d_raw:.2f} neutral(start)={d_neutral:.2f} best={d_best:.2f} "
          f"({100*(d_raw-d_best)/d_raw:.0f}% closer than raw)")
    # target IS reachable (base+knobs): the search must improve clearly on its start
    # and land well below the unprocessed distance.
    assert d_best < d_neutral and d_best < 0.5 * d_raw, (d_best, d_raw, d_neutral)
    print("OK: optimizer improved on its start and more than halved the raw distance")


def main() -> None:
    ap = argparse.ArgumentParser(description="Match base vocal's tone to a target vocal.")
    ap.add_argument("base", nargs="?", help="my dry vocal .wav")
    ap.add_argument("target", nargs="?", help="target isolated vocal .wav")
    ap.add_argument("--out", default=None, help="output matched .wav path")
    ap.add_argument("--seconds", type=float, default=8.0, help="audio length used for the search")
    ap.add_argument("--maxfevals", type=int, default=1250, help="optimizer budget (more = slower, closer; ~1250 is the taper point)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        _demo()
        return
    if not (args.base and args.target):
        ap.error("pass <base.wav> <target.wav>, or --demo")

    res = run(args.base, args.target, args.out, args.seconds, args.maxfevals)
    print(json.dumps({k: v for k, v in res.items() if k != "params"}, indent=2))
    print(f"\nsaved matched vocal -> {res['out_wav']}")
    print("listen: does it sit tonally like the target? distance going down != sounds right.")


if __name__ == "__main__":
    main()
