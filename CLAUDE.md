# vox_cloner

Make **my dry vocal** sound tonally like a **target mixed vocal** (e.g. a reference
song's vocal) by auto-finding DSP ("knob") settings. Done per-pair by an **optimizer** —
no training, no model, no dataset.

```
knobs → render(my_vocal, knobs) → features() → distance to features(target) → minimize
```

For each (my_vocal, target) pair, a search (CMA-ES) tries knob settings, renders my
vocal, scores it by feature distance to the target, and homes in. The NN/dataset path
was tried and dropped: it couldn't recover knobs from feature deltas (MAE ≈ guessing),
and it never saw the real cross-singer task. See `README.md` for the full overview.

## Layout

```
README.md                         overview, setup, usage, system diagram
scripts/
  knobs.py                        pedalboard chain; KNOBS spec; vector<->params; render()
  audio_feature_extraction.py     librosa+pyloudnorm 61-dim vector; z-score; distance()
  optimize_match.py               the optimizer: match base -> target, save matched wav
vocal_data/
  raw/                            my dry vocals, the input to match (OBS recordings -> mono 44.1k wav)
  mixed/                          target pre-mixed reference vocals (30s mono from the sample lib)
  matched/                        optimizer output: matched wavs + knob json
pyproject.toml / poetry.lock      Poetry env (pinned numpy/librosa + cma)
```
Env lives in `D:\.venvs` (Poetry-managed, out of repo). Run everything via `poetry run`.

## The three modules

- **`scripts/knobs.py`** — `KNOBS` dict defines each of 32 knobs as `(min, max, default)`;
  that order *is* the optimizer vector order. `render(audio, sr, params)` applies the
  pedalboard chain (HPF/LPF → EQ → comp → sat → chorus → delay → reverb → gain). Pass a
  param dict or a raw vector (auto-clipped). Helpers: `params_to_unit`/`unit_to_params`
  (↔ [0,1] search space), `random_params`. No autotune/pitch — not a function of the
  target's tone.

- **`scripts/audio_feature_extraction.py`** — `extract_features(audio, sr)` → 61-dim
  vector (MFCC+deltas, spectral stats, crest, LUFS, sibilance ratio). `fit_norm`/
  `apply_norm` z-score; `distance(a, b, stats)` compares two vectors.

- **`scripts/optimize_match.py`** — `poetry run python scripts/optimize_match.py <base.wav>
  <target.wav>`. Searches knobs (CMA-ES) to minimize **clipped** z-scored feature distance
  (per-dim diff clipped to ±4σ so a few unreachable features don't dominate). Scaling is
  self-contained (built from random renders of the base). Analyzes the first `--seconds`
  (default 8) for speed, applies the winning knobs to the full vocal, saves to `vocal_data/matched/`.
  `--demo` self-check recovers a reachable target. `--maxfevals` = budget (more = closer).

## Conventions

- **Run everything through Poetry, not global python.** Global Python 3.11 has
  numpy 2.4.4 + librosa 0.11, where `librosa.stft` **deadlocks** (hangs forever). The
  Poetry env (`D:\.venvs`, Python 3.11) pins the known-good **numpy 1.26.4 + librosa 0.10.2**.
  ```
  cd M:\my_files\my_programs\vox_cloner
  poetry run python scripts/optimize_match.py <base.wav> <target.wav>
  poetry install     # rebuild env   |   poetry sync   # prune removed deps
  ```
- First librosa call in a fresh process spends ~15s on numba JIT compile, then cached.
- Audio is mono float32 at 44100 Hz throughout.
- Self-checks (all pass): `poetry run python scripts/knobs.py`,
  `... audio_feature_extraction.py`, `... optimize_match.py --demo`.
