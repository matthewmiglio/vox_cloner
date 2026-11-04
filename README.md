# vox_cloner

## Repo purpose

Make **my dry vocal** sound tonally like a **target mixed vocal** (a reference song's
vocal) by automatically finding DSP ("knob") settings — EQ, compression, saturation,
chorus, delay, reverb, gain.

It works **per pair** with an optimizer: no training, no model, no dataset. Given my
vocal and a target vocal, a search tries knob settings, renders my vocal through them,
scores how close the result is to the target in a feature space, and homes in on the
best match.

> An earlier neural-net approach (predict knobs from a big dataset) was tried and
> dropped — it couldn't recover knob settings from feature differences (barely beat
> guessing the average), and it never matched the real cross-singer goal. The optimizer
> needs none of that.

## Lay of the land

```
vox_cloner/
  README.md                       this file
  CLAUDE.md                       orientation for AI agents working in the repo
  scripts/
    knobs.py                      32-knob pedalboard chain; param spec; render()
    audio_feature_extraction.py   61-dim feature vector (tone + dynamics); z-score
    optimize_match.py             CMA-ES optimizer: match base -> target, save matched wav
  data/
    raw_vocal_samples/            my dry vocals (mono 44.1k wav)
    raw_vocal_clips/              more of my dry clips (30s)
    mixed_vocal_clips/            target reference vocals (30s mono)
  matches/                        optimizer output: matched wavs + knob json + summary
  pyproject.toml / poetry.lock    Poetry env (pinned numpy/librosa + cma)
```

Three modules, each with a runnable self-check (`--demo` / run directly):

- **`knobs.py`** — `KNOBS` defines each knob as `(min, max, default)`; that order is the
  optimizer's vector order. `render(audio, sr, params)` applies the chain
  (HPF/LPF → EQ → compressor → saturation → chorus → delay → reverb → gain). No
  autotune/pitch — those depend on the performance, not the target's tone.
- **`audio_feature_extraction.py`** — `extract_features(audio, sr)` → 61 numbers:
  MFCCs + deltas, spectral stats, crest factor, loudness (LUFS), sibilance ratio.
  Captures tone and dynamics, not melody, so two different singers can be compared.
- **`optimize_match.py`** — the optimizer. Minimizes a **clipped** z-scored feature
  distance (each dimension capped at ±4σ so a few features my vocal can't reach don't
  dominate the score). Searches on the first few seconds for speed, then applies the
  winning knobs to the full vocal and writes it to `matches/`.

## System diagram

```
  my dry vocal (base.wav)                 target mixed vocal (target.wav)
          │                                          │
          │                                   extract_features
          │                                          │
          │                                   target feature vector ──┐
          ▼                                                           │
  ┌───────────────────────────  CMA-ES search loop  ─────────────┐    │
  │                                                              │    │
  │   knob setting ──► render(base, knobs) ──► extract_features  │    │
  │        ▲                                        │            │    │
  │        │                                        ▼            │    │
  │        │                         clipped z-scored distance ◄─┼────┘
  │        │                                        │            │
  │        └──────── pick next setting (lower) ◄────┘            │
  │                     (repeat ~N evals)                        │
  └──────────────────────────────┬───────────────────────────────┘
                                 │ best knobs
                                 ▼
                 render(full base, best knobs)  ──►  matches/…_matched.wav
                                 └─────────────► …_knobs.json
```

## Setup

Requires [Poetry](https://python-poetry.org/) and Python 3.11.

```bash
cd M:\my_files\my_programs\vox_cloner
poetry install            # builds the env in D:\.venvs from the lockfile
```

Pinned on purpose: **numpy 1.26.4 + librosa 0.10.2**. The machine's global Python 3.11
has numpy 2.4.4 + librosa 0.11, where `librosa.stft` **hangs forever** — always run
through Poetry, never global `python`.

Verify:

```bash
poetry run python scripts/knobs.py                     # 32 knobs OK
poetry run python scripts/audio_feature_extraction.py  # 61 features OK
poetry run python scripts/optimize_match.py --demo     # optimizer recovers a target
```

(The first librosa call in a fresh process spends ~15s on a one-time numba compile.)

## Usage

Match one dry vocal to one target:

```bash
poetry run python scripts/optimize_match.py \
    "data/raw_vocal_samples/my_take.wav" \
    "data/mixed_vocal_clips/us - lil peep - 149bpm - 72 bars.wav"
```

Options:

| Flag | Meaning | Default |
|------|---------|---------|
| `--out PATH` | where to write the matched wav | `matches/<base>__matched_to__<target>.wav` |
| `--seconds N` | audio length used during the search (longer = slower) | `8` |
| `--maxfevals N` | optimizer budget — more evals = closer match, slower | `1250` |

Output per run: the full matched vocal (`matches/…_matched…wav`) and the chosen knob
values (`…_knobs.json`). It also prints the distance of the raw vocal vs the target,
the best matched distance, and the **% closer** the match got.

**Judge by ear, not just the number.** A lower distance means the features line up; it
doesn't guarantee it sounds right. Audio is mono 44.1 kHz throughout. Audio files are
inputs/outputs, not tracked as source.
