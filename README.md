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
  vocal_data/
    raw/                          my dry vocals, the input to match (mono 44.1k wav)
    mixed/                        target pre-mixed reference vocals (30s mono)
    matched/                      optimizer output: matched wavs + knob json
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
  winning knobs to the full vocal and writes it to `vocal_data/matched/`.

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
                 render(full base, best knobs)  ──►  vocal_data/matched/…_matched.wav
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

## Usage — walkthrough

Match one dry vocal (a file in `vocal_data/raw/`) to one pre-mixed target (a file in
`vocal_data/mixed/`). Worked example with files already in the repo:

**1. Pick your raw vocal.** One of your dry recordings:

```
vocal_data/raw/2026-11-04 09-32-20.wav
```

**2. Pick the pre-mixed vocal to copy the tone of.** A reference whose sound you want:

```
vocal_data/mixed/us - lil peep - 149bpm - 72 bars.wav
```

**3. Run the optimizer** (base first, target second):

```bash
poetry run python scripts/optimize_match.py \
    "vocal_data/raw/2026-11-04 09-32-20.wav" \
    "vocal_data/mixed/us - lil peep - 149bpm - 72 bars.wav"
```

It searches knob settings on the first 8 seconds, then renders your full vocal with the
winning knobs. ~15s one-time numba compile on the first run, then the search.

**4. Listen to the result.** Two files land in `vocal_data/matched/`:

```
vocal_data/matched/2026-11-04_09-32-20__matched_to__us_-_lil_peep_-_149bpm_-_72_bars.wav
...__matched_to__..._knobs.json   # the knob values it chose
```

The console also prints the raw-vs-target distance, the best matched distance, and the
**% closer** the match got. For the example above it reported the matched vocal landing
**14.5% closer** to the SuicideBoys target than the dry take.

### Hear it

<!-- Audio is hosted via a GitHub issue (drag the wavs in, paste the resulting
     *.githubusercontent.com URLs below). Local .wav paths won't play on github.com. -->

**Raw** — my dry vocal:

https://github.com/REPLACE_WITH_RAW_URL

**Target** — the pre-mixed SuicideBoys vocal whose tone we're copying:

https://github.com/REPLACE_WITH_TARGET_URL

**Matched** — my vocal after the optimizer applied its knobs (14.5% closer to the target):

https://github.com/REPLACE_WITH_MATCHED_URL

Options:

| Flag | Meaning | Default |
|------|---------|---------|
| `--out PATH` | where to write the matched wav | `vocal_data/matched/<base>__matched_to__<target>.wav` |
| `--seconds N` | audio length used during the search (longer = slower) | `8` |
| `--maxfevals N` | optimizer budget — more evals = closer match, slower | `1250` |

**Judge by ear, not just the number.** A lower distance means the features line up; it
doesn't guarantee it sounds right. Audio is mono 44.1 kHz throughout. Audio files are
inputs/outputs, not tracked as source.
