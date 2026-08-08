# PRISM-Gait reproducibility repository

This repository accompanies **PRISM-Gait: Robust Inertial Biometrics for
Wearable Authentication Under Controlled Trunk-Sway Perturbations**. It
contains the implementation, matched comparison models, compact analysis
outputs, figure source data, and table source data used in the manuscript.

The release is deliberately scoped to the current paper. It excludes unrelated
FoG, EEG, Kyoto, exploratory model-search, cached checkpoints, logs, previews,
and superseded figures.

## Included

- `ablation/`: PRISM-Gait, duration and component analyses, nested selection,
  verification, public-dataset evaluation, and the five matched comparators.
- `figs/`: reproducible scripts for the paper result figures.
- `baselines_out/`: only the compact derived evidence required by those figures.
- `paper/figures/`: the exact figures placed in the manuscript.
- `paper/tables/`: paper-ready LaTeX tables and their CSV source values.
- `third_party/`: the minimal pinned author-code files required by InceptionTime
  and Multi-Model LSTM, with upstream provenance.
- `gait_id_pelvis_62.h5`: the AlterGait analysis file already distributed in
  this repository.

## Quick verification

```powershell
conda env create -f environment.yml
conda activate prism-gait
python scripts/verify_release.py
python scripts/rebuild_paper_figures.py
pytest -q
```

`verify_release.py` checks the dataset checksum and schema, required evidence,
table files, figure labels, and accidental machine-specific paths.

## Reproducing the analyses

The complete command order and the mapping from manuscript results to scripts
are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Data acquisition,
licensing, expected directory layouts, and the large-data policy are in
[docs/DATA.md](docs/DATA.md). The exact feature dimensions and model-component
definitions are in [docs/PRISM_GAIT_MODEL.md](docs/PRISM_GAIT_MODEL.md).

The classical PRISM-Gait and random-kernel analyses use `environment.yml`.
FCN-BiLSTM and Multi-Model LSTM use `environment-deep.yml`. InceptionTime uses
`environment-inceptiontime.yml` because its reported run used TensorFlow 2.10.

## Large data and Figshare

Raw WHU-Gait and OU-ISIR files are not redistributed here. Obtain them from
their official providers and place them as described in `docs/DATA.md`.

The command below creates a Figshare-ready ZIP containing only paper-derived
scores, fold assignments, source tables, and checksums. It excludes raw sensor
data and third-party datasets.

```powershell
python scripts/package_figshare.py
```

Publication of that archive requires an author-controlled Figshare account and
confirmation that participant-level derived scores are covered by the study's
data-sharing approval. No DOI is claimed until Figshare completes the deposit.

## Citation and license

A publication DOI will be added after acceptance. Code and data-use conditions
are described in [LICENSE.md](LICENSE.md). Third-party files retain their
upstream licenses.
