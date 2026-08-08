# PRISM-Gait

Code and results for **PRISM-Gait: Robust Inertial Biometrics for Wearable
Authentication Under Controlled Trunk-Sway Perturbations**.

## Repository layout

- `analysis/prism_gait/`: our PRISM-Gait implementation
- `analysis/comparators/`: adaptations of the public comparison models
- `analysis/datasets/`: WHU-Gait and OU-ISIR loaders
- `analysis/figures/`: manuscript figure code
- `data/`: dataset download instructions and the AlterGait analysis file
- `paper/`: final result figures and table sources

## Setup

```powershell
conda env create -f environment.yml
conda activate prism-gait
Expand-Archive results.zip -DestinationPath .
```

See [data/README.md](data/README.md) for the exact public-dataset download links
and folder locations.

## Run

```powershell
python -m analysis.run_prism_gait
python -m analysis.run_benchmark_models rocket
python -m analysis.run_public_datasets all
python -m analysis.run_figures
```

The comparison models and their original GitHub repositories are listed in
[analysis/comparators/README.md](analysis/comparators/README.md).
