# GaitPrint-IMU

Code and recordings for **GaitPrint-IMU: Wearer Identification and Verification Under Controlled Trunk Sway**.

We collected lower-back IMU recordings from 62 participants during baseline walking and up to nine instructed gait modifications.

## Setup

Install Conda and Git, then run:

```sh
git clone https://github.com/zakir300408/Gait_Based_Person_Identification.git
cd Gait_Based_Person_Identification
conda env create -f environment.yml
conda activate gaitprint
```

## Reproduce results

```sh
python -m analysis.reproduce
```

Recalculates the analyses from saved predictions, checks the reference results, and generates tables and Figures 3–5 in `outputs/paper/`.

## Train models

```sh
conda env create -f environment-neural.yml
conda activate gaitprint
python -m analysis.train
```

Runs all models and analyses with the study settings in `configs/`. Results are saved in `outputs/retrained/paper/`. The neural environment is selected automatically. GPU training requires an NVIDIA driver compatible with CUDA 12.8; training times depend on hardware.

## Files

- `analysis/`: models, training, evaluation and plotting.
- `configs/`: preprocessing, model and evaluation settings.
- `data/`: recordings and [format description](data/README.md).
- `results/`: saved predictions and reference results.
- `paper/`: result figures.

The figures in `paper/` match the manuscript sent to co-authors for review for *npj Digital Medicine*. Comparator reference numbers in the plotting code follow that manuscript bibliography.

## Citation and license

See [CITATION.cff](CITATION.cff), the [license](LICENSE.md), and [third-party sources](THIRD_PARTY.md).
