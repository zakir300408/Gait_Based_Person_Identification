# GaitPrint-IMU

Code and recordings for **GaitPrint-IMU**, the model presented in the manuscript

> **Wearer identification and verification using inertial signals from the lower back during controlled trunk sway**

The study uses AlterGait lower-back inertial measurement unit (IMU) recordings from 62 adults, including 38 healthy participants and 24 with lower-limb osteoarthritis. Participants completed baseline treadmill walking and up to nine instructed gait modifications in one session.

## Model

GaitPrint-IMU combines three feature branches from six acceleration and angular-velocity channels sampled at 100 Hz. Each two-second window produces 13,049 features.

| Branch | Representation | Features |
| --- | --- | ---: |
| Temporal | [MiniROCKET](https://doi.org/10.1145/3447548.3467231) transforms of the signal and its first and second differences | 7,812 |
| Multiscale | Signed competing-kernel features adapted from [HYDRA](https://doi.org/10.1007/s10618-023-00939-3) | 4,608 |
| Spectral–coordination | Signal statistics, spectra, relative phase, correlations and log-covariance | 629 |

The branches are concatenated, normalized and scaled before ridge classification. Identity models are fitted on baseline walking and evaluated on modified gait. The evaluation covers closed-set identification, one-to-one verification and open-set identification.

Baseline-trained GaitPrint-IMU achieved 92.8% window-level accuracy during small trunk sway and 87.0% during large sway. At independently calibrated thresholds targeting a 0.1% false acceptance rate, one-to-one verification yielded an observed false acceptance rate of 0.095% and a false rejection rate of 16.74%. Open-set false positive identification rate was 4.72% with galleries of 49–50 enrolled identities.

These results describe controlled treadmill walking in one session with fixed sensor placement. Repeated-session, sensor-reattachment and free-living performance have not been established.

## Comparator models

The six primary comparators are [MiniROCKET](https://doi.org/10.1145/3447548.3467231), [MultiROCKET](https://doi.org/10.1007/s10618-022-00844-1), [HYDRA](https://doi.org/10.1007/s10618-023-00939-3), MultiROCKET–HYDRA (a combination of the preceding two representations), [InceptionTime](https://doi.org/10.1007/s10618-020-00710-y) and [Multi-Model long short-term memory (LSTM)](https://doi.org/10.1109/ACCESS.2021.3056880). Author implementation sources and their terms are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

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

Recalculates the analyses from saved predictions, checks the reference results, and generates tables and the result figures in `outputs/paper/`.

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

## Citation and license

See [CITATION.cff](CITATION.cff), the [license](LICENSE.md), and [third-party sources](THIRD_PARTY.md).
