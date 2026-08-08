# PRISM-Gait

Code, results, figures, and tables for **PRISM-Gait: Robust Inertial
Biometrics for Wearable Authentication Under Controlled Trunk-Sway
Perturbations**.

## Contents

- `ablation/`: PRISM-Gait, ablation, verification, public-dataset, and comparison-model analyses
- `figs/`: scripts used to create the manuscript figures
- `paper/`: final figures and table source files
- `baselines_out/`: numerical results used by the figures and tables

## Run

```powershell
conda env create -f environment.yml
conda activate prism-gait
python scripts/verify_release.py
python scripts/rebuild_paper_figures.py
```

Main PRISM-Gait analyses:

```powershell
python -m ablation.prism_architecture_selection
python -m ablation.prism_duration_selection
python -m ablation.prism_nested_design_evaluation
python -m ablation.prism_subject_window_accuracy
```

The comparison-model entry points are in `ablation/`. Deep-learning models use
`environment-deep.yml`; InceptionTime uses `environment-inceptiontime.yml`.

## Model sources

- [MiniROCKET](https://github.com/angus924/minirocket)
- [MultiROCKET](https://github.com/ChangWeiTan/MultiRocket)
- [FCN-BiLSTM](https://github.com/utshabkg/gait_recognition)
- [InceptionTime](https://github.com/hfawaz/InceptionTime/tree/952d115e83fb3a66d75c5858c702f8dd5eeb18b7)
- [Multi-Model LSTM](https://github.com/halam189/Gait_Recognition_LSTM)

## Public datasets

- [WHU-Gait](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones)
- [OU-ISIR inertial gait database](https://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html)

Please follow the original code and dataset licenses.
