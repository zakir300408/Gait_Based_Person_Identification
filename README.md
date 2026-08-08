# PRISM-Gait

Code and results for **PRISM-Gait: Robust Inertial Biometrics for Wearable
Authentication Under Controlled Trunk-Sway Perturbations**.

## Repository

- `analysis/` contains the PRISM-Gait implementation, evaluations, and figure scripts.
- `paper/` contains the manuscript figures and `tables.zip`.
- `results.zip` contains the result files used by the figures and tables.
- `gait_id_pelvis_62.h5` contains the AlterGait analysis data.

## Run

```powershell
conda env create -f environment.yml
conda activate prism-gait
Expand-Archive results.zip -DestinationPath .
python -m analysis.prism_nested_design_evaluation
python -m analysis.treadmill_identification_overview
python -m analysis.prism_design_selection_figure
python -m analysis.identification_rank_robustness
```

## Model sources

- [MiniROCKET](https://github.com/angus924/minirocket)
- [MultiROCKET](https://github.com/ChangWeiTan/MultiRocket)
- [FCN-BiLSTM](https://github.com/utshabkg/gait_recognition)
- [InceptionTime](https://github.com/hfawaz/InceptionTime)
- [Multi-Model LSTM](https://github.com/halam189/Gait_Recognition_LSTM)

## Public datasets

- [WHU-Gait](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones)
- [OU-ISIR](https://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html)
