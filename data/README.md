# Dataset setup

Run all commands from the repository root. Do not rename the extracted folders;
the loaders use the paths shown below.

## AlterGait

The analysis file is already included at:

```text
data/altergait/gait_id_pelvis_62.h5
```

## WHU-Gait Dataset #3

- Project: <https://github.com/qinnzou/Gait-Recognition-Using-Smartphones>
- Dataset download folder: <https://drive.google.com/drive/folders/1KOm-zROeOZH3e2tqYUpHAvIaBZSJGFm_>

Download and extract Dataset #3 so that the repository contains:

```text
data/whu_gait/Dataset #3/Dataset #3/
  train/y_train.txt
  train/Inertial Signals/train_acc_x.txt
  test/y_test.txt
  test/Inertial Signals/test_acc_x.txt
```

The other five accelerometer/gyroscope channel files remain beside the example
channel files shown above.

## OU-ISIR

- Official database and release agreement: <https://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html>
- Zou-preprocessed identification split: <https://drive.google.com/drive/folders/1KOm-zROeOZH3e2tqYUpHAvIaBZSJGFm_>

After obtaining permission and downloading `OU-ISIR_sensors_dataset.zip`, place
the identification split at:

```text
data/ou_isir/identification/data/
  train/train_y.txt
  train/data/train_acc_x.txt
  test/test_y.txt
  test/data/test_acc_x.txt
```

Run the public evaluations after both datasets are in place:

```powershell
python -m analysis.run_public_datasets all
```
