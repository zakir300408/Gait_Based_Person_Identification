# Data access and release boundaries

## AlterGait

`gait_id_pelvis_62.h5` is the analysis file used for the 62-participant
AlterGait experiments. Its SHA-256 digest is
`1454c65a91054ae0c5ce70d897ddce0f584bb5865dca2faecb044323f7f969e5`.

The HDF5 root contains 62 pseudonymous participant groups and one `index`
group. Participant groups contain baseline and controlled gait-condition
recordings. The default sampling frequency is 100 Hz.

This file was already tracked in the public repository before this
reproducibility update. Do not mirror it to Figshare until the authors confirm
that the applicable ethics approval, consent language, and institutional policy
permit that additional distribution channel.

## WHU-Gait Dataset #3

The repository does not redistribute WHU-Gait. Obtain the released files from
the source linked by the original project:

- Dataset folder: <https://drive.google.com/drive/folders/1KOm-zROeOZH3e2tqYUpHAvIaBZSJGFm_>
- Original code and dataset documentation:
  <https://github.com/qinnzou/Gait-Recognition-Using-Smartphones>

Expected location:

```text
public_datasets/whugait/Dataset #3/Dataset #3/
```

The evaluated split contains 26,283 training windows and 2,991 test windows,
each with 128 samples and six inertial channels.

## OU-ISIR inertial gait data

Register for the database through the official provider:

<https://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html>

The paper evaluates the Zou-preprocessed identification split at:

```text
public_datasets/whugait/OU-ISIR_sensors_dataset/identification/data/
```

The released split contains 13,212 training windows, 1,409 test windows, and
745 identity labels. The official center-IMUZ demographic roster contains 744
participants; the manuscript keeps this distinction explicit.

## Figshare archive

`python scripts/package_figshare.py` packages only derived paper evidence. It
excludes the AlterGait HDF5 file, WHU-Gait, OU-ISIR, model checkpoints, caches,
and third-party code. Review the generated file list and ethics constraints
before upload.
