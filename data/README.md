# Recordings

`altergait/gait_id_pelvis_62.h5` contains the lower-back inertial recordings collected for this study: 62 participants, 62 baseline trials and 538 altered-gait trials. Participants completed up to nine modifications; availability varies by condition. The historical directory name is retained for file compatibility.

The HDF5 `index` group supplies parallel arrays `subject`, `condition`, `trial_id` and `h5_group`. Each indexed recording group contains `imu` (samples × channels) and `time_s`. Read channel names from `columns_json` and sampling rate from `fs_hz` attributes; the analysis selects three pelvis acceleration and three angular-velocity channels at 100 Hz. Acceleration is in m/s² and angular velocity is in degrees/s, as recorded in the channel names.

`analysis.data` generates accepted windows and records the source trial, start/end samples, participant, condition and partition for every window. It verifies raw-time separation between baseline selection-training and validation. Final enrollment uses the complete accepted baseline recording; modified walking is reserved for evaluation. Identity-held-out folds are fixed by the configuration and exported to `subject_folds.csv`.

The original HDF5 SHA256 is `1454c65a91054ae0c5ce70d897ddce0f584bb5865dca2faecb044323f7f969e5`. Both the full-training input check and the supplied evidence manifest identify this source.
