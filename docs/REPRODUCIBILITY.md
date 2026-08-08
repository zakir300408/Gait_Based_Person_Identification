# Reproducing the PRISM-Gait paper

Run commands from the repository root. The stored random seed is 42 unless a
script explicitly records branch seeds 42, 43, and 44 for MiniROCKET.

## 1. Verify the release

```powershell
conda activate prism-gait
python scripts/verify_release.py
pytest -q
```

## 2. Rebuild the paper figures from released evidence

```powershell
python scripts/rebuild_paper_figures.py
```

This regenerates:

- treadmill identification benchmark;
- duration selection and ablation;
- identification-rank and subject-held-out verification curves;
- operational timing.

## 3. Re-run PRISM-Gait analyses from AlterGait

```powershell
python -m ablation.prism_architecture_selection
python -m ablation.prism_duration_selection
python -m ablation.prism_subject_window_accuracy
python -m ablation.prism_nested_design_evaluation
```

Core implementation mapping:

| Component | Source |
|---|---|
| Windowing, movement gate, HCGF | `ablation/pelvis_harmonic_covariance_model.py` |
| MiniROCKET branches | `ablation/pelvis_mblite_fusion.py` |
| CDKC | `ablation/pelvis_kernel_duel_model.py` |
| Fusion, scaling, Ridge | `ablation/mblite_verification_suite.py` |
| Nested duration/composition selection | `ablation/prism_nested_design_evaluation.py` |
| Subject-held-out and frozen-threshold verification | `ablation/export_closed_verification_trial_scores.py` and verification scripts |

## 4. Re-run matched comparison models

| Model | Entry point | Environment |
|---|---|---|
| MiniROCKET and MultiROCKET | `ablation/run_strict_published_rocket_benchmarks.py` | `prism-gait` |
| FCN-BiLSTM | `ablation/fcn_bilstm_official_benchmark.py` | `prism-gait-deep` |
| InceptionTime | `ablation/run_inceptiontime_author_ensemble_treadmill.py` | `gaitid-inceptiontime` |
| Multi-Model LSTM | `ablation/multi_model_lstm_official_benchmark.py` | `prism-gait-deep` |

The deep-model scripts train from enrollment data. Checkpoints are deliberately
excluded because the paper reports deterministic configurations and evaluated
predictions rather than distributing deployment models.

## 5. Public benchmarks

After placing the licensed datasets as documented in `docs/DATA.md`, run:

```powershell
python -m ablation.public_imu_benchmark
python -m ablation.public_exact_prism_registration
```

The public test partitions never fit MiniROCKET, scaling, or Ridge parameters.
Each public cohort receives a separately fitted identity classifier.

## 6. Tables and numerical provenance

`paper/tables/` contains every displayed value as CSV and the corresponding
LaTeX source. Unrounded values are retained in CSV. Figure-level statistical
exports are stored beside their inputs under `baselines_out/`.
