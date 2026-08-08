# Plot-ready treadmill accuracy data

## Recommended files

- `participant_window_accuracy_long.csv`: one row per method, participant, and condition. Use
  `window_accuracy_percent` for box, raincloud, violin, swarm, ridge, or paired-distribution plots.
- `participant_window_accuracy_wide.csv`: the same continuous values with one method per column.
  Use this version for paired tests or connected participant plots.
- `condition_accuracy_long.csv`: one unpooled row per method, condition, and accuracy level
  (`Window` or `Trial`). Use for point, bar, heatmap, slope, or faceted condition comparisons.

## Important metric distinction

`participant_window_accuracy_long.csv` contains the continuous fraction of correctly identified
2-s windows within one participant's nominal 1-min condition trial. This is appropriate for
distribution plots. The `Trial` rows in `condition_accuracy_long.csv` are condition-level
identification rates after majority vote within each trial. Individual majority-vote trial
correctness was not stored for every method, so binary trial rows are intentionally not fabricated.

## Key columns

- `model`: reader-facing method name.
- `method_role`: published comparator or PRISM-Gait (`This study`).
- `condition_code`: stable machine-readable condition identifier.
- `condition_label`: reader-facing condition name.
- `condition_family`: perturbation family for grouping or faceting.
- `perturbation_level`: small, large, or single level.
- `is_trunk_sway`: Boolean filter for the trunk-sway analyses.
- `accuracy_fraction`: condition accuracy on a 0-1 scale.
- `accuracy_percent`: condition accuracy on a 0-100 scale.
- `accuracy_level_order`: stable ordering with window first and trial second.
- `window_accuracy_percent`: participant/trial window accuracy on a 0-100 scale.
- `n_observations` / `n_windows`: denominator for reliability checks.
- `method_order` / `condition_order`: stable plotting order.
