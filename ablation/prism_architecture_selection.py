"""Select the smallest participant-non-inferior PRISM-Gait configuration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ablation.mblite_ablation_study import (
    evaluate_variant,
    load_original_h5_windowed,
    set_seed,
    write_csv,
)


@dataclass(frozen=True)
class Settings:
    h5_path: Path
    reference_subject_path: Path
    reference_summary_path: Path
    output_dir: Path
    seed: int
    acc_std_min: float
    pkd_groups: int
    pkd_kernels_per_group: int
    pkd_dilations: tuple[int, ...]
    batch_size: int
    bootstrap_repetitions: int
    noninferiority_margin_pct: float


VARIANTS = (
    {"variant": "PRISM-24", "label": "PRISM-Gait, half CDKC", "branches": ("raw", "diff", "diff2"), "pkd": True, "phcf": True, "pkd_groups": 24},
    {"variant": "PRISM-12", "label": "PRISM-Gait, quarter CDKC", "branches": ("raw", "diff", "diff2"), "pkd": True, "phcf": True, "pkd_groups": 12},
    {"variant": "Core+PHCF", "label": "Temporal core + HCGF", "branches": ("raw", "diff", "diff2"), "pkd": False, "phcf": True, "pkd_groups": 0},
)


def create_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("prism_architecture_selection")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    path = output_dir / f"prism_architecture_selection_{datetime.now():%Y%m%d_%H%M%S}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def bootstrap_interval(delta: np.ndarray, repetitions: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(repetitions, len(delta)))
    means = delta[indices].mean(axis=1)
    return float(delta.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compare_to_reference(reference: pd.DataFrame, candidate: pd.DataFrame, settings: Settings) -> list[dict]:
    merged = reference.merge(candidate, on=["condition", "subject"], suffixes=("_reference", "_candidate"), validate="one_to_one")
    rows = []
    for condition, group in merged.groupby("condition", sort=True):
        delta = 100 * (group["subject_window_accuracy_candidate"] - group["subject_window_accuracy_reference"]).to_numpy()
        mean, lower, upper = bootstrap_interval(delta, settings.bootstrap_repetitions, settings.seed)
        rows.append({"scope": condition, "participants": len(delta), "mean_delta_pct": mean, "ci_lower_pct": lower, "ci_upper_pct": upper})
    by_subject = merged.groupby("subject", observed=True)[["subject_window_accuracy_reference", "subject_window_accuracy_candidate"]].mean()
    delta = 100 * (by_subject.iloc[:, 1] - by_subject.iloc[:, 0]).to_numpy()
    mean, lower, upper = bootstrap_interval(delta, settings.bootstrap_repetitions, settings.seed + 1)
    rows.append({"scope": "overall_participant_mean", "participants": len(delta), "mean_delta_pct": mean, "ci_lower_pct": lower, "ci_upper_pct": upper})
    return rows


def choose_variant(rows: pd.DataFrame, dimensions: dict[str, int], margin_pct: float) -> str:
    eligible = []
    for variant, group in rows.groupby("variant", sort=False):
        required = group.loc[group["scope"].isin(["trunk_sway_large", "overall_participant_mean"])]
        if len(required) != 2:
            raise ValueError(f"missing primary comparison scopes for {variant}")
        if (required["ci_lower_pct"] >= -margin_pct).all():
            eligible.append(variant)
    return min(eligible, key=lambda name: dimensions[name]) if eligible else "PRISM-Gait"


def run(settings: Settings) -> tuple[Path, Path, Path]:
    for path in (settings.h5_path, settings.reference_subject_path, settings.reference_summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path.name}")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(settings.output_dir)
    reference_summary = pd.read_csv(settings.reference_summary_path)
    reference_dim = set(reference_summary["feature_dim"].astype(int))
    if reference_dim != {13049}:
        raise ValueError("reference result must be the exact 13,049-feature PRISM-Gait model")
    reference = pd.read_csv(settings.reference_subject_path)
    reference = reference.loc[reference["variant"].eq("PRISM-Gait"), ["condition", "subject", "subject_window_accuracy"]]
    logger.info("step_01: loading treadmill windows and exact PRISM reference participants=%s", reference["subject"].nunique())
    data = load_original_h5_windowed(settings.h5_path, 2.0, settings.acc_std_min)
    summary_rows, subject_frames, dimensions = [], [], {"PRISM-Gait": 13049}
    for index, variant in enumerate(VARIANTS, 1):
        logger.info("step_02_%s: evaluating %s", index, variant["variant"])
        set_seed(settings.seed)
        args = replace(settings, pkd_groups=variant["pkd_groups"])
        rows, subject_rows, blocks = evaluate_variant(data, variant, args)
        dimensions[variant["variant"]] = sum(blocks.values())
        for row in rows:
            row.update({"study": "architecture_selection", "pkd_groups": variant["pkd_groups"]})
        for row in subject_rows:
            row.update({"study": "architecture_selection", "pkd_groups": variant["pkd_groups"]})
        summary_rows.extend(rows)
        subject_frames.append(pd.DataFrame(subject_rows))
    subject_table = pd.concat(subject_frames, ignore_index=True)
    comparison_rows = []
    for variant in VARIANTS:
        candidate = subject_table.loc[subject_table["variant"].eq(variant["variant"]), ["condition", "subject", "subject_window_accuracy"]]
        for row in compare_to_reference(reference, candidate, settings):
            row.update({"variant": variant["variant"], "feature_dim": dimensions[variant["variant"]]})
            comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    selected = choose_variant(comparison, dimensions, settings.noninferiority_margin_pct)
    summary_path = settings.output_dir / "architecture_candidate_summary.csv"
    subject_path = settings.output_dir / "architecture_candidate_subject_accuracy.csv"
    comparison_path = settings.output_dir / "architecture_noninferiority.csv"
    write_csv(summary_path, summary_rows)
    write_csv(subject_path, subject_table.to_dict("records"))
    comparison.to_csv(comparison_path, index=False)
    decision = {"selected_variant": selected, "feature_dimensions": dimensions, "noninferiority_margin_pct": settings.noninferiority_margin_pct, "primary_scopes": ["trunk_sway_large", "overall_participant_mean"]}
    (settings.output_dir / "architecture_selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    logger.info("step_03: selected=%s dimensions=%s", selected, dimensions[selected])
    return summary_path, subject_path, comparison_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run(Settings(root / "gait_id_pelvis_62.h5", root / "baselines_out/prism_subject_window_accuracy/prism_subject_condition_accuracy.csv", root / "baselines_out/prism_subject_window_accuracy/prism_condition_summary.csv", root / "baselines_out/prism_architecture_selection", 42, 0.4, 48, 6, (1, 2, 4, 8), 256, 10000, 1.0))
