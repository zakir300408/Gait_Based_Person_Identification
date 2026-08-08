from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Settings:
    assignments_path: Path
    trial_paths: tuple[Path, ...]
    output_dir: Path
    conditions: tuple[str, ...]
    models: tuple[str, ...]
    model_rename: dict[str, str]


def create_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("prism_nested_trial_evaluation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    path = output_dir / f"prism_nested_trials_{datetime.now():%Y%m%d_%H%M%S}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def run(settings: Settings) -> tuple[Path, Path]:
    for path in (settings.assignments_path, *settings.trial_paths):
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path.name}")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(settings.output_dir)
    logger.info("step_01: loading outer evaluation assignments")
    assignments = pd.read_csv(settings.assignments_path)
    assignments = assignments.loc[assignments["role"].eq("evaluation"), ["fold", "participant"]]
    if assignments["participant"].duplicated().any() or len(assignments) != 61:
        raise ValueError("each sway participant must have one outer evaluation fold")
    logger.info("step_02: loading trial decisions")
    frames = []
    for path in settings.trial_paths:
        frame = pd.read_csv(path)
        frame["model"] = frame["model"].replace(settings.model_rename)
        frames.append(frame.loc[frame["condition"].isin(settings.conditions)])
    trials = pd.concat(frames, ignore_index=True)
    trials = trials.loc[trials["model"].isin(settings.models)].copy()
    expected = len(settings.models) * 120
    if len(trials) != expected or trials.duplicated(["model", "condition", "subject", "trial"]).any():
        raise ValueError("trial decisions are incomplete or duplicated")
    trials = trials.merge(assignments, left_on="subject", right_on="participant", validate="many_to_one")
    if len(trials) != expected:
        raise ValueError("one or more trial decisions lack an outer fold")
    fold = trials.groupby(["fold", "condition", "model"], as_index=False).agg(
        participants=("subject", "nunique"), correct_trials=("correct", "sum"), trials=("correct", "size")
    )
    fold["trial_accuracy_pct"] = 100 * fold["correct_trials"] / fold["trials"]
    summary = trials.groupby(["condition", "model"], as_index=False).agg(
        participants=("subject", "nunique"), correct_trials=("correct", "sum"), trials=("correct", "size")
    )
    summary["trial_accuracy_pct"] = 100 * summary["correct_trials"] / summary["trials"]
    fold_path = settings.output_dir / "nested_trial_fold_metrics.csv"
    summary_path = settings.output_dir / "nested_trial_summary.csv"
    trials.to_csv(settings.output_dir / "nested_oof_trial_decisions.csv", index=False)
    fold.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    logger.info("step_03: verified decisions=%s", len(trials))
    return fold_path, summary_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    table = root / "baselines_out/treadmill_published_accuracy_tables"
    published = root / "baselines_out/treadmill_published_subject_sd"
    print(*run(Settings(
        root / "baselines_out/prism_nested_design_evaluation/outer_fold_assignments.csv",
        (table / "sway_trial_decisions_prism.csv", table / "sway_trial_decisions_rocket.csv",
         published / "fcn_bilstm_official/official_trial_predictions.csv",
         published / "inceptiontime_author_ensemble/inceptiontime_ensemble_trial_predictions.csv",
         published / "multi_model_lstm_official/official_trial_predictions.csv"),
        root / "baselines_out/prism_nested_design_evaluation",
        ("trunk_sway_small", "trunk_sway_large"),
        ("PRISM-Gait", "MultiRocket", "MiniRocket", "FCN-BiLSTM", "InceptionTime", "Multi-Model LSTM"),
        {"fcn_bilstm_official": "FCN-BiLSTM", "multi_model_lstm_official": "Multi-Model LSTM"},
    )), sep="\n")
