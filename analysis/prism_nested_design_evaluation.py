from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
@dataclass(frozen=True)
class Settings:
    duration_path: Path
    component_path: Path
    hcgf_path: Path
    full_path: Path
    benchmark_path: Path
    output_dir: Path
    folds: int
    seed: int
    bootstrap_repetitions: int
    selection_condition: str
    expected_duration: float
    expected_architecture: str
    architecture_dimensions: dict[str, int]
    models: tuple[str, ...]
def create_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("prism_nested_design_evaluation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    path = output_dir / f"prism_nested_design_{datetime.now():%Y%m%d_%H%M%S}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def select_best(values: pd.Series, complexity: dict | None = None):
    maximum = values.max()
    tied = list(values.index[np.isclose(values.to_numpy(float), maximum)])
    if complexity is None:
        return min(tied)
    return min(tied, key=lambda name: (complexity[name], name))
def load_architectures(settings: Settings) -> pd.DataFrame:
    component = pd.read_csv(settings.component_path)
    component = component.loc[component["variant"].isin(["Lite core", "Core+PKD"])]
    hcgf = pd.read_csv(settings.hcgf_path)
    hcgf = hcgf.loc[hcgf["variant"].eq("Core+PHCF")]
    full = pd.read_csv(settings.full_path).assign(variant="PRISM-Gait")
    return pd.concat([component, hcgf, full], ignore_index=True)
def validate_inputs(
    duration: pd.DataFrame,
    architecture: pd.DataFrame,
    benchmark: pd.DataFrame,
    settings: Settings,
) -> np.ndarray:
    expected_architectures = set(settings.architecture_dimensions)
    if set(architecture["variant"].unique()) != expected_architectures:
        raise ValueError("architecture candidates do not match the declared set")
    if set(settings.models) - set(benchmark.columns):
        raise ValueError("benchmark is missing one or more declared models")
    subjects = np.asarray(sorted(benchmark["participant_id"].unique()))
    if len(subjects) != 61:
        raise ValueError("expected 61 participants with at least one sway trial")
    large = duration.loc[duration["condition"].eq(settings.selection_condition)]
    if large.groupby("win_s")["subject"].nunique().nunique() != 1:
        raise ValueError("duration candidates have unequal participant coverage")
    large = architecture.loc[architecture["condition"].eq(settings.selection_condition)]
    if large.groupby("variant")["subject"].nunique().nunique() != 1:
        raise ValueError("architecture candidates have unequal participant coverage")
    if benchmark.loc[:, settings.models].isna().any().any():
        raise ValueError("benchmark contains missing model accuracy")
    return subjects
def nested_rows(duration: pd.DataFrame, architecture: pd.DataFrame, benchmark: pd.DataFrame,
                subjects: np.ndarray, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments, choices, held_out = [], [], []
    splitter = KFold(settings.folds, shuffle=True, random_state=settings.seed)
    for fold, (development_index, evaluation_index) in enumerate(splitter.split(subjects), 1):
        development = set(subjects[development_index])
        evaluation = set(subjects[evaluation_index])
        if development & evaluation or development | evaluation != set(subjects):
            raise ValueError(f"invalid participant partition in fold {fold}")
        assignments.extend(
            {"fold": fold, "participant": subject, "role": role}
            for role, group in (("development", development), ("evaluation", evaluation))
            for subject in sorted(group)
        )
        duration_development = duration.loc[
            duration["subject"].isin(development)
            & duration["condition"].eq(settings.selection_condition)
        ]
        duration_means = duration_development.groupby("win_s")["subject_window_accuracy"].mean()
        selected_duration = float(select_best(duration_means))
        architecture_development = architecture.loc[
            architecture["subject"].isin(development)
            & architecture["condition"].eq(settings.selection_condition)
        ]
        architecture_means = architecture_development.groupby("variant")["subject_window_accuracy"].mean()
        selected_architecture = select_best(architecture_means, settings.architecture_dimensions)
        for candidate, value in duration_means.items():
            choices.append({"fold": fold, "choice": "duration_seconds", "candidate": candidate,
                            "outer_development_participants": len(development),
                            "selection_participants": duration_development["subject"].nunique(), "mean_accuracy": value,
                            "selected": float(candidate) == selected_duration})
        for candidate, value in architecture_means.items():
            choices.append({"fold": fold, "choice": "feature_composition", "candidate": candidate,
                            "outer_development_participants": len(development),
                            "selection_participants": architecture_development["subject"].nunique(), "mean_accuracy": value,
                            "selected": candidate == selected_architecture})
        if (selected_duration, selected_architecture) != (
            settings.expected_duration,
            settings.expected_architecture,
        ):
            raise ValueError(f"fold {fold} selected an unavailable benchmark design")
        fold_rows = benchmark.loc[benchmark["participant_id"].isin(evaluation)].copy()
        fold_rows["fold"] = fold
        fold_rows["selected_duration_seconds"] = selected_duration
        fold_rows["selected_architecture"] = selected_architecture
        held_out.append(fold_rows)
    return pd.DataFrame(assignments), pd.DataFrame(choices), pd.concat(held_out, ignore_index=True)
def summarize_oof(oof: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(settings.seed)
    summaries, comparisons, folds = [], [], []
    for (fold, condition), group in oof.groupby(["fold", "condition_code"], sort=True):
        weights = group["n_windows"].to_numpy(float)
        for model in settings.models:
            folds.append({"fold": fold, "condition": condition, "model": model, "participants": len(group),
                          "windows": int(weights.sum()), "window_accuracy_pct": np.average(group[model], weights=weights)})
    for condition, group in oof.groupby("condition_code", sort=True):
        weights = group["n_windows"].to_numpy(float)
        for model in settings.models:
            values = group[model].to_numpy(float)
            summaries.append({"condition": condition, "model": model, "participants": len(group),
                              "windows": int(weights.sum()), "window_accuracy_pct": np.average(values, weights=weights)})
        prism = group[settings.expected_architecture].to_numpy(float)
        indices = rng.integers(0, len(group), size=(settings.bootstrap_repetitions, len(group)))
        for comparator in settings.models[1:]:
            other = group[comparator].to_numpy(float)
            delta = prism - other
            draws = np.sum(delta[indices] * weights[indices], axis=1) / np.sum(weights[indices], axis=1)
            test = wilcoxon(prism, other, zero_method="pratt", alternative="two-sided")
            comparisons.append({"condition": condition, "comparator": comparator, "participants": len(group),
                                "weighted_delta_pp": np.average(delta, weights=weights),
                                "ci_lower_pp": np.quantile(draws, 0.025), "ci_upper_pp": np.quantile(draws, 0.975),
                                "p_raw": test.pvalue})
    comparison = pd.DataFrame(comparisons)
    comparison["p_holm"] = np.nan
    for _, indices in comparison.groupby("condition").groups.items():
        comparison.loc[indices, "p_holm"] = multipletests(comparison.loc[indices, "p_raw"], method="holm")[1]
    return pd.DataFrame(folds), pd.DataFrame(summaries), comparison
def run(settings: Settings) -> Path:
    inputs = (settings.duration_path, settings.component_path, settings.hcgf_path,
              settings.full_path, settings.benchmark_path)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path.name}")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(settings.output_dir)
    logger.info("step_01: loading participant-level candidate and benchmark results")
    duration = pd.read_csv(settings.duration_path)
    architecture = load_architectures(settings)
    benchmark = pd.read_csv(settings.benchmark_path)
    benchmark = benchmark.loc[benchmark["condition_code"].isin(["trunk_sway_small", "trunk_sway_large"])]
    subjects = validate_inputs(duration, architecture, benchmark, settings)
    logger.info("step_02: constructing %s participant folds with seed=%s", settings.folds, settings.seed)
    assignments, choices, oof = nested_rows(duration, architecture, benchmark, subjects, settings)
    folds, summary, comparisons = summarize_oof(oof, settings)
    outputs = {"outer_fold_assignments.csv": assignments, "inner_selection_metrics.csv": choices,
               "nested_oof_participant_accuracy.csv": oof, "nested_fold_metrics.csv": folds, "nested_model_summary.csv": summary,
               "nested_prism_comparisons.csv": comparisons}
    for name, frame in outputs.items():
        frame.to_csv(settings.output_dir / name, index=False)
    selected = choices.loc[choices["selected"]].pivot(index="fold", columns="choice", values="candidate")
    metadata = {"protocol": "five-fold participant-nested design evaluation", "fold_seed": settings.seed,
                "selection_condition": settings.selection_condition, "duration_rule": "highest development-participant mean; shorter duration breaks exact ties",
                "architecture_rule": "highest development-participant mean; lower dimension breaks exact ties",
                "evaluation_rule": "outer participants contribute no sway result to design selection",
                "enrollment_rule": "baseline trials from all identities remain eligible for identity enrollment",
                "selected_by_fold": selected.reset_index().to_dict("records"),
                "input_sha256": {path.name: file_sha256(path) for path in inputs}}
    metadata_path = settings.output_dir / "nested_design_protocol.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("step_03: verified outputs=%s", list(outputs) + [metadata_path.name])
    return metadata_path
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(run(Settings(
        root / "baselines_out/prism_duration_trunk_sway/prism_duration_subject_accuracy.csv", root / "baselines_out/prism_gait_ablation_final/subject_condition_accuracy.csv",
        root / "baselines_out/prism_architecture_selection/architecture_candidate_subject_accuracy.csv", root / "baselines_out/prism_subject_window_accuracy/prism_subject_condition_accuracy.csv",
        root / "baselines_out/treadmill_published_plot_data/participant_window_accuracy_wide.csv",
        root / "baselines_out/prism_nested_design_evaluation", 5, 42, 10_000, "trunk_sway_large", 2.0, "PRISM-Gait",
        {"Lite core": 7812, "Core+PKD": 12420, "Core+PHCF": 8441, "PRISM-Gait": 13049}, ("PRISM-Gait", "MultiRocket", "MiniRocket", "FCN-BiLSTM", "InceptionTime", "Multi-Model LSTM"),
    )))
