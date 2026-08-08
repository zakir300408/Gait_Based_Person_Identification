"""Evaluate fixed PRISM-Gait across 1–5 s trunk-sway windows and latency."""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
from csv import DictReader
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from analysis.prism_gait.ablation import (
    load_original_h5_windowed,
    set_seed,
    stack,
    subject_condition_rows,
    trial_metrics,
    write_csv,
)
from analysis.prism_gait.verification import make_classifier
from analysis.prism_gait.hcgf import featurize as hcgf_featurize
from analysis.prism_gait.cdkc import CDKCFeatures
from analysis.prism_gait.temporal_features import fit_temporal_features, transform_temporal_features


TRUNK_CONDITIONS = ("trunk_sway_small", "trunk_sway_large")
KERNELS = {"raw": 3000, "diff": 3000, "diff2": 2000}


@dataclass(frozen=True)
class Settings:
    h5_path: Path
    architecture_selection_path: Path
    output_dir: Path
    seed: int
    acc_std_min: float
    pkd_groups: int
    pkd_kernels_per_group: int
    pkd_dilations: tuple[int, ...]
    batch_size: int
    durations_seconds: tuple[float, ...]
    latency_windows: int
    latency_repeats: int
    latency_warmups: int


def create_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("prism_duration_selection")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    path = output_dir / f"prism_duration_selection_{datetime.now():%Y%m%d_%H%M%S}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(DictReader(stream))


def fit_model(data: dict, settings: Settings):
    started = perf_counter()
    x_train = stack(data["x_train"])
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    x_test = {name: stack(data["x_test_by_cond"][name]) for name in TRUNK_CONDITIONS}
    core, train_core = fit_temporal_features(x_train, y_train, KERNELS, settings.seed)
    test_blocks = {name: [transform_temporal_features(core, x).astype(np.float32)] for name, x in x_test.items()}
    train_blocks = [train_core.astype(np.float32)]
    cdkc = CDKCFeatures(
        groups=settings.pkd_groups,
        kernels_per_group=settings.pkd_kernels_per_group,
        kernel_size=9,
        dilations=settings.pkd_dilations,
        seed=settings.seed,
        batch_size=settings.batch_size,
    ).fit(x_train.astype(np.float32))
    train_cdkc = cdkc.transform(x_train.astype(np.float32))
    train_blocks.append(train_cdkc)
    for name, x in x_test.items():
        test_blocks[name].append(cdkc.transform(x.astype(np.float32)))
    train_hcgf = hcgf_featurize([window.astype(np.float32) for window in x_train])
    train_blocks.append(train_hcgf)
    for name, x in x_test.items():
        test_blocks[name].append(hcgf_featurize([window.astype(np.float32) for window in x]))
    train_features = np.hstack(train_blocks).astype(np.float32)
    test_features = {name: np.hstack(blocks).astype(np.float32) for name, blocks in test_blocks.items()}
    classifier = make_classifier(cosine=True).fit(train_features, y_train)
    dimensions = {"core": int(train_core.shape[1]), "cdkc": int(train_cdkc.shape[1]),
                  "hcgf": int(train_hcgf.shape[1]), "total": int(train_features.shape[1])}
    return core, cdkc, classifier, x_test, test_features, dimensions, perf_counter() - started


def predict_raw(core, cdkc, classifier, windows: np.ndarray) -> np.ndarray:
    blocks = [transform_temporal_features(core, windows).astype(np.float32)]
    blocks.append(cdkc.transform(windows.astype(np.float32)))
    blocks.append(hcgf_featurize([window.astype(np.float32) for window in windows]))
    return classifier.predict(np.hstack(blocks).astype(np.float32))


def fixed_latency_windows(x_test: dict[str, np.ndarray], count: int) -> np.ndarray:
    per_condition = count // len(TRUNK_CONDITIONS)
    selected = []
    for condition in TRUNK_CONDITIONS:
        windows = x_test[condition]
        if len(windows) < per_condition:
            raise ValueError(f"not enough {condition} windows for latency benchmark")
        indices = np.linspace(0, len(windows) - 1, per_condition, dtype=int)
        selected.append(windows[indices])
    return np.concatenate(selected, axis=0)


def measure_latency(core, cdkc, classifier, windows: np.ndarray, settings: Settings) -> dict:
    for _ in range(settings.latency_warmups):
        predict_raw(core, cdkc, classifier, windows[:1])
    batch_seconds = []
    for _ in range(settings.latency_repeats):
        started = perf_counter()
        predict_raw(core, cdkc, classifier, windows)
        batch_seconds.append(perf_counter() - started)
    single_ms = []
    for window in windows:
        started = perf_counter()
        predict_raw(core, cdkc, classifier, window[None, ...])
        single_ms.append(1000 * (perf_counter() - started))
    duration_ms = 1000 * float(windows.shape[2] / 100)
    batch_ms = 1000 * float(np.median(batch_seconds)) / len(windows)
    median_ms = float(np.median(single_ms))
    p95_ms = float(np.percentile(single_ms, 95))
    return {
        "latency_windows": int(len(windows)),
        "batch_repeats": settings.latency_repeats,
        "batch_ms_per_window": batch_ms,
        "single_window_mean_ms": float(np.mean(single_ms)),
        "single_window_median_ms": median_ms,
        "single_window_p95_ms": p95_ms,
        "first_id_median_ms": duration_ms + median_ms,
        "update_interval_ms": duration_ms / 2,
    }


def performance_rows(data: dict, test_features: dict, classifier, duration: float, fit_seconds: float):
    rows, subjects = [], []
    for condition in TRUNK_CONDITIONS:
        y = np.asarray(data["y_test_by_cond"][condition], dtype=np.int64)
        prediction = classifier.predict(test_features[condition])
        row = {"study": "fixed_prism_duration", "variant": "PRISM-Gait",
               "win_s": duration, "condition": condition,
               "window_accuracy": float(accuracy_score(y, prediction)),
               "window_macro_f1": float(f1_score(y, prediction, average="macro")),
               "n_windows": int(len(y)), "fit_seconds": fit_seconds}
        row.update(trial_metrics(y, prediction, data["meta_test_by_cond"][condition]))
        rows.append(row)
        subjects.extend(subject_condition_rows("fixed_prism_duration", "PRISM-Gait", duration,
                        condition, y, prediction, data["meta_test_by_cond"][condition]))
    return rows, subjects


def run(settings: Settings) -> tuple[Path, Path, Path]:
    for path in (settings.h5_path, settings.architecture_selection_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path.name}")
    selection = json.loads(settings.architecture_selection_path.read_text(encoding="utf-8"))
    if selection.get("selected_variant") != "PRISM-Gait":
        raise ValueError("duration analysis requires selected full PRISM-Gait")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(settings.output_dir)
    summary_path = settings.output_dir / "prism_duration_condition_summary.csv"
    subject_path = settings.output_dir / "prism_duration_subject_accuracy.csv"
    latency_path = settings.output_dir / "prism_duration_latency.csv"
    summaries = read_rows(summary_path)
    subjects = read_rows(subject_path)
    latencies = read_rows(latency_path)
    dimensions = {
        str(row["win_s"]): {name: int(float(row[name])) for name in ("core", "cdkc", "hcgf", "total")}
        for row in latencies
    }
    completed = {float(row["win_s"]) for row in latencies}
    summaries = [row for row in summaries if float(row["win_s"]) in completed]
    subjects = [row for row in subjects if float(row["win_s"]) in completed]
    for duration in settings.durations_seconds:
        if duration in completed:
            logger.info("step_duration_%ss: already complete; skipping", duration)
            continue
        set_seed(settings.seed)
        logger.info("step_duration_%ss: loading and fitting", duration)
        print(f"step_duration_{duration:g}s: loading and fitting", flush=True)
        data = load_original_h5_windowed(settings.h5_path, duration, settings.acc_std_min)
        core, cdkc, clf, x_test, test_features, dims, fit_seconds = fit_model(data, settings)
        rows, people = performance_rows(data, test_features, clf, duration, fit_seconds)
        latency = measure_latency(core, cdkc, clf, fixed_latency_windows(x_test, settings.latency_windows), settings)
        latency.update({"win_s": duration, "fit_seconds": fit_seconds, **dims})
        summaries.extend(rows); subjects.extend(people); latencies.append(latency); dimensions[str(duration)] = dims
        write_csv(summary_path, summaries)
        write_csv(subject_path, subjects)
        write_csv(latency_path, latencies)
        logger.info("duration=%s large_sway=%.4f median_ms=%.3f", duration,
                    next(row["window_accuracy"] for row in rows if row["condition"] == "trunk_sway_large"),
                    latency["single_window_median_ms"])
        del data, core, cdkc, clf, x_test, test_features
        gc.collect()
    metadata = {"selected_architecture": selection, "durations_seconds": list(settings.durations_seconds),
                "feature_dimensions": dimensions, "latency_definition": "normalized window to identity prediction",
                "first_id_definition": "window acquisition duration plus median single-window compute latency",
                "overlap": "50% window overlap", "platform": platform.platform(),
                "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                "python": platform.python_version(), "numpy": version("numpy"),
                "scikit_learn": version("scikit-learn"), "sktime": version("sktime")}
    (settings.output_dir / "prism_duration_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary_path, subject_path, latency_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(*run(Settings(root / "data/altergait/gait_id_pelvis_62.h5",
          root / "baselines_out/prism_architecture_selection/architecture_selection.json",
          root / "baselines_out/prism_duration_trunk_sway", 42, 0.4, 48, 6, (1, 2, 4, 8),
          256, (1.0, 2.0, 3.0, 4.0, 5.0), 64, 5, 3)), sep="\n")
