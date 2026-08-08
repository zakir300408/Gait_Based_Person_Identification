"""Benchmark unmodified published MiniRocket and MultiRocket pipelines."""

from __future__ import annotations

import json
import gc
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import sktime
from sklearn.metrics import accuracy_score, f1_score
from sktime.classification.kernel_based import RocketClassifier

from ablation.pelvis_harmonic_covariance_model import load_original_h5

MODEL_SPECS = {
    "MiniRocket": {"rocket_transform": "minirocket", "num_kernels": 10_000},
    "MultiRocket": {"rocket_transform": "multirocket", "num_kernels": 6_250},
}
WHUGAIT_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")


@dataclass(frozen=True)
class Settings:
    h5_path: Path
    whugait_root: Path
    output_dir: Path
    seed: int = 42


def build_classifier(model: str, seed: int) -> RocketClassifier:
    spec = MODEL_SPECS[model]
    return RocketClassifier(
        num_kernels=spec["num_kernels"],
        rocket_transform=spec["rocket_transform"],
        use_multivariate="yes",
        n_jobs=-1,
        random_state=seed,
    )


def stack(windows: list[np.ndarray]) -> np.ndarray:
    return np.ascontiguousarray(np.stack(windows), dtype=np.float64)


def load_whugait_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    signals = root / split / "Inertial Signals"
    arrays = [np.loadtxt(signals / f"{split}_{channel}.txt", dtype=np.float32) for channel in WHUGAIT_CHANNELS]
    x = np.stack(arrays, axis=1).astype(np.float32)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    return x, y


def trial_metrics(labels: np.ndarray, predictions: np.ndarray, metadata: list[dict]) -> dict:
    grouped, truths = defaultdict(list), {}
    for truth, prediction, meta in zip(labels, predictions, metadata):
        key = f"{meta['subject']}|{meta['condition']}|{meta['trial']}"
        grouped[key].append(int(prediction))
        truths[key] = int(truth)
    predicted = [Counter(values).most_common(1)[0][0] for values in grouped.values()]
    actual = list(truths.values())
    return {
        "trial_accuracy": float(accuracy_score(actual, predicted)),
        "trial_macro_f1": float(f1_score(actual, predicted, average="macro")),
        "n_trials": len(actual),
    }


def run_treadmill(settings: Settings) -> tuple[list[dict], list[dict]]:
    data = load_original_h5(settings.h5_path)
    x_train = stack(data["x_train"])
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    condition_path = settings.output_dir / "treadmill_condition_results.csv"
    subject_path = settings.output_dir / "treadmill_subject_condition_accuracy.csv"
    condition_rows = pd.read_csv(condition_path).to_dict("records") if condition_path.is_file() else []
    subject_rows = pd.read_csv(subject_path).to_dict("records") if subject_path.is_file() else []
    for model_name in MODEL_SPECS:
        if sum(row["model"] == model_name for row in condition_rows) == 9:
            print(f"[Treadmill] Reusing complete {model_name} result", flush=True)
            continue
        condition_rows = [row for row in condition_rows if row["model"] != model_name]
        subject_rows = [row for row in subject_rows if row["model"] != model_name]
        print(f"[Treadmill] Fitting published {model_name}", flush=True)
        classifier = build_classifier(model_name, settings.seed)
        started = perf_counter()
        classifier.fit(x_train, y_train)
        elapsed = perf_counter() - started
        for condition, windows in sorted(data["x_test_by_cond"].items()):
            labels = np.asarray(data["y_test_by_cond"][condition], dtype=np.int64)
            metadata = data["meta_test_by_cond"][condition]
            predictions = classifier.predict(stack(windows))
            row = {
                "model": model_name,
                "condition": condition,
                "window_accuracy": float(accuracy_score(labels, predictions)),
                "window_macro_f1": float(f1_score(labels, predictions, average="macro")),
                "n_windows": len(labels),
                "fit_seconds": elapsed,
                "source": "sktime RocketClassifier published default pipeline",
            }
            row.update(trial_metrics(labels, predictions, metadata))
            condition_rows.append(row)
            grouped = defaultdict(list)
            for truth, prediction, meta in zip(labels, predictions, metadata):
                grouped[meta["subject"]].append(int(truth == prediction))
            subject_rows.extend(
                {
                    "model": model_name,
                    "condition": condition,
                    "subject": subject,
                    "subject_window_accuracy": float(np.mean(correct)),
                    "n_windows": len(correct),
                }
                for subject, correct in sorted(grouped.items())
            )
        pd.DataFrame(condition_rows).to_csv(condition_path, index=False)
        pd.DataFrame(subject_rows).to_csv(subject_path, index=False)
        del classifier
        gc.collect()
        print(f"[Treadmill] Saved {model_name}", flush=True)
    return condition_rows, subject_rows


def run_whugait(settings: Settings, requested_models: tuple[str, ...]) -> list[dict]:
    x_train, y_train = load_whugait_split(settings.whugait_root, "train")
    x_test, y_test = load_whugait_split(settings.whugait_root, "test")
    result_path = settings.output_dir / "whugait_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.is_file() else []
    for model_name in requested_models:
        if any(row["model"] == model_name for row in rows):
            print(f"[WHU-Gait] Reusing complete {model_name} result", flush=True)
            continue
        print(f"[WHU-Gait] Fitting published {model_name}", flush=True)
        classifier = build_classifier(model_name, settings.seed)
        started = perf_counter()
        classifier.fit(np.ascontiguousarray(x_train, dtype=np.float64), y_train)
        predictions = classifier.predict(np.ascontiguousarray(x_test, dtype=np.float64))
        rows.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "macro_f1": float(f1_score(y_test, predictions, average="macro")),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "fit_predict_seconds": perf_counter() - started,
                "source": "sktime RocketClassifier published default pipeline",
            }
        )
        pd.DataFrame(rows).to_csv(result_path, index=False)
        del classifier
        gc.collect()
        print(f"[WHU-Gait] Saved {model_name}", flush=True)
    return rows


def validate_outputs(condition_rows: list[dict], subject_rows: list[dict], whugait_rows: list[dict]) -> None:
    if len(condition_rows) != 18 or len(subject_rows) != 1_076:
        raise RuntimeError("incomplete treadmill MiniRocket/MultiRocket output")
    if len(whugait_rows) != 2 or {row["model"] for row in whugait_rows} != set(MODEL_SPECS):
        raise RuntimeError("incomplete WHU-Gait MiniRocket/MultiRocket output")


def run(settings: Settings, stage: str = "all") -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    condition_path = settings.output_dir / "treadmill_condition_results.csv"
    subject_path = settings.output_dir / "treadmill_subject_condition_accuracy.csv"
    whugait_path = settings.output_dir / "whugait_results.csv"
    if stage in {"all", "treadmill"}:
        condition_rows, subject_rows = run_treadmill(settings)
    else:
        condition_rows = pd.read_csv(condition_path).to_dict("records") if condition_path.is_file() else []
        subject_rows = pd.read_csv(subject_path).to_dict("records") if subject_path.is_file() else []
    requested = {
        "all": tuple(MODEL_SPECS),
        "whugait": tuple(MODEL_SPECS),
        "whugait_minirocket": ("MiniRocket",),
        "whugait_multirocket": ("MultiRocket",),
    }.get(stage, ())
    whugait_rows = run_whugait(settings, requested) if requested else (
        pd.read_csv(whugait_path).to_dict("records") if whugait_path.is_file() else []
    )
    if not (len(condition_rows) == 18 and len(subject_rows) == 1_076 and len(whugait_rows) == 2):
        print(f"[Partial] treadmill={len(condition_rows)} subjects={len(subject_rows)} whugait={len(whugait_rows)}", flush=True)
        return
    validate_outputs(condition_rows, subject_rows, whugait_rows)
    payload = {
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()},
        "environment": {"sktime": sktime.__version__, "sklearn": sklearn.__version__},
        "pipelines": {
            "MiniRocket": "MiniRocketMultivariate(10000 kernels) -> StandardScaler(with_mean=False) -> RidgeClassifierCV(logspace[-3,3], 10)",
            "MultiRocket": "MultiRocketMultivariate(6250 requested; 6216 effective kernels, 49728 features) -> StandardScaler(with_mean=False) -> RidgeClassifierCV(logspace[-3,3], 10)",
        },
        "excluded_local_changes": ["cosine feature normalization", "reduced ridge alpha grid", "10000-kernel MultiRocket"],
        "treadmill": condition_rows,
        "whugait": whugait_rows,
    }
    (settings.output_dir / "benchmark_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run(
        Settings(
            root / "gait_id_pelvis_62.h5",
            root / "public_datasets" / "whugait" / "Dataset #3" / "Dataset #3",
            root / "baselines_out" / "published_rocket_strict",
        ),
        os.environ.get("ROCKET_BENCHMARK_STAGE", "all"),
    )


if __name__ == "__main__":
    main()
