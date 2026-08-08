"""Export paired majority-vote decisions for the sway benchmark.

The two supported stages use their original evaluation environments:

* ``rocket``: the ``gaitid`` environment (sktime).
* ``prism``: the ``exoskeleton`` environment (PyTorch feature branch).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
H5_PATH = ROOT / "gait_id_pelvis_62.h5"
OUTPUT_DIR = ROOT / "baselines_out" / "treadmill_published_accuracy_tables"
CONDITIONS = ("trunk_sway_small", "trunk_sway_large")


def _trial_rows(
    model: str,
    condition: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    metadata: list[dict],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    truths: dict[tuple[str, str], int] = {}
    for target, estimate, meta in zip(truth, prediction, metadata):
        key = (str(meta["subject"]), str(meta["trial"]))
        grouped[key].append(int(estimate))
        truths[key] = int(target)
    rows = []
    for (subject, trial), estimates in sorted(grouped.items()):
        vote = Counter(estimates).most_common(1)[0][0]
        target = truths[(subject, trial)]
        rows.append(
            {
                "model": model,
                "condition": condition,
                "subject": subject,
                "trial": trial,
                "truth": target,
                "prediction": vote,
                "correct": int(target == vote),
                "n_windows": len(estimates),
            }
        )
    return rows


def export_rocket() -> Path:
    from ablation.pelvis_harmonic_covariance_model import load_original_h5
    from ablation.run_strict_published_rocket_benchmarks import (
        MODEL_SPECS,
        build_classifier,
        stack,
    )

    data = load_original_h5(H5_PATH)
    x_train = stack(data["x_train"])
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    rows: list[dict[str, object]] = []
    for model in MODEL_SPECS:
        print(f"[{model}] fitting exact published pipeline", flush=True)
        classifier = build_classifier(model, seed=42)
        # Serial execution avoids a Windows OpenMP/joblib crash in the current
        # environment; it does not alter the deterministic fitted pipeline.
        classifier.set_params(n_jobs=1)
        classifier.fit(x_train, y_train)
        for condition in CONDITIONS:
            prediction = classifier.predict(stack(data["x_test_by_cond"][condition]))
            rows.extend(
                _trial_rows(
                    model,
                    condition,
                    np.asarray(data["y_test_by_cond"][condition], dtype=np.int64),
                    prediction,
                    data["meta_test_by_cond"][condition],
                )
            )
            print(f"[{model}] {condition} exported", flush=True)
    output = pd.DataFrame(rows)
    path = OUTPUT_DIR / "sway_trial_decisions_rocket.csv"
    _validate(output, {"MiniRocket", "MultiRocket"})
    output.to_csv(path, index=False)
    return path


def export_prism() -> Path:
    from ablation.mblite_ablation_study import (
        COMPONENT_VARIANTS,
        add_phcf_features,
        add_pkd_features,
        fit_branch_features,
        load_original_h5_windowed,
        make_classifier,
        set_seed,
        stack,
    )

    set_seed(42)
    data = load_original_h5_windowed(H5_PATH, 2.0, 0.4)
    data["x_test_by_cond"] = {
        condition: data["x_test_by_cond"][condition] for condition in CONDITIONS
    }
    data["y_test_by_cond"] = {
        condition: data["y_test_by_cond"][condition] for condition in CONDITIONS
    }
    data["meta_test_by_cond"] = {
        condition: data["meta_test_by_cond"][condition] for condition in CONDITIONS
    }
    config = next(item for item in COMPONENT_VARIANTS if item["variant"] == "MB-Lite")
    settings = SimpleNamespace(
        seed=42,
        pkd_groups=48,
        pkd_kernels_per_group=6,
        pkd_dilations=(1, 2, 4, 8),
        batch_size=256,
    )
    x_train_raw = stack(data["x_train"])
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    x_test_raw = {
        condition: stack(data["x_test_by_cond"][condition]) for condition in CONDITIONS
    }
    print("[PRISM-Gait] extracting exact full representation", flush=True)
    x_train, x_test, _ = fit_branch_features(
        x_train_raw,
        y_train,
        x_test_raw,
        config["branches"],
        settings.seed,
    )
    x_train, x_test, _ = add_pkd_features(
        x_train,
        x_test,
        x_train_raw,
        x_test_raw,
        settings.seed,
        settings,
    )
    x_train, x_test, _ = add_phcf_features(x_train, x_test, data)
    classifier = make_classifier()
    classifier.fit(x_train, y_train)
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        prediction = classifier.predict(x_test[condition])
        rows.extend(
            _trial_rows(
                "PRISM-Gait",
                condition,
                np.asarray(data["y_test_by_cond"][condition], dtype=np.int64),
                prediction,
                data["meta_test_by_cond"][condition],
            )
        )
        print(f"[PRISM-Gait] {condition} exported", flush=True)
    output = pd.DataFrame(rows)
    path = OUTPUT_DIR / "sway_trial_decisions_prism.csv"
    _validate(output, {"PRISM-Gait"})
    output.to_csv(path, index=False)
    return path


def _validate(frame: pd.DataFrame, models: set[str]) -> None:
    expected_rows = 120 * len(models)
    if len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} paired decisions, found {len(frame)}")
    if set(frame["model"]) != models or set(frame["condition"]) != set(CONDITIONS):
        raise ValueError("decision export has an unexpected model or condition roster")
    if frame.duplicated(["model", "condition", "subject", "trial"]).any():
        raise ValueError("decision export contains duplicate trials")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("rocket", "prism"))
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(export_rocket() if args.stage == "rocket" else export_prism())


if __name__ == "__main__":
    main()
