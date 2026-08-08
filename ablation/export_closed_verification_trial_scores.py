"""Export trial-level claim scores for enrolled-participant verification.

The existing verification suites retain complete trial scores for the
subject-held-out open-set protocol but only summary metrics for the enrolled
protocol.  This script fills that narrow gap without changing either benchmark.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import baseline_verification_suite as baseline
import mblite_verification_suite as prism


def aggregate_trial_claims(
    model: str,
    condition: str,
    scores: np.ndarray,
    labels: np.ndarray,
    truth: np.ndarray,
    metadata: list[dict],
) -> list[dict]:
    """Average window score vectors within trials and emit every identity claim."""
    by_trial: dict[str, list[np.ndarray]] = defaultdict(list)
    trial_truth: dict[str, int] = {}
    trial_meta: dict[str, dict] = {}
    for row, label, meta in zip(scores, truth, metadata):
        key = f"{meta['subject']}|{meta['condition']}|{meta['trial']}"
        by_trial[key].append(row)
        trial_truth[key] = int(label)
        trial_meta[key] = meta

    rows: list[dict] = []
    for key, score_rows in by_trial.items():
        mean_scores = np.vstack(score_rows).mean(axis=0)
        true_label = trial_truth[key]
        meta = trial_meta[key]
        for column, claim in enumerate(labels):
            rows.append(
                {
                    "dataset": "original_treadmill",
                    "model": model,
                    "protocol": "enrolled_participant_verification",
                    "group": prism.group_name(condition),
                    "subject": meta["subject"],
                    "condition": condition,
                    "trial": meta["trial"],
                    "claim": int(claim),
                    "genuine": int(int(claim) == true_label),
                    "score": float(mean_scores[column]),
                }
            )
    return rows


def fit_prism(data: dict, seed: int) -> list[dict]:
    args = SimpleNamespace(
        raw_kernels=3000,
        diff_kernels=3000,
        diff2_kernels=2000,
        seed=seed,
        add_pkd=True,
        add_phcf=True,
        pkd_groups=48,
        pkd_kernels_per_group=6,
        batch_size=256,
    )
    train = prism.stack(data["x_train"])
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    blocks = {condition: prism.stack(windows) for condition, windows in data["x_test_by_cond"].items()}
    train_features, test_features, dimensions = prism.fit_features(train, y_train, blocks, args)
    if sum(dimensions.values()) != 13049:
        raise ValueError(f"expected 13,049 PRISM-Gait features, observed {dimensions}")
    classifier = prism.make_classifier(cosine=True)
    classifier.fit(train_features, y_train)
    labels = np.asarray(classifier.named_steps["ridge"].classes_, dtype=np.int64)

    rows: list[dict] = []
    for condition, features in sorted(test_features.items()):
        rows.extend(
            aggregate_trial_claims(
                "PRISM-Gait",
                condition,
                prism.score_matrix(classifier, features),
                labels,
                np.asarray(data["y_test_by_cond"][condition], dtype=np.int64),
                data["meta_test_by_cond"][condition],
            )
        )
    return rows


def fit_baseline(data: dict, model_key: str, seed: int) -> list[dict]:
    args = SimpleNamespace(
        seed=seed,
        minirocket_kernels=10000,
        multirocket_kernels=10000,
        raw_kernels=3000,
        diff_kernels=3000,
        diff2_kernels=2000,
        add_kernel_duel=False,
        duel_groups=64,
        duel_kernels_per_group=6,
        batch_size=256,
        fixed_ridge=False,
        ridge_alpha=1.0,
    )
    train = baseline.stack(data["x_train"])
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    blocks = {condition: baseline.stack(windows) for condition, windows in data["x_test_by_cond"].items()}
    model_name, train_features, test_features, _ = baseline.fit_feature_model(
        model_key, train, y_train, blocks, args
    )
    classifier = baseline.make_classifier(cosine=True)
    classifier.fit(train_features, y_train)
    labels = np.asarray(classifier.named_steps["ridge"].classes_, dtype=np.int64)

    rows: list[dict] = []
    for condition, features in sorted(test_features.items()):
        rows.extend(
            aggregate_trial_claims(
                model_name,
                condition,
                baseline.score_matrix(classifier, features),
                labels,
                np.asarray(data["y_test_by_cond"][condition], dtype=np.int64),
                data["meta_test_by_cond"][condition],
            )
        )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "protocol",
        "group",
        "subject",
        "condition",
        "trial",
        "claim",
        "genuine",
        "score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=prism.H5_PATH)
    parser.add_argument("--out", type=Path, default=Path("baselines_out/prism_verification_figure/enrolled_trial_scores.csv"))
    parser.add_argument("--models", default="prism,minirocket,multirocket")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = prism.load_original_h5(args.h5)
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    rows: list[dict] = []
    for model in selected:
        print(f"[Fit] {model}", flush=True)
        if model == "prism":
            rows.extend(fit_prism(data, args.seed))
        elif model in {"minirocket", "multirocket"}:
            rows.extend(fit_baseline(data, model, args.seed))
        else:
            raise ValueError(f"unknown model: {model}")
        write_rows(args.out, rows)
        print(f"[Saved checkpoint] {args.out} ({len(rows):,} claim scores)", flush=True)


if __name__ == "__main__":
    main()
