"""Fit the exact PRISM-Gait architecture on a public identification split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.metrics import accuracy_score, f1_score

from analysis.prism_gait.verification import fit_features, make_classifier, score_matrix
from analysis.datasets.loaders import load_dataset


DATASET_NAMES = {
    "whugait_dataset3": "whuGAIT Dataset #3",
    "ou_isir_identification": "OU-ISIR inertial sensor dataset",
}


def prepare(x: np.ndarray, samples: int, batch_size: int) -> np.ndarray:
    output = np.empty((len(x), x.shape[1], samples), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        block = resample(
            np.asarray(x[start:stop], dtype=np.float32), samples, axis=2
        ).astype(np.float32)
        mean = block.mean(axis=2, keepdims=True)
        std = block.std(axis=2, keepdims=True) + 1e-8
        output[start:stop] = (block - mean) / std
    return output


def participant_rows(
    labels: np.ndarray, truth: np.ndarray, scores: np.ndarray
) -> pd.DataFrame:
    rows = []
    for subject in sorted(np.unique(truth)):
        index = np.where(truth == subject)[0]
        mean_score = scores[index].mean(axis=0)
        prediction = int(labels[np.argmax(mean_score)])
        ordered = np.sort(mean_score)
        rows.append(
            {
                "subject": int(subject),
                "test_windows": int(len(index)),
                "prediction": prediction,
                "correct": int(prediction == subject),
                "top_score": float(ordered[-1]),
                "score_margin": float(ordered[-1] - ordered[-2]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_NAMES),
        default="ou_isir_identification",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("baselines_out/ou_isir_exact_prism_registration"),
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--raw-kernels", type=int, default=3000)
    parser.add_argument("--diff-kernels", type=int, default=3000)
    parser.add_argument("--diff2-kernels", type=int, default=2000)
    parser.add_argument("--add-pkd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-phcf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pkd-groups", type=int, default=48)
    parser.add_argument("--pkd-kernels-per-group", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cosine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()

    x_train_raw, y_train, x_test_raw, y_test, source = load_dataset(args.dataset)
    print(
        f"[Data] dataset={args.dataset} train={x_train_raw.shape} "
        f"test={x_test_raw.shape} subjects={len(np.unique(y_train))}",
        flush=True,
    )
    x_train = prepare(x_train_raw, args.samples, args.batch_size)
    x_test = prepare(x_test_raw, args.samples, args.batch_size)
    print(f"[Prepare] train={x_train.shape} test={x_test.shape}", flush=True)

    train_features, test_blocks, dimensions = fit_features(
        x_train,
        np.asarray(y_train, dtype=np.int64),
        {"official_test": x_test},
        args,
    )
    classifier = make_classifier(cosine=args.cosine)
    classifier.fit(train_features, y_train)
    scores = score_matrix(classifier, test_blocks["official_test"])
    labels = np.asarray(classifier.named_steps["ridge"].classes_, dtype=np.int64)
    predictions = labels[np.argmax(scores, axis=1)]
    participants = participant_rows(labels, y_test, scores)

    window_predictions = pd.DataFrame(
        {
            "window": np.arange(len(y_test)),
            "subject": y_test.astype(int),
            "prediction": predictions.astype(int),
            "correct": (predictions == y_test).astype(int),
        }
    )
    summary = {
        "dataset": DATASET_NAMES[args.dataset],
        "dataset_id": args.dataset,
        "protocol": (
            "complete provided train split registration; provided test split evaluation; "
            "200-sample per-window channel z-score preprocessing"
        ),
        "model": (
            "PRISM-Gait exact four-scale architecture: "
            "raw3000/diff3000/diff2_2000 + PKD + PHCF + cosine Ridge"
        ),
        "identity_head": "separately fitted for this dataset",
        "source_protocol": source["paper"],
        "subjects": int(len(labels)),
        "train_windows": int(len(y_train)),
        "test_windows": int(len(y_test)),
        "feature_dimensions": dimensions,
        "total_feature_dimensions": int(train_features.shape[1]),
        "window_accuracy": float(accuracy_score(y_test, predictions)),
        "window_macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "participant_accuracy": float(participants.correct.mean()),
        "participant_macro_f1": float(
            f1_score(participants.subject, participants.prediction, average="macro")
        ),
        "misidentified_participants": int((participants.correct == 0).sum()),
        "elapsed_seconds": float(perf_counter() - t0),
    }
    window_predictions.to_csv(args.out_dir / "window_predictions.csv", index=False)
    participants.to_csv(args.out_dir / "participant_predictions.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
