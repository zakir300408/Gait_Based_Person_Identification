"""Shared adapters and aggregation for published gait-identification benchmarks."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as torch_functional


def resample_time(inputs: np.ndarray, length: int) -> np.ndarray:
    """Linearly adapt only the temporal axis of channels-first windows."""

    tensor = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32))
    return (
        torch_functional.interpolate(
            tensor,
            size=length,
            mode="linear",
            align_corners=False,
        )
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def majority_vote_trial_rows(
    model_key: str,
    condition: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    metadata: list[dict],
) -> tuple[dict, list[dict]]:
    """Aggregate window predictions within each participant-condition trial."""

    grouped_predictions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    grouped_truth: dict[tuple[str, str, str], int] = {}
    for target, estimate, meta in zip(truth, prediction, metadata):
        key = (str(meta["subject"]), str(meta["condition"]), str(meta["trial"]))
        grouped_predictions[key].append(int(estimate))
        grouped_truth[key] = int(target)
    detail = []
    for key, estimates in sorted(grouped_predictions.items()):
        vote = Counter(estimates).most_common(1)[0][0]
        detail.append(
            {
                "model": model_key,
                "condition": condition,
                "subject": key[0],
                "trial": key[2],
                "truth": grouped_truth[key],
                "prediction": vote,
                "correct": int(grouped_truth[key] == vote),
                "n_windows": len(estimates),
            }
        )
    return (
        {
            "trial_accuracy": float(np.mean([row["correct"] for row in detail])),
            "n_trials": len(detail),
        },
        detail,
    )
