"""Benchmark batch-one FCN-BiLSTM preprocessing and inference on CPU."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from ablation.fcn_bilstm_official_benchmark import build_model
from ablation.pelvis_harmonic_covariance_model import load_original_h5
from ablation.published_gait_benchmark_common import resample_time


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_median_ms": float(np.median(values)),
        f"{prefix}_q1_ms": float(np.percentile(values, 25)),
        f"{prefix}_q3_ms": float(np.percentile(values, 75)),
        f"{prefix}_p95_ms": float(np.percentile(values, 95)),
    }


def run(root: Path, repetitions: int = 50, warmups: int = 5) -> Path:
    data = load_original_h5(root / "gait_id_pelvis_62.h5")
    conditions = (
        "fpa_small",
        "fpa_large",
        "step_width_small",
        "step_width_large",
        "walk_speed_small",
        "walk_speed_large",
        "dual_task",
    )
    windows = [
        window
        for condition in conditions
        for window in data["x_test_by_cond"][condition]
    ]
    rng = np.random.RandomState(42)
    indices = rng.choice(
        len(windows),
        size=repetitions + warmups,
        replace=False,
    )
    selected = [windows[index] for index in indices]
    model = build_model(classes=len(data["subjects"]))
    model.load_weights(
        root
        / "baselines_out"
        / "treadmill_published_subject_sd"
        / "fcn_bilstm_official"
        / "treadmill.weights.h5"
    )

    preprocessing_ms = []
    inference_ms = []
    for index, window in enumerate(selected):
        started = perf_counter()
        inputs = resample_time(
            np.asarray(window, dtype=np.float32)[None, ...],
            128,
        )
        after_preprocessing = perf_counter()
        with torch.no_grad():
            probabilities = model(inputs, training=False)
            probabilities.detach().cpu().numpy().argmax(axis=1)
        finished = perf_counter()
        if index >= warmups:
            preprocessing_ms.append(
                1000.0 * (after_preprocessing - started)
            )
            inference_ms.append(1000.0 * (finished - after_preprocessing))

    preprocessing = np.asarray(preprocessing_ms, dtype=float)
    inference = np.asarray(inference_ms, dtype=float)
    total = preprocessing + inference
    row = {
        "model": "FCN-BiLSTM",
        **summarize(preprocessing, "preprocessing"),
        **summarize(inference, "inference"),
        **summarize(total, "online"),
        "repetitions": repetitions,
        "batch_size": 1,
        "definition": (
            "linear 200-to-128 temporal resampling plus checkpoint inference "
            "for one accepted 2-s window; common acquisition, segmentation, "
            "and quality gating excluded"
        ),
        "device": "Intel Core i9-14900K CPU",
    }
    output = (
        root
        / "baselines_out"
        / "treadmill_published_plot_data"
        / "fcn_bilstm_online_latency.csv"
    )
    pd.DataFrame([row]).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(run(Path(__file__).resolve().parents[1]))
