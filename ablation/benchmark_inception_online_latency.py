"""Benchmark batch-one five-member InceptionTime latency on CPU."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ablation.export_inception_trial_decisions import Settings, load_models
from ablation.pelvis_harmonic_covariance_model import load_original_h5
from ablation.run_inceptiontime_author_ensemble_treadmill import prepare_windows


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_median_ms": float(np.median(values)),
        f"{prefix}_q1_ms": float(np.percentile(values, 25)),
        f"{prefix}_q3_ms": float(np.percentile(values, 75)),
        f"{prefix}_p95_ms": float(np.percentile(values, 95)),
    }


def run(root: Path, repetitions: int = 50, warmups: int = 5) -> Path:
    checkpoint_root = (
        root
        / "baselines_out"
        / "treadmill_published_subject_sd"
        / "inceptiontime_author_ensemble"
    )
    settings = Settings(
        h5_path=root / "gait_id_pelvis_62.h5",
        author_source=root / "third_party" / "InceptionTime",
        checkpoint_root=checkpoint_root,
        output_path=checkpoint_root / "unused.csv",
        batch_size=1,
    )
    models = load_models(settings)
    data = load_original_h5(settings.h5_path)
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

    preprocessing_ms = []
    inference_ms = []
    for index, window in enumerate(selected):
        started = perf_counter()
        inputs = prepare_windows([window])
        after_preprocessing = perf_counter()
        probabilities = np.mean(
            np.stack(
                [
                    model.predict(inputs, batch_size=1, verbose=0)
                    for model in models
                ]
            ),
            axis=0,
        )
        probabilities.argmax(axis=1)
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
        "model": "InceptionTime",
        **summarize(preprocessing, "preprocessing"),
        **summarize(inference, "inference"),
        **summarize(total, "online"),
        "repetitions": repetitions,
        "batch_size": 1,
        "definition": (
            "per-window channel standardization plus sequential inference "
            "through all five ensemble members for one accepted 2-s window; "
            "common acquisition, segmentation, and quality gating excluded"
        ),
        "device": "Intel Core i9-14900K CPU",
    }
    output = (
        root
        / "baselines_out"
        / "treadmill_published_plot_data"
        / "inceptiontime_online_latency.csv"
    )
    pd.DataFrame([row]).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(run(Path(__file__).resolve().parents[1]))
