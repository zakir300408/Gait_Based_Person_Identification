"""Benchmark fit-only enrollment time for the three classical gait models."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ablation import mblite_ablation_study as prism
from ablation import run_strict_published_rocket_benchmarks as rocket
from ablation.benchmark_classical_online_latency import Settings, fit_prism_pipeline


def run(settings: Settings) -> Path:
    rows = []
    prism_data = prism.load_original_h5_windowed(
        settings.root / "gait_id_pelvis_62.h5",
        2.0,
        settings.acc_std_min,
    )
    started = perf_counter()
    fit_prism_pipeline(prism_data, settings)
    rows.append(
        {
            "model": "PRISM-Gait",
            "registration_seconds": perf_counter() - started,
            "definition": (
                "model-specific feature-transform and classifier fitting "
                "after accepted enrollment windows were loaded"
            ),
            "device": "Intel Core i9-14900K CPU",
            "source": "current standardized timing run",
        }
    )

    data = rocket.load_original_h5(settings.root / "gait_id_pelvis_62.h5")
    inputs = rocket.stack(data["x_train"])
    labels = np.asarray(data["y_train"], dtype=np.int64)
    for model in ("MultiRocket", "MiniRocket"):
        classifier = rocket.build_classifier(model, settings.seed)
        started = perf_counter()
        classifier.fit(inputs, labels)
        rows.append(
            {
                "model": model,
                "registration_seconds": perf_counter() - started,
                "definition": (
                    "transform and classifier fitting after accepted enrollment "
                    "windows were loaded"
                ),
                "device": "Intel Core i9-14900K CPU",
                "source": "current standardized timing run",
            }
        )
    output = (
        settings.root
        / "baselines_out"
        / "treadmill_published_plot_data"
        / "classical_model_registration_timing.csv"
    )
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        run(
            Settings(
                root=root,
                seed=42,
                repetitions=50,
                warmups=5,
                acc_std_min=0.4,
                pkd_groups=48,
                pkd_kernels_per_group=6,
                pkd_dilations=(1, 2, 4, 8),
                batch_size=256,
            )
        )
    )
