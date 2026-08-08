"""Benchmark batch-one online latency for the three classical gait models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ablation import mblite_ablation_study as prism
from ablation import run_strict_published_rocket_benchmarks as rocket


@dataclass(frozen=True)
class Settings:
    root: Path
    seed: int
    repetitions: int
    warmups: int
    acc_std_min: float
    pkd_groups: int
    pkd_kernels_per_group: int
    pkd_dilations: tuple[int, ...]
    batch_size: int


def create_logger(output_dir: Path) -> logging.Logger:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"classical_latency_{stamp}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        output_dir / f"classical_latency_{stamp}.log",
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def summarize(model: str, samples_ms: list[float]) -> dict[str, object]:
    values = np.asarray(samples_ms, dtype=float)
    return {
        "model": model,
        "online_median_ms": float(np.median(values)),
        "online_q1_ms": float(np.percentile(values, 25)),
        "online_q3_ms": float(np.percentile(values, 75)),
        "online_p95_ms": float(np.percentile(values, 95)),
        "repetitions": len(values),
        "batch_size": 1,
        "definition": (
            "model-specific preprocessing plus inference for one accepted "
            "2-s window; common acquisition, segmentation, and quality gating excluded"
        ),
        "device": "Intel Core i9-14900K CPU",
    }


def benchmark_rocket(
    model: str,
    classifier,
    windows: list[np.ndarray],
    warmups: int,
) -> dict[str, object]:
    samples = []
    for index, window in enumerate(windows):
        inputs = np.ascontiguousarray(window[None, ...], dtype=np.float64)
        started = perf_counter()
        classifier.predict(inputs)
        elapsed_ms = 1000.0 * (perf_counter() - started)
        if index >= warmups:
            samples.append(elapsed_ms)
    return summarize(model, samples)


def fit_prism_pipeline(data: dict, settings: Settings):
    config = next(
        item for item in prism.COMPONENT_VARIANTS if item["variant"] == "MB-Lite"
    )
    train_raw = prism.stack(data["x_train"])
    labels = np.asarray(data["y_train"], dtype=np.int64)
    minirocket = prism.import_minirocket()
    branch_models = {}
    train_blocks = []
    for index, branch in enumerate(config["branches"]):
        transform = minirocket(
            num_kernels=int(prism.BRANCH_KERNELS[branch]),
            random_state=settings.seed + index,
            n_jobs=-1,
        )
        branch_input = np.ascontiguousarray(
            prism.branch_transform(branch, train_raw),
            dtype=np.float64,
        )
        transform.fit(branch_input, labels)
        train_blocks.append(
            np.asarray(transform.transform(branch_input), dtype=np.float32)
        )
        branch_models[branch] = transform
    features = np.hstack(train_blocks).astype(np.float32)
    args = SimpleNamespace(
        pkd_groups=settings.pkd_groups,
        pkd_kernels_per_group=settings.pkd_kernels_per_group,
        pkd_dilations=settings.pkd_dilations,
        batch_size=settings.batch_size,
    )
    pkd = prism.PelvisKernelDuelFeatures(
        groups=args.pkd_groups,
        kernels_per_group=args.pkd_kernels_per_group,
        kernel_size=9,
        dilations=tuple(args.pkd_dilations),
        seed=settings.seed,
        batch_size=args.batch_size,
    ).fit(train_raw.astype(np.float32))
    features = np.hstack(
        [features, pkd.transform(train_raw.astype(np.float32))]
    ).astype(np.float32)
    phcf = prism.phcf_featurize(
        [window.astype(np.float32) for window in data["x_train"]]
    )
    features = np.hstack([features, phcf]).astype(np.float32)
    classifier = prism.make_classifier().fit(features, labels)
    return branch_models, pkd, classifier


def benchmark_prism(
    branch_models: dict,
    pkd,
    classifier,
    windows: list[np.ndarray],
    warmups: int,
) -> dict[str, object]:
    samples = []
    for index, window in enumerate(windows):
        raw = np.ascontiguousarray(window[None, ...], dtype=np.float64)
        started = perf_counter()
        branch_features = [
            np.asarray(
                transform.transform(
                    np.ascontiguousarray(
                        prism.branch_transform(branch, raw),
                        dtype=np.float64,
                    )
                ),
                dtype=np.float32,
            )
            for branch, transform in branch_models.items()
        ]
        pkd_features = pkd.transform(raw.astype(np.float32))
        phcf_features = prism.phcf_featurize([raw[0].astype(np.float32)])
        features = np.hstack(
            [*branch_features, pkd_features, phcf_features]
        ).astype(np.float32)
        classifier.predict(features)
        elapsed_ms = 1000.0 * (perf_counter() - started)
        if index >= warmups:
            samples.append(elapsed_ms)
    return summarize("PRISM-Gait", samples)


def run(settings: Settings) -> Path:
    output_dir = (
        settings.root / "baselines_out" / "treadmill_published_plot_data"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_dir)
    prism_data = prism.load_original_h5_windowed(
        settings.root / "gait_id_pelvis_62.h5",
        2.0,
        settings.acc_std_min,
    )
    all_windows = [
        window
        for condition in (
            "fpa_small",
            "fpa_large",
            "step_width_small",
            "step_width_large",
            "walk_speed_small",
            "walk_speed_large",
            "dual_task",
        )
        for window in prism_data["x_test_by_cond"][condition]
    ]
    rng = np.random.RandomState(settings.seed)
    indices = rng.choice(
        len(all_windows),
        size=settings.warmups + settings.repetitions,
        replace=False,
    )
    windows = [all_windows[index] for index in indices]

    logger.info("step_01: fitting PRISM-Gait")
    registration_started = perf_counter()
    branches, pkd, prism_classifier = fit_prism_pipeline(prism_data, settings)
    prism_registration = perf_counter() - registration_started
    rows = [
        benchmark_prism(
            branches,
            pkd,
            prism_classifier,
            windows,
            settings.warmups,
        )
    ]

    rocket_data = rocket.load_original_h5(
        settings.root / "gait_id_pelvis_62.h5"
    )
    training = rocket.stack(rocket_data["x_train"])
    labels = np.asarray(rocket_data["y_train"], dtype=np.int64)
    for model in ("MultiRocket", "MiniRocket"):
        logger.info("step_02: fitting %s", model)
        classifier = rocket.build_classifier(model, settings.seed)
        registration_started = perf_counter()
        classifier.fit(training, labels)
        registration_seconds = perf_counter() - registration_started
        if model == "MultiRocket":
            multirocket_registration = registration_seconds
        else:
            minirocket_registration = registration_seconds
        rows.append(
            benchmark_rocket(
                model,
                classifier,
                windows,
                settings.warmups,
            )
        )
    output_path = output_dir / "classical_model_online_latency.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    pd.DataFrame(
        [
            {
                "model": "PRISM-Gait",
                "registration_seconds": prism_registration,
                "definition": (
                    "model-specific feature-transform and classifier fitting "
                    "after accepted enrollment windows were loaded"
                ),
                "device": "Intel Core i9-14900K CPU",
                "source": "current standardized timing run",
            },
            {
                "model": "MultiRocket",
                "registration_seconds": multirocket_registration,
                "definition": (
                    "transform and classifier fitting after accepted enrollment "
                    "windows were loaded"
                ),
                "device": "Intel Core i9-14900K CPU",
                "source": "current standardized timing run",
            },
            {
                "model": "MiniRocket",
                "registration_seconds": minirocket_registration,
                "definition": (
                    "transform and classifier fitting after accepted enrollment "
                    "windows were loaded"
                ),
                "device": "Intel Core i9-14900K CPU",
                "source": "current standardized timing run",
            },
        ]
    ).to_csv(
        output_dir / "classical_model_registration_timing.csv",
        index=False,
    )
    logger.info("step_03: saved %s", output_path.name)
    return output_path


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
