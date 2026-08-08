"""Export exact trial decisions from saved InceptionTime checkpoints."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ablation.pelvis_harmonic_covariance_model import load_original_h5
from ablation.run_inceptiontime_author_ensemble_treadmill import (
    author_classifier,
    prepare_windows,
)


CONDITIONS = (
    "fpa_small",
    "fpa_large",
    "step_width_small",
    "step_width_large",
    "walk_speed_small",
    "walk_speed_large",
    "dual_task",
    "trunk_sway_small",
    "trunk_sway_large",
)
SEEDS = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class Settings:
    h5_path: Path
    author_source: Path
    checkpoint_root: Path
    output_path: Path
    batch_size: int


def create_logger(output_dir: Path) -> logging.Logger:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"inception_trial_decisions_{stamp}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        output_dir / f"inception_trial_decisions_{stamp}.log",
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def load_models(settings: Settings) -> list:
    classifier_type = author_classifier(settings.author_source)
    models = []
    for seed in SEEDS:
        checkpoint = (
            settings.checkpoint_root
            / f"member_{seed}"
            / "best_validation_model.h5"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint.name}")
        classifier = classifier_type(
            "",
            (200, 6),
            62,
            build=False,
            batch_size=settings.batch_size,
            nb_filters=32,
            use_residual=True,
            use_bottleneck=True,
            depth=6,
            kernel_size=41,
            nb_epochs=60,
        )
        model = classifier.build_model((200, 6), 62)
        model.load_weights(checkpoint)
        models.append(model)
    return models


def trial_rows(
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
        rows.append(
            {
                "model": "InceptionTime",
                "condition": condition,
                "subject": subject,
                "trial": trial,
                "truth": truths[(subject, trial)],
                "prediction": vote,
                "correct": int(truths[(subject, trial)] == vote),
                "n_windows": len(estimates),
            }
        )
    return rows


def run(settings: Settings) -> Path:
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = create_logger(settings.output_path.parent)
    logger.info("step_01: loading saved five-member ensemble")
    models = load_models(settings)
    data = load_original_h5(settings.h5_path)
    rows: list[dict[str, object]] = []
    logger.info("step_02: predicting all nine treadmill conditions")
    for condition in CONDITIONS:
        inputs = prepare_windows(data["x_test_by_cond"][condition])
        probabilities = np.mean(
            np.stack(
                [
                    model.predict(
                        inputs,
                        batch_size=settings.batch_size,
                        verbose=0,
                    )
                    for model in models
                ]
            ),
            axis=0,
        )
        rows.extend(
            trial_rows(
                condition,
                np.asarray(data["y_test_by_cond"][condition], dtype=np.int64),
                probabilities.argmax(axis=1),
                data["meta_test_by_cond"][condition],
            )
        )
        logger.info("condition=%s trials=%s", condition, len(rows))
    output = pd.DataFrame(rows)
    if len(output) != 538 or output.duplicated(
        ["model", "condition", "subject", "trial"]
    ).any():
        raise ValueError("InceptionTime trial-decision export is incomplete")
    output.to_csv(settings.output_path, index=False)
    logger.info("step_03: verified rows=%s", len(output))
    return settings.output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output_root = (
        root
        / "baselines_out"
        / "treadmill_published_subject_sd"
        / "inceptiontime_author_ensemble"
    )
    print(
        run(
            Settings(
                h5_path=root / "gait_id_pelvis_62.h5",
                author_source=root / "third_party" / "InceptionTime",
                checkpoint_root=output_root,
                output_path=output_root
                / "inceptiontime_ensemble_trial_predictions.csv",
                batch_size=64,
            )
        )
    )
