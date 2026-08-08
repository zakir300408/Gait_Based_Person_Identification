"""Five-fold subject-disjoint FCN-BiLSTM verification."""

from __future__ import annotations

import os
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ablation.fcn_bilstm_official_benchmark import (
    build_model, fit_model, load_settings, resample_time, set_seed,
)
from ablation.neural_verification_common import (
    CONDITIONS, OUTPUT, ROOT, create_logger, load_data, save_audit, save_scores,
    subject_folds, subset, trial_rows,
)


MODEL = "FCN-BiLSTM"
KEY = "fcn_bilstm"


def prepare(windows: list[np.ndarray]) -> np.ndarray:
    return resample_time(np.stack(windows).astype(np.float32), 128)


def predict(model, windows: list[np.ndarray], batch: int) -> np.ndarray:
    return model.predict(prepare(windows), batch_size=batch, verbose=0)


def main() -> None:
    settings, data, logger = load_settings(), load_data(), create_logger(KEY)
    requested = os.environ.get("FCN_FOLDS", "1,2,3,4,5")
    requested_folds = {int(value) for value in requested.split(",")}
    if not requested_folds.issubset({1, 2, 3, 4, 5}):
        raise ValueError("FCN_FOLDS must contain values from 1 through 5")
    for fold, enrolled, impostors in subject_folds(data):
        if fold not in requested_folds:
            continue
        split = subset(data, enrolled, impostors)
        set_seed(settings.seed + fold)
        x, y = prepare(split["train_x"]), split["train_y"]
        fold_dir = OUTPUT / KEY / f"fold_{fold}"
        checkpoint = fold_dir / "model.weights.h5"
        model = build_model(len(enrolled))
        if checkpoint.is_file():
            model.load_weights(checkpoint)
            epochs = "checkpoint"
        else:
            train, validation = train_test_split(
                np.arange(len(y)), test_size=settings.validation_fraction,
                random_state=settings.seed, stratify=y,
            )
            history = fit_model(
                model, x[train], y[train], settings,
                validation_data=(
                    x[validation],
                    keras.utils.to_categorical(y[validation], len(enrolled)),
                ),
            )
            fold_dir.mkdir(parents=True, exist_ok=True)
            model.save_weights(checkpoint)
            epochs = len(history.history["loss"])
        labels = np.arange(len(enrolled), dtype=np.int64)
        fold_rows = []
        for condition in CONDITIONS:
            gx, gy, gm = split["genuine"][condition]
            fold_rows += trial_rows(
                MODEL, fold, condition, predict(model, gx, settings.batch_size),
                gm, labels, np.asarray(gy),
            )
        for condition, (ix, im) in split["impostor"].items():
            fold_rows += trial_rows(
                MODEL, fold, condition, predict(model, ix, settings.batch_size),
                im, labels, None,
            )
        pd.DataFrame(fold_rows).to_csv(
            fold_dir / "open_set_trial_scores.csv", index=False
        )
        logger.info("fold=%s enrolled=%s impostors=%s epochs=%s rows=%s",
                    fold, len(enrolled), len(impostors), epochs, len(fold_rows))
        keras.backend.clear_session()
    fold_paths = [
        OUTPUT / KEY / f"fold_{fold}" / "open_set_trial_scores.csv"
        for fold in range(1, 6)
    ]
    if not all(path.is_file() for path in fold_paths):
        logger.info("worker complete; awaiting remaining fold score files")
        return
    open_rows = pd.concat(
        [pd.read_csv(path) for path in fold_paths], ignore_index=True
    ).to_dict(orient="records")
    open_path = save_scores(KEY, "open_set", open_rows)

    set_seed(settings.seed)
    full = build_model(len(data["subjects"]))
    full.load_weights(
        ROOT / "baselines_out/treadmill_published_subject_sd/fcn_bilstm_official/treadmill.weights.h5"
    )
    enrolled_rows = []
    labels = np.arange(len(data["subjects"]), dtype=np.int64)
    for condition in CONDITIONS:
        enrolled_rows += trial_rows(
            MODEL, "", condition,
            predict(full, data["x_test_by_cond"][condition], settings.batch_size),
            data["meta_test_by_cond"][condition], labels,
            np.asarray(data["y_test_by_cond"][condition]), enrolled_protocol=True,
        )
    enrolled_path = save_scores(KEY, "enrolled", enrolled_rows)
    save_audit(KEY, {"model": MODEL, "folds": 5, "seed": 42,
                    "open_scores": str(open_path), "enrolled_scores": str(enrolled_path)})


if __name__ == "__main__":
    main()
