"""Time the exact FCN-BiLSTM treadmill enrollment fit without overwriting it."""

from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
from time import perf_counter

import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ablation.fcn_bilstm_official_benchmark import (
    build_model,
    load_settings,
    set_seed,
)
from ablation.pelvis_harmonic_covariance_model import load_original_h5
from ablation.published_gait_benchmark_common import resample_time


def run(root: Path) -> Path:
    settings = load_settings()
    set_seed(settings.seed)
    data = load_original_h5(settings.treadmill_h5)
    inputs = resample_time(
        np.stack(data["x_train"]).astype(np.float32),
        128,
    )
    labels = np.asarray(data["y_train"], dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        np.arange(len(labels)),
        test_size=settings.validation_fraction,
        random_state=settings.seed,
        stratify=labels,
    )
    classes = int(np.max(labels)) + 1
    model = build_model(classes)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=settings.early_stopping_patience,
            restore_best_weights=True,
        )
    ]
    started = perf_counter()
    history = model.fit(
        inputs[train_indices],
        keras.utils.to_categorical(labels[train_indices], classes),
        validation_data=(
            inputs[validation_indices],
            keras.utils.to_categorical(labels[validation_indices], classes),
        ),
        batch_size=settings.batch_size,
        epochs=settings.max_epochs,
        verbose=0,
        callbacks=callbacks,
        shuffle=True,
    )
    elapsed = perf_counter() - started
    output = (
        root
        / "baselines_out"
        / "treadmill_published_plot_data"
        / "fcn_bilstm_registration_timing.csv"
    )
    pd.DataFrame(
        [
            {
                "model": "FCN-BiLSTM",
                "registration_seconds": elapsed,
                "epochs_completed": len(history.history["loss"]),
                "definition": (
                    "exact treadmill enrollment model fit after loading and "
                    "200-to-128 resampling; data-collection time excluded"
                ),
                "device": (
                    "NVIDIA GeForce RTX 5090 GPU"
                    if keras.backend.backend() == "torch"
                    else keras.backend.backend()
                ),
                "source": "current exact reconstruction",
            }
        ]
    ).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(run(Path(__file__).resolve().parents[1]))
