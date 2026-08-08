"""Reproduce and transfer the published FCN-BiLSTM identity-squeeze model.

The network body is transcribed from the authors' executable notebook cell
``Bi_LSTM_all_se_experiments_identity_best.ipynb::MLSTM_FCN``.  The WHU-Gait
run uses Dataset #3 exactly as stored by the dataset authors.  Treadmill data
are resampled only along time to the published 128-sample input contract.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ablation.published_gait_benchmark_common import (
    majority_vote_trial_rows,
    resample_time,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("fcn_bilstm_official_config.json")
MODEL_NAME = "FCN-BiLSTM"
MODEL_KEY = "fcn_bilstm_official"
SOURCE = (
    "gait_recognition_fcn_bilstm/"
    "Bi_LSTM_all_se_experiments_identity_best.ipynb::identity-squeeze MLSTM_FCN"
)
CONDITIONS = (
    "fpa_large",
    "fpa_small",
    "step_width_large",
    "step_width_small",
    "trunk_sway_large",
    "trunk_sway_small",
    "walk_speed_large",
    "walk_speed_small",
    "dual_task",
)


@dataclass(frozen=True)
class Settings:
    seed: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    validation_fraction: float
    published_whugait_accuracy_percent: float
    admission_tolerance_percentage_points: float
    whugait_table_value_source: str
    whugait_root: Path
    treadmill_h5: Path
    output_root: Path
    treadmill_output: Path
    run_whugait: bool
    run_treadmill_after_admission: bool


def load_settings() -> Settings:
    values = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in ("whugait_root", "treadmill_h5", "output_root", "treadmill_output"):
        values[key] = ROOT / values[key]
    settings = Settings(**values)
    valid_sources = {"authors_reported", "local_reproduction"}
    if settings.whugait_table_value_source not in valid_sources:
        raise ValueError(
            "whugait_table_value_source must be authors_reported or local_reproduction"
        )
    return settings


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def squeeze_excite_block(tensor, ratio: int = 16):
    """Authors' squeeze-excitation block, including the six-channel zero bottleneck."""

    filters = tensor.shape[-1]
    if filters // ratio == 0:
        # Legacy tf.keras accepted Dense(0).  With use_bias=False in the
        # following Dense(filters), its exact output was sigmoid(0) = 0.5.
        return keras.layers.Lambda(lambda values: values * 0.5)(tensor)
    se = keras.layers.GlobalAveragePooling1D()(tensor)
    se = keras.layers.Reshape((1, filters))(se)
    se = keras.layers.Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)
    se = keras.layers.Dense(
        filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)
    return keras.layers.multiply([tensor, se])


def build_model(classes: int):
    """Identity-squeeze FCN-BiLSTM copied from the authors' final notebook cell."""

    inputs = keras.layers.Input(shape=(6, 128))
    recurrent = keras.layers.Masking()(inputs)
    recurrent = keras.layers.Bidirectional(keras.layers.LSTM(units=8))(recurrent)
    recurrent = keras.layers.Dropout(0.8)(recurrent)

    convolutional = keras.layers.Permute((2, 1))(inputs)
    residual_1 = squeeze_excite_block(convolutional)
    convolutional = keras.layers.Conv1D(
        32, 9, padding="same", kernel_initializer="he_uniform"
    )(convolutional)
    convolutional = keras.layers.BatchNormalization()(convolutional)
    convolutional = keras.layers.Activation("relu")(convolutional)
    convolutional = keras.layers.concatenate([residual_1, convolutional])

    residual_2 = squeeze_excite_block(convolutional)
    convolutional = keras.layers.Conv1D(
        512, 9, padding="same", kernel_initializer="he_uniform"
    )(convolutional)
    convolutional = keras.layers.BatchNormalization()(convolutional)
    convolutional = keras.layers.Activation("relu")(convolutional)
    convolutional = keras.layers.concatenate([residual_2, convolutional])

    residual_3 = squeeze_excite_block(convolutional)
    convolutional = keras.layers.Conv1D(
        512, 9, padding="same", kernel_initializer="he_uniform"
    )(convolutional)
    convolutional = keras.layers.BatchNormalization()(convolutional)
    convolutional = keras.layers.Activation("relu")(convolutional)
    convolutional = keras.layers.concatenate([residual_3, convolutional])
    convolutional = keras.layers.GlobalAveragePooling1D()(convolutional)

    features = keras.layers.concatenate([recurrent, convolutional])
    outputs = keras.layers.Dense(classes, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_whugait_partition(root: Path, partition: str) -> tuple[np.ndarray, np.ndarray]:
    signal_root = root / partition / "Inertial Signals"
    channel_files = sorted(signal_root.glob("*.txt"), key=lambda path: path.name.lower())
    if len(channel_files) != 6:
        raise FileNotFoundError(f"expected six WHU-Gait channels in {signal_root}")
    channels = [np.loadtxt(path, dtype=np.float32) for path in channel_files]
    inputs = np.stack(channels, axis=1).astype(np.float32, copy=False)
    labels = np.loadtxt(root / partition / f"y_{partition}.txt", dtype=np.int64) - 1
    if inputs.shape[0] != labels.shape[0] or inputs.shape[1:] != (6, 128):
        raise ValueError(f"invalid WHU-Gait {partition} shape: {inputs.shape}")
    return inputs, labels


def fit_model(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    settings: Settings,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=settings.early_stopping_patience,
            restore_best_weights=True,
        )
    ]
    fit_options = {
        "batch_size": settings.batch_size,
        "epochs": settings.max_epochs,
        "verbose": 2,
        "callbacks": callbacks,
        "shuffle": True,
    }
    if validation_data is None:
        fit_options["validation_split"] = settings.validation_fraction
    else:
        fit_options["validation_data"] = validation_data
    labels = keras.utils.to_categorical(y_train, num_classes=model.output_shape[-1])
    return model.fit(x_train, labels, **fit_options)


def predict_classes(model, inputs: np.ndarray, batch_size: int) -> np.ndarray:
    probabilities = model.predict(inputs, batch_size=batch_size, verbose=0)
    return np.argmax(probabilities, axis=1).astype(np.int64)


def run_whugait(settings: Settings) -> dict:
    settings.output_root.mkdir(parents=True, exist_ok=True)
    set_seed(settings.seed)
    x_train, y_train = load_whugait_partition(settings.whugait_root, "train")
    x_test, y_test = load_whugait_partition(settings.whugait_root, "test")
    started = perf_counter()
    model = build_model(classes=int(np.max(y_train)) + 1)
    history = fit_model(model, x_train, y_train, settings)
    predictions = predict_classes(model, x_test, settings.batch_size)
    accuracy_percent = 100.0 * accuracy_score(y_test, predictions)
    difference = accuracy_percent - settings.published_whugait_accuracy_percent
    admitted = abs(difference) <= settings.admission_tolerance_percentage_points
    result = {
        "model": MODEL_NAME,
        "model_key": MODEL_KEY,
        "source": SOURCE,
        "backend": keras.backend.backend(),
        "keras_version": keras.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seed": settings.seed,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "epochs_completed": int(len(history.history["loss"])),
        "published_accuracy_percent": settings.published_whugait_accuracy_percent,
        "measured_accuracy_percent": accuracy_percent,
        "difference_percentage_points": difference,
        "admission_tolerance_percentage_points": settings.admission_tolerance_percentage_points,
        "admitted": admitted,
        "elapsed_seconds": perf_counter() - started,
        "protocol": (
            "Authors' Dataset #3 files; identity-squeeze MLSTM_FCN; Adam; "
            "batch 32; 10% validation; early stopping on validation loss"
        ),
        "compatibility_note": (
            "Legacy Dense(0) squeeze bottleneck is evaluated as its exact "
            "bias-free sigmoid result: a constant 0.5 channel gate"
        ),
    }
    pd.DataFrame(history.history).to_csv(settings.output_root / "whugait_history.csv", index=False)
    pd.DataFrame({"truth": y_test, "prediction": predictions}).to_csv(
        settings.output_root / "whugait_predictions.csv", index=False
    )
    model.save_weights(settings.output_root / "whugait.weights.h5")
    (settings.output_root / "whugait_admission.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_treadmill(settings: Settings, whugait_evidence: dict) -> None:
    from ablation.pelvis_harmonic_covariance_model import load_original_h5

    set_seed(settings.seed)
    data = load_original_h5(settings.treadmill_h5)
    x_all = resample_time(np.stack(data["x_train"]).astype(np.float32), 128)
    y_all = np.asarray(data["y_train"], dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        np.arange(len(y_all)),
        test_size=settings.validation_fraction,
        random_state=settings.seed,
        stratify=y_all,
    )
    classes = int(np.max(y_all)) + 1
    model = build_model(classes)
    history = fit_model(
        model,
        x_all[train_indices],
        y_all[train_indices],
        settings,
        validation_data=(
            x_all[validation_indices],
            keras.utils.to_categorical(y_all[validation_indices], classes),
        ),
    )

    condition_rows = []
    subject_rows = []
    prediction_rows = []
    trial_details = []
    for condition in CONDITIONS:
        inputs = resample_time(
            np.stack(data["x_test_by_cond"][condition]).astype(np.float32),
            128,
        )
        truth = np.asarray(data["y_test_by_cond"][condition], dtype=np.int64)
        metadata = data["meta_test_by_cond"][condition]
        prediction = predict_classes(model, inputs, settings.batch_size)
        trial_summary, current_trials = majority_vote_trial_rows(
            MODEL_KEY, condition, truth, prediction, metadata
        )
        trial_details.extend(current_trials)
        condition_rows.append(
            {
                "model": MODEL_KEY,
                "display_name": MODEL_NAME,
                "condition": condition,
                "window_accuracy": float(accuracy_score(truth, prediction)),
                "window_macro_f1": float(f1_score(truth, prediction, average="macro")),
                "n_windows": int(len(truth)),
                **trial_summary,
                "input_length": 128,
                "normalization": "none",
                "source": SOURCE,
            }
        )
        per_subject: dict[str, list[int]] = defaultdict(list)
        for target, estimate, meta in zip(truth, prediction, metadata):
            subject = str(meta["subject"])
            per_subject[subject].append(int(target == estimate))
            prediction_rows.append(
                {
                    "model": MODEL_KEY,
                    "condition": condition,
                    "subject": subject,
                    "trial": str(meta["trial"]),
                    "truth": int(target),
                    "prediction": int(estimate),
                    "correct": int(target == estimate),
                }
            )
        for subject, correct in sorted(per_subject.items()):
            subject_rows.append(
                {
                    "model": MODEL_KEY,
                    "condition": condition,
                    "subject": subject,
                    "subject_window_accuracy": float(np.mean(correct)),
                    "n_windows": len(correct),
                }
            )

    settings.treadmill_output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(condition_rows).to_csv(
        settings.treadmill_output / "official_treadmill_results.csv", index=False
    )
    pd.DataFrame(subject_rows).to_csv(
        settings.treadmill_output / "official_subject_condition_accuracy.csv",
        index=False,
    )
    pd.DataFrame(prediction_rows).to_csv(
        settings.treadmill_output / "official_window_predictions.csv", index=False
    )
    pd.DataFrame(trial_details).to_csv(
        settings.treadmill_output / "official_trial_predictions.csv", index=False
    )
    pd.DataFrame(history.history).to_csv(
        settings.treadmill_output / "training_history.csv", index=False
    )
    model.save_weights(settings.treadmill_output / "treadmill.weights.h5")
    audit = {
        "model": MODEL_NAME,
        "model_key": MODEL_KEY,
        "source": SOURCE,
        "architecture_modified": False,
        "whugait_reference": {
            "table_value_source": settings.whugait_table_value_source,
            "table_accuracy_percent": (
                settings.published_whugait_accuracy_percent
                if settings.whugait_table_value_source == "authors_reported"
                else whugait_evidence["measured_accuracy_percent"]
            ),
            "authors_reported_accuracy_percent": settings.published_whugait_accuracy_percent,
            "local_cross_framework_accuracy_percent": whugait_evidence[
                "measured_accuracy_percent"
            ],
            "local_reproduction_status": (
                "cross-framework reproduction inconclusive"
                if settings.whugait_table_value_source == "authors_reported"
                else "local reproduction used"
            ),
        },
        "compatibility_note": (
            "Legacy Dense(0) squeeze bottleneck is evaluated as its exact "
            "bias-free sigmoid result: a constant 0.5 channel gate"
        ),
        "input_adapter": "linear temporal resampling from 200 to 128 samples",
        "normalization": "none",
        "training_split": "stratified 90/10 split of treadmill enrollment windows",
        "settings": {
            **asdict(settings),
            "whugait_root": str(settings.whugait_root),
            "treadmill_h5": str(settings.treadmill_h5),
            "output_root": str(settings.output_root),
            "treadmill_output": str(settings.treadmill_output),
        },
    }
    (settings.treadmill_output / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def main() -> None:
    settings = load_settings()
    if settings.run_whugait:
        admission = run_whugait(settings)
        print(json.dumps(admission, indent=2), flush=True)
    else:
        admission_path = settings.output_root / "whugait_admission.json"
        if not admission_path.is_file():
            raise FileNotFoundError(
                "the configured WHU-Gait reproduction audit does not exist"
            )
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if settings.run_treadmill_after_admission:
        if (
            settings.whugait_table_value_source == "local_reproduction"
            and not admission["admitted"]
        ):
            raise RuntimeError("FCN-BiLSTM did not pass the declared WHU-Gait admission gate")
        run_treadmill(settings, admission)


if __name__ == "__main__":
    main()
