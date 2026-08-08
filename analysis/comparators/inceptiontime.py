"""Run the authors' five-member InceptionTime ensemble on treadmill data."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from analysis.prism_gait.hcgf import load_original_h5

AUTHOR_COMMIT = "952d115e83fb3a66d75c5858c702f8dd5eeb18b7"
AUTHOR_SOURCE = f"https://github.com/hfawaz/InceptionTime/tree/{AUTHOR_COMMIT}"
MODEL_KEY = "inceptiontime_author_five_member_ensemble"


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_metadata(settings: "Settings", histories: dict, rows: list[dict]) -> Path:
    metadata = {
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()},
        "environment": {"tensorflow": tf.__version__, "keras": keras.__version__, "sklearn": sklearn.__version__},
        "compatibility_change": "Runtime alias for Keras BatchNormalization namespace only",
        "source": AUTHOR_SOURCE, "histories": histories, "rows": rows,
    }
    path = settings.output_dir / "inceptiontime_ensemble_results.json"
    path.write_text(json.dumps(metadata, indent=2, default=json_default), encoding="utf-8")
    return path


def rebuild_metadata(settings: "Settings") -> Path:
    histories = {
        str(seed): pd.read_csv(settings.output_dir / f"member_{seed}" / "training.csv").to_dict(orient="list")
        for seed in settings.member_seeds
    }
    rows = pd.read_csv(settings.output_dir / "inceptiontime_ensemble_results.csv").to_dict(orient="records")
    return write_metadata(settings, histories, rows)


@dataclass(frozen=True)
class Settings:
    h5_path: Path
    source_dir: Path
    output_dir: Path
    member_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    epochs: int = 60
    patience: int = 12
    batch_size: int = 64
    validation_fraction: float = 0.15


def create_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"inceptiontime_author_{stamp}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_dir / f"run_{stamp}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_windows(windows: list[np.ndarray]) -> np.ndarray:
    x = np.stack(windows).astype(np.float32, copy=False)
    x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)
    return np.ascontiguousarray(x.transpose(0, 2, 1))


def author_classifier(source_dir: Path):
    if not (source_dir / "classifiers" / "inception.py").is_file():
        raise FileNotFoundError(f"missing pinned authors' repository: {source_dir}")
    if not hasattr(keras.layers.normalization, "BatchNormalization"):
        keras.layers.normalization.BatchNormalization = keras.layers.BatchNormalization
    from classifiers.inception import Classifier_INCEPTION

    return Classifier_INCEPTION


def average_five(probabilities: list[np.ndarray]) -> np.ndarray:
    if len(probabilities) != 5:
        raise ValueError("The published InceptionTime ensemble requires exactly five members")
    return np.mean(np.stack(probabilities), axis=0)


def trial_metrics(labels: np.ndarray, predictions: np.ndarray, metadata: list[dict]) -> dict:
    grouped, truths = defaultdict(list), {}
    for truth, prediction, meta in zip(labels, predictions, metadata):
        key = f"{meta['subject']}|{meta['condition']}|{meta['trial']}"
        grouped[key].append(int(prediction))
        truths[key] = int(truth)
    predicted = [Counter(values).most_common(1)[0][0] for values in grouped.values()]
    actual = list(truths.values())
    return {
        "trial_accuracy": float(accuracy_score(actual, predicted)),
        "trial_macro_f1": float(f1_score(actual, predicted, average="macro")),
        "n_trials": len(actual),
    }


def subject_rows(condition: str, labels: np.ndarray, predictions: np.ndarray, metadata: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for truth, prediction, meta in zip(labels, predictions, metadata):
        grouped[meta["subject"]].append(int(truth == prediction))
    return [
        {
            "model": MODEL_KEY,
            "condition": condition,
            "subject": subject,
            "subject_window_accuracy": float(np.mean(correct)),
            "n_windows": len(correct),
        }
        for subject, correct in sorted(grouped.items())
    ]


def train_member(classifier_type, settings: Settings, x_train, y_train, x_val, y_val, seed: int):
    keras.backend.clear_session()
    set_seed(seed)
    member_dir = settings.output_dir / f"member_{seed}"
    member_dir.mkdir(parents=True, exist_ok=True)
    classifier = classifier_type(
        f"{member_dir}/", x_train.shape[1:], y_train.shape[1], batch_size=settings.batch_size,
        nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41,
        nb_epochs=settings.epochs,
    )
    callbacks = [
        keras.callbacks.CSVLogger(member_dir / "training.csv", append=False),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=settings.patience, mode="max", restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50, min_lr=0.0001),
    ]
    history = classifier.model.fit(
        x_train, y_train, validation_data=(x_val, y_val), batch_size=settings.batch_size,
        epochs=settings.epochs, callbacks=callbacks, verbose=2, shuffle=True,
    )
    classifier.model.save_weights(member_dir / "best_validation_model.h5")
    return classifier.model, history.history


def run(settings: Settings) -> None:
    if len(settings.member_seeds) != 5:
        raise ValueError("InceptionTime requires five independent member seeds")
    logger = create_logger(settings.output_dir)
    classifier_type = author_classifier(settings.source_dir)
    data = load_original_h5(settings.h5_path)
    x_all = prepare_windows(data["x_train"])
    y_all = np.asarray(data["y_train"], dtype=np.int64)
    classes = len(data["subjects"])
    y_categorical = keras.utils.to_categorical(y_all, classes)
    indices = np.random.RandomState(settings.member_seeds[0]).permutation(len(y_all))
    validation_count = round(len(indices) * settings.validation_fraction)
    val_idx, train_idx = indices[:validation_count], indices[validation_count:]
    tests = {condition: prepare_windows(windows) for condition, windows in data["x_test_by_cond"].items()}
    probabilities = {condition: [] for condition in tests}
    histories, parameters = {}, None
    logger.info("source=%s train=%s validation=%s classes=%s", AUTHOR_SOURCE, len(train_idx), len(val_idx), classes)
    for seed in settings.member_seeds:
        model, history = train_member(
            classifier_type, settings, x_all[train_idx], y_categorical[train_idx],
            x_all[val_idx], y_categorical[val_idx], seed,
        )
        parameters = model.count_params()
        histories[str(seed)] = history
        for condition, x_test in tests.items():
            probabilities[condition].append(model.predict(x_test, batch_size=settings.batch_size, verbose=0))
        logger.info("member=%s epochs=%s best_val_accuracy=%.6f", seed, len(history["loss"]), max(history["val_accuracy"]))
    rows, participants = [], []
    for condition in sorted(tests):
        labels = np.asarray(data["y_test_by_cond"][condition])
        predictions = average_five(probabilities[condition]).argmax(axis=1)
        metadata = data["meta_test_by_cond"][condition]
        row = {
            "model": MODEL_KEY, "condition": condition,
            "window_accuracy": float(accuracy_score(labels, predictions)),
            "window_macro_f1": float(f1_score(labels, predictions, average="macro")),
            "n_windows": len(labels), "members": 5, "parameters_per_member": parameters,
            "source": AUTHOR_SOURCE,
        }
        row.update(trial_metrics(labels, predictions, metadata))
        rows.append(row)
        participants.extend(subject_rows(condition, labels, predictions, metadata))
        logger.info("condition=%s window_accuracy=%.6f", condition, row["window_accuracy"])
    pd.DataFrame(rows).to_csv(settings.output_dir / "inceptiontime_ensemble_results.csv", index=False)
    pd.DataFrame(participants).to_csv(
        settings.output_dir / "inceptiontime_ensemble_subject_condition_accuracy.csv", index=False
    )
    write_metadata(settings, histories, rows)
    if len(rows) != 9 or len(participants) != 538:
        raise RuntimeError(f"incomplete output: conditions={len(rows)}, participant_rows={len(participants)}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    run(Settings(root / "data/altergait/gait_id_pelvis_62.h5", root / "external" / "InceptionTime",
                 root / "baselines_out" / "treadmill_published_subject_sd" / "inceptiontime_author_ensemble"))


if __name__ == "__main__":
    main()
