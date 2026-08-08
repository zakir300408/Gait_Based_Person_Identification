"""Five-fold subject-disjoint five-member InceptionTime verification."""

from __future__ import annotations

import random

import keras
import numpy as np

from ablation.neural_verification_common import (
    CONDITIONS, OUTPUT, ROOT, create_logger, load_data, save_audit, save_scores,
    subject_folds, subset, trial_rows,
)


MODEL = "InceptionTime"
KEY = "inceptiontime"
SEEDS = (42, 43, 44, 45, 46)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def prepare(windows: list[np.ndarray]) -> np.ndarray:
    x = np.stack(windows).astype(np.float32, copy=False)
    x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)
    return np.ascontiguousarray(x.transpose(0, 2, 1))


def inception_module(inputs):
    bottleneck = keras.layers.Conv1D(32, 1, padding="same", use_bias=False)(inputs)
    branches = [
        keras.layers.Conv1D(32, kernel, padding="same", use_bias=False)(bottleneck)
        for kernel in (40, 20, 10)
    ]
    pooled = keras.layers.MaxPool1D(3, strides=1, padding="same")(inputs)
    branches.append(keras.layers.Conv1D(32, 1, padding="same", use_bias=False)(pooled))
    x = keras.layers.Concatenate(axis=2)(branches)
    return keras.layers.Activation("relu")(keras.layers.BatchNormalization()(x))


def build_model(classes: int):
    inputs = keras.layers.Input((200, 6))
    x, residual = inputs, inputs
    for depth in range(6):
        x = inception_module(x)
        if depth % 3 == 2:
            shortcut = keras.layers.Conv1D(int(x.shape[-1]), 1, padding="same", use_bias=False)(residual)
            shortcut = keras.layers.BatchNormalization()(shortcut)
            x = keras.layers.Activation("relu")(keras.layers.Add()([shortcut, x]))
            residual = x
    outputs = keras.layers.Dense(classes, activation="softmax")(
        keras.layers.GlobalAveragePooling1D()(x)
    )
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_members(x, y, classes: int, fold: int):
    order = np.random.RandomState(SEEDS[0]).permutation(len(y))
    count = round(len(order) * 0.15)
    validation, train = order[:count], order[count:]
    members, histories = [], []
    for seed in SEEDS:
        keras.backend.clear_session()
        set_seed(seed + fold)
        model = build_model(classes)
        history = model.fit(
            x[train], keras.utils.to_categorical(y[train], classes),
            validation_data=(x[validation], keras.utils.to_categorical(y[validation], classes)),
            batch_size=64, epochs=60, shuffle=True, verbose=2,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=12, mode="max", restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                ),
            ],
        )
        fold_dir = OUTPUT / KEY / f"fold_{fold}" / f"member_{seed}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model.save_weights(fold_dir / "model.weights.h5")
        members.append(model)
        histories.append(len(history.history["loss"]))
    return members, histories


def predict(members, windows: list[np.ndarray]) -> np.ndarray:
    x = prepare(windows)
    return np.mean([model.predict(x, batch_size=64, verbose=0) for model in members], axis=0)


def main() -> None:
    data, logger, open_rows = load_data(), create_logger(KEY), []
    for fold, enrolled, impostors in subject_folds(data):
        split = subset(data, enrolled, impostors)
        members, histories = train_members(
            prepare(split["train_x"]), split["train_y"], len(enrolled), fold
        )
        labels = np.arange(len(enrolled), dtype=np.int64)
        for condition in CONDITIONS:
            gx, gy, gm = split["genuine"][condition]
            open_rows += trial_rows(
                MODEL, fold, condition, predict(members, gx), gm, labels, np.asarray(gy)
            )
        for condition, (ix, im) in split["impostor"].items():
            open_rows += trial_rows(
                MODEL, fold, condition, predict(members, ix), im, labels, None
            )
        logger.info("fold=%s enrolled=%s impostors=%s epochs=%s rows=%s",
                    fold, len(enrolled), len(impostors), histories, len(open_rows))
    open_path = save_scores(KEY, "open_set", open_rows)

    members = []
    for seed in SEEDS:
        model = build_model(len(data["subjects"]))
        model.load_weights(
            ROOT / f"baselines_out/treadmill_published_subject_sd/inceptiontime_author_ensemble/member_{seed}/best_validation_model.h5"
        )
        members.append(model)
    enrolled_rows, labels = [], np.arange(len(data["subjects"]), dtype=np.int64)
    for condition in CONDITIONS:
        enrolled_rows += trial_rows(
            MODEL, "", condition, predict(members, data["x_test_by_cond"][condition]),
            data["meta_test_by_cond"][condition], labels,
            np.asarray(data["y_test_by_cond"][condition]), enrolled_protocol=True,
        )
    enrolled_path = save_scores(KEY, "enrolled", enrolled_rows)
    save_audit(KEY, {"model": MODEL, "folds": 5, "seed": 42, "members": list(SEEDS),
                    "open_scores": str(open_path), "enrolled_scores": str(enrolled_path)})


if __name__ == "__main__":
    main()
