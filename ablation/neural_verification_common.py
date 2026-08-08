"""Shared, leakage-free score export for neural treadmill verification."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ablation.pelvis_harmonic_covariance_model import load_original_h5


ROOT = Path(__file__).resolve().parents[1]
H5_PATH = ROOT / "gait_id_pelvis_62.h5"
OUTPUT = ROOT / "baselines_out" / "neural_open_set_verification_treadmill"
CONDITIONS = (
    "fpa_large", "fpa_small", "step_width_large", "step_width_small",
    "trunk_sway_large", "trunk_sway_small", "walk_speed_large",
    "walk_speed_small", "dual_task",
)
SEED = 42
FOLDS = 5
SCORE_COLUMNS = (
    "dataset", "model", "fold", "group", "subject", "condition", "trial",
    "claim", "genuine", "score",
)


def create_logger(model_key: str) -> logging.Logger:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"{model_key}_{stamp}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(OUTPUT / f"{model_key}_{stamp}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def load_data() -> dict:
    data = load_original_h5(H5_PATH)
    if len(data["subjects"]) != 62 or tuple(sorted(data["x_test_by_cond"])) != tuple(sorted(CONDITIONS)):
        raise ValueError("unexpected treadmill dataset identity or condition inventory")
    return data


def subject_folds(data: dict) -> list[tuple[int, list[str], list[str]]]:
    subjects = np.asarray(sorted(data["subjects"]))
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    return [
        (fold, subjects[train].tolist(), subjects[test].tolist())
        for fold, (train, test) in enumerate(splitter.split(subjects), 1)
    ]


def subset(data: dict, enrolled: list[str], impostors: list[str]) -> dict:
    enrolled_set, impostor_set = set(enrolled), set(impostors)
    label = {subject: index for index, subject in enumerate(sorted(enrolled))}
    result = {
        "train_x": [], "train_y": [], "genuine": {}, "impostor": {},
    }
    for window, meta in zip(data["x_train"], data["train_meta"]):
        subject = meta["subject"]
        if subject in enrolled_set:
            result["train_x"].append(window)
            result["train_y"].append(label[subject])
        elif subject in impostor_set:
            result["impostor"].setdefault("baseline", [[], []])
            result["impostor"]["baseline"][0].append(window)
            result["impostor"]["baseline"][1].append(meta)
    for condition in CONDITIONS:
        genuine = [[], [], []]
        impostor = [[], []]
        for window, target, meta in zip(
            data["x_test_by_cond"][condition],
            data["y_test_by_cond"][condition],
            data["meta_test_by_cond"][condition],
        ):
            subject = meta["subject"]
            if subject in enrolled_set:
                genuine[0].append(window)
                genuine[1].append(label[subject])
                genuine[2].append(meta)
            elif subject in impostor_set:
                impostor[0].append(window)
                impostor[1].append(meta)
        result["genuine"][condition] = genuine
        result["impostor"][condition] = impostor
    result["train_y"] = np.asarray(result["train_y"], dtype=np.int64)
    return result


def group_name(condition: str) -> str:
    return "trunk_sway_only" if condition.startswith("trunk_sway_") else "non_sway"


def trial_rows(
    model: str,
    fold: int | str,
    condition: str,
    probabilities: np.ndarray,
    metadata: list[dict],
    labels: np.ndarray,
    truth: np.ndarray | None,
    enrolled_protocol: bool = False,
) -> list[dict]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    targets: dict[str, int] = {}
    trial_meta = {}
    for index, (row, meta) in enumerate(zip(probabilities, metadata)):
        key = f"{meta['subject']}|{meta['condition']}|{meta['trial']}"
        grouped[key].append(row)
        trial_meta[key] = meta
        if truth is not None:
            targets[key] = int(truth[index])
    rows = []
    label_to_column = {int(value): index for index, value in enumerate(labels)}
    for key, score_rows in grouped.items():
        scores = np.vstack(score_rows).mean(axis=0)
        meta = trial_meta[key]
        base = {
            "dataset": "original_treadmill", "model": model, "fold": fold,
            "group": group_name(condition), "subject": meta["subject"],
            "condition": condition, "trial": meta["trial"],
        }
        if truth is not None and not enrolled_protocol:
            claim = targets[key]
            rows.append({**base, "claim": claim, "genuine": 1,
                         "score": float(scores[label_to_column[claim]])})
        else:
            target = targets.get(key)
            for column, claim in enumerate(labels):
                rows.append({**base, "claim": int(claim),
                             "genuine": int(target == int(claim)),
                             "score": float(scores[column])})
    return rows


def save_scores(model_key: str, kind: str, rows: list[dict]) -> Path:
    path = OUTPUT / f"{model_key}_{kind}_trial_scores.csv"
    frame = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    if frame.empty or frame["score"].isna().any():
        raise ValueError(f"invalid {model_key} {kind} score export")
    frame.to_csv(path, index=False)
    return path


def save_audit(model_key: str, payload: dict) -> Path:
    path = OUTPUT / f"{model_key}_audit.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
