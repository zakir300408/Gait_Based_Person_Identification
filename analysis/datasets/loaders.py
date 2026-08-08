"""Load the released WHU-Gait and OU-ISIR train/test matrices."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
DATASETS = {
    "whugait_dataset3": {
        "root": ROOT / "data" / "whu_gait" / "Dataset #3" / "Dataset #3",
        "layout": "whugait",
        "paper": "Zou et al. WHU-Gait Dataset #3 split",
    },
    "ou_isir_identification": {
        "root": ROOT / "data" / "ou_isir" / "identification" / "data",
        "layout": "ou_isir",
        "paper": "Zou-preprocessed OU-ISIR identification split",
    },
}


def load_whugait_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    signal_dir = root / split / "Inertial Signals"
    arrays = [
        np.loadtxt(signal_dir / f"{split}_{channel}.txt", dtype=np.float32)
        for channel in CHANNELS
    ]
    labels = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    return np.stack(arrays, axis=1).astype(np.float32), labels


def load_ou_isir_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    split_dir = root / split
    signal_dir = split_dir / "data"
    arrays = [
        np.loadtxt(signal_dir / f"{split}_{channel}.txt", dtype=np.float32)
        for channel in CHANNELS
    ]
    labels = np.loadtxt(split_dir / f"{split}_y.txt", dtype=np.int64) - 1
    return np.stack(arrays, axis=1).astype(np.float32), labels


def load_dataset(dataset_id: str):
    if dataset_id not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset_id}")
    settings = DATASETS[dataset_id]
    root = settings["root"]
    if not root.is_dir():
        raise FileNotFoundError(f"dataset is missing: {root.relative_to(ROOT)}")
    loader = load_whugait_split if settings["layout"] == "whugait" else load_ou_isir_split
    x_train, y_train = loader(root, "train")
    x_test, y_test = loader(root, "test")
    return x_train, y_train, x_test, y_test, settings
