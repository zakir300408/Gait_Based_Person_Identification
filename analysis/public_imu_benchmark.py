"""
Run MB-Lite and standard baselines on downloaded public IMU gait datasets.

Supported local datasets:
  - whuGAIT Dataset #1, #2, #3
  - Zou-preprocessed OU-ISIR identification split

The script reuses the model runners from whugait_public_benchmark.py but adds a
dataset registry and loaders so public benchmark tables can be regenerated from
one command.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from whugait_public_benchmark import (
    run_mblite,
    run_minirocket,
    run_multirocket,
    run_resnet1d,
    set_seed,
)

CHANNELS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
OUT_DIR = Path("baselines_out/public_imu_benchmark")

DATASETS = {
    "whugait_dataset1": {
        "root": Path("public_datasets/whugait/Dataset #1"),
        "layout": "whugait",
        "paper": "Zou2020 / Rifaat2022",
    },
    "whugait_dataset2": {
        "root": Path("public_datasets/whugait/Dataset #2"),
        "layout": "whugait",
        "paper": "Zou2020 / Rifaat2022",
    },
    "whugait_dataset3": {
        "root": Path("public_datasets/whugait/Dataset #3/Dataset #3"),
        "layout": "whugait",
        "paper": "He2026 exact public whuGAIT split",
    },
    "ou_isir_identification": {
        "root": Path("public_datasets/whugait/OU-ISIR_sensors_dataset/identification/data"),
        "layout": "ou_isir",
        "paper": "Zou2020-preprocessed OU-ISIR / He2026",
    },
}


def load_whugait_split(root, split):
    sig_dir = root / split / "Inertial Signals"
    arrays = [np.loadtxt(sig_dir / f"{split}_{ch}.txt", dtype=np.float32) for ch in CHANNELS]
    x = np.stack(arrays, axis=1).astype(np.float32)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    return x, y


def load_ou_isir_split(root, split):
    split_dir = root / split
    sig_dir = split_dir / "data"
    arrays = [np.loadtxt(sig_dir / f"{split}_{ch}.txt", dtype=np.float32) for ch in CHANNELS]
    x = np.stack(arrays, axis=1).astype(np.float32)
    y = np.loadtxt(split_dir / f"{split}_y.txt", dtype=np.int64) - 1
    return x, y


def load_dataset(dataset_id):
    cfg = DATASETS[dataset_id]
    if cfg["layout"] == "whugait":
        x_train, y_train = load_whugait_split(cfg["root"], "train")
        x_test, y_test = load_whugait_split(cfg["root"], "test")
    elif cfg["layout"] == "ou_isir":
        x_train, y_train = load_ou_isir_split(cfg["root"], "train")
        x_test, y_test = load_ou_isir_split(cfg["root"], "test")
    else:
        raise ValueError(f"Unknown layout: {cfg['layout']}")
    return x_train, y_train, x_test, y_test, cfg


def maybe_subsample(x, y, max_n, seed):
    if not max_n or max_n >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_n, replace=False)
    idx.sort()
    return x[idx], y[idx]


def run_one_dataset(dataset_id, args):
    x_train, y_train, x_test, y_test, cfg = load_dataset(dataset_id)
    x_train, y_train = maybe_subsample(x_train, y_train, args.max_train, args.seed)
    x_test, y_test = maybe_subsample(x_test, y_test, args.max_test, args.seed + 1)
    print(
        f"[Data] {dataset_id} train={x_train.shape} test={x_test.shape} "
        f"classes={len(np.unique(y_train))} source={cfg['paper']}",
        flush=True,
    )

    selected = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    rows = []
    if "mblite" in selected:
        print(f"[Run:{dataset_id}] MB-Lite", flush=True)
        rows.append(run_mblite(x_train, y_train, x_test, y_test, args))
    if "minirocket" in selected:
        print(f"[Run:{dataset_id}] MiniROCKET", flush=True)
        rows.append(run_minirocket(x_train, y_train, x_test, y_test, args.seed, args.minirocket_kernels))
    if "multirocket" in selected:
        print(f"[Run:{dataset_id}] MultiROCKET", flush=True)
        rows.append(run_multirocket(x_train, y_train, x_test, y_test, args.seed, args.multirocket_kernels))
    if "resnet1d" in selected:
        print(f"[Run:{dataset_id}] ResNet1D", flush=True)
        rows.append(run_resnet1d(x_train, y_train, x_test, y_test, args))
    if "mha_cnn" in selected or "mha-cnn" in selected:
        raise ValueError(
            "mha_cnn was a local style approximation, not released author code. "
            "Use exact_sota_trunksway_benchmark.py or an official implementation instead."
        )

    for row in rows:
        row["dataset"] = dataset_id
        row["source_protocol"] = cfg["paper"]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="whugait_dataset3,ou_isir_identification")
    parser.add_argument("--models", default="mblite,minirocket,resnet1d")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--raw-kernels", type=int, default=3000)
    parser.add_argument("--diff-kernels", type=int, default=3000)
    parser.add_argument("--diff2-kernels", type=int, default=2000)
    parser.add_argument("--minirocket-kernels", type=int, default=10000)
    parser.add_argument("--multirocket-kernels", type=int, default=5000)
    parser.add_argument("--add-kernel-duel", action="store_true")
    parser.add_argument("--duel-groups", type=int, default=64)
    parser.add_argument("--duel-kernels-per-group", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_id in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if dataset_id not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset_id}; choose from {sorted(DATASETS)}")
        rows.extend(run_one_dataset(dataset_id, args))
    rows = sorted(rows, key=lambda r: (r["dataset"], -float(r["accuracy"])))

    fields = [
        "dataset",
        "model",
        "accuracy",
        "macro_f1",
        "top5_accuracy",
        "n_train",
        "n_test",
        "feature_count",
        "params",
        "elapsed_seconds",
        "source_protocol",
        "notes",
    ]
    csv_path = args.out_dir / "public_imu_benchmark.csv"
    json_path = args.out_dir / "public_imu_benchmark.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args) | {"out_dir": str(args.out_dir)}, "rows": rows}, f, indent=2)
    print(f"[Saved] {csv_path}", flush=True)
    for row in rows:
        print(f"{row['dataset']} | {row['model']} | acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
