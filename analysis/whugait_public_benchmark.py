"""
Public whuGAIT Dataset #3 benchmark for MB-Lite-style IMU gait recognition.

Dataset #3 is the 118-subject, fixed-window whuGAIT identification split used
by recent wearable gait-recognition papers. Each sample is six IMU channels
(3-axis acceleration + 3-axis angular velocity) over 128 samples at 50 Hz.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pelvis_kernel_duel_model import PelvisKernelDuelFeatures

SEED = 42
FS_HZ = 50
DEFAULT_ROOT = Path("public_datasets/whugait/Dataset #3/Dataset #3")
OUT_DIR = Path("baselines_out/public_whugait_dataset3")
CHANNELS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_split(root, split):
    sig_dir = root / split / "Inertial Signals"
    arrays = []
    for ch in CHANNELS:
        path = sig_dir / f"{split}_{ch}.txt"
        arrays.append(np.loadtxt(path, dtype=np.float32))
    x = np.stack(arrays, axis=1).astype(np.float32)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    return x, y


def maybe_subsample(x, y, max_n, seed):
    if not max_n or max_n >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_n, replace=False)
    idx.sort()
    return x[idx], y[idx]


def zscore_windows(x):
    return ((x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)).astype(np.float32)


def import_minirocket():
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    return MiniRocketMultivariate


def import_multirocket():
    from sktime.transformations.panel.rocket import MultiRocketMultivariate

    return MultiRocketMultivariate


def as_float32(mat):
    return np.asarray(mat, dtype=np.float32)


def branches():
    return {
        "raw": lambda x: x,
        "diff": lambda x: np.diff(x, axis=2),
        "diff2": lambda x: np.diff(x, n=2, axis=2),
    }


def fit_branch_minirocket(x_train, y_train, kernels, seed):
    MiniRocket = import_minirocket()
    models = {}
    feats = []
    for i, (name, fn) in enumerate(branches().items()):
        n_k = int(kernels[name])
        rocket = MiniRocket(num_kernels=n_k, random_state=seed + i, n_jobs=-1)
        xb = np.ascontiguousarray(fn(x_train), dtype=np.float64)
        rocket.fit(xb, y_train)
        feats.append(as_float32(rocket.transform(xb)))
        models[name] = rocket
    return models, np.concatenate(feats, axis=1)


def transform_branch_minirocket(models, x):
    feats = []
    for name, fn in branches().items():
        xb = np.ascontiguousarray(fn(x), dtype=np.float64)
        feats.append(as_float32(models[name].transform(xb)))
    return np.concatenate(feats, axis=1)


def harmonic_fingerprint_one(x, fs_hz=FS_HZ):
    feats = []
    dx = np.diff(x, axis=1, prepend=x[:, :1])
    ddx = np.diff(dx, axis=1, prepend=dx[:, :1])
    for block in (x, dx, ddx):
        feats.extend(block.mean(axis=1).tolist())
        feats.extend(block.std(axis=1).tolist())
        feats.extend(np.percentile(block, [5, 25, 50, 75, 95], axis=1).T.reshape(-1).tolist())
    feats.extend(np.corrcoef(x).reshape(-1).tolist())
    cov = np.cov(x) + np.eye(x.shape[0]) * 1e-5
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-8, None)
    log_cov = (vecs * np.log(vals)) @ vecs.T
    feats.extend(log_cov[np.triu_indices(x.shape[0])].tolist())

    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / fs_hz)
    spec = np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)
    amp = np.abs(spec)
    bands = ((0.3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 15.0), (15.0, 24.0))
    for ch in range(x.shape[0]):
        p = amp[ch] ** 2 + 1e-12
        feats.append(float((freqs * p).sum() / p.sum()))
        feats.append(float(freqs[np.argmax(p[1:]) + 1] if len(p) > 1 else 0.0))
        total = p.sum()
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            feats.append(float(p[mask].sum() / total))
    return np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def harmonic_fingerprint(x):
    return np.vstack([harmonic_fingerprint_one(win) for win in x]).astype(np.float32)


def make_classifier():
    return Pipeline(
        [
            ("norm", Normalizer()),
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", RidgeClassifierCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
        ]
    )


def score_predictions(model, y_true, y_pred, elapsed, extra=None):
    row = {
        "model": model,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "top5_accuracy": "",
        "n_train": "",
        "n_test": int(len(y_true)),
        "feature_count": "",
        "params": "",
        "elapsed_seconds": float(elapsed),
        "notes": "",
    }
    if extra:
        row.update(extra)
    return row


def score_estimator(model_name, clf, x_train_feat, y_train, x_test_feat, y_test, elapsed, extra=None):
    pred = clf.predict(x_test_feat)
    row = score_predictions(model_name, y_test, pred, elapsed, extra)
    row["n_train"] = int(len(y_train))
    if hasattr(clf, "decision_function"):
        try:
            scores = clf.decision_function(x_test_feat)
            row["top5_accuracy"] = float(top_k_accuracy_score(y_test, scores, k=5, labels=np.arange(len(np.unique(y_train)))))
        except Exception:
            pass
    return row


def run_minirocket(x_train, y_train, x_test, y_test, seed, kernels):
    t0 = perf_counter()
    MiniRocket = import_minirocket()
    rocket = MiniRocket(num_kernels=kernels, random_state=seed, n_jobs=-1)
    rocket.fit(np.ascontiguousarray(x_train, dtype=np.float64), y_train)
    tr = as_float32(rocket.transform(np.ascontiguousarray(x_train, dtype=np.float64)))
    te = as_float32(rocket.transform(np.ascontiguousarray(x_test, dtype=np.float64)))
    clf = make_classifier()
    clf.fit(tr, y_train)
    return score_estimator(
        "MiniROCKET + ridge",
        clf,
        tr,
        y_train,
        te,
        y_test,
        perf_counter() - t0,
        {"feature_count": int(tr.shape[1])},
    )


def run_multirocket(x_train, y_train, x_test, y_test, seed, kernels):
    t0 = perf_counter()
    MultiRocket = import_multirocket()
    rocket = MultiRocket(num_kernels=kernels, random_state=seed, n_jobs=-1)
    rocket.fit(np.ascontiguousarray(x_train, dtype=np.float64), y_train)
    tr = as_float32(rocket.transform(np.ascontiguousarray(x_train, dtype=np.float64)))
    te = as_float32(rocket.transform(np.ascontiguousarray(x_test, dtype=np.float64)))
    clf = make_classifier()
    clf.fit(tr, y_train)
    return score_estimator(
        "MultiROCKET + ridge",
        clf,
        tr,
        y_train,
        te,
        y_test,
        perf_counter() - t0,
        {"feature_count": int(tr.shape[1])},
    )


def run_mblite(x_train, y_train, x_test, y_test, args):
    t0 = perf_counter()
    kernels = {"raw": args.raw_kernels, "diff": args.diff_kernels, "diff2": args.diff2_kernels}
    models, tr_rocket = fit_branch_minirocket(x_train, y_train, kernels, args.seed)
    te_rocket = transform_branch_minirocket(models, x_test)
    train_blocks = [tr_rocket, harmonic_fingerprint(x_train)]
    test_blocks = [te_rocket, harmonic_fingerprint(x_test)]
    if args.add_kernel_duel:
        duel = PelvisKernelDuelFeatures(
            groups=args.duel_groups,
            kernels_per_group=args.duel_kernels_per_group,
            kernel_size=9,
            dilations=(1, 2, 4, 8),
            seed=args.seed,
            batch_size=args.batch_size,
        ).fit(x_train)
        train_blocks.append(duel.transform(x_train))
        test_blocks.append(duel.transform(x_test))
    tr = np.hstack(train_blocks).astype(np.float32)
    te = np.hstack(test_blocks).astype(np.float32)
    clf = make_classifier()
    clf.fit(tr, y_train)
    return score_estimator(
        "MB-Lite public fusion",
        clf,
        tr,
        y_train,
        te,
        y_test,
        perf_counter() - t0,
        {"feature_count": int(tr.shape[1]), "notes": f"kernels={kernels}; kernel_duel={int(args.add_kernel_duel)}"},
    )


class WindowDataset:
    def __init__(self, x, y):
        import torch

        self.x = torch.from_numpy(zscore_windows(x)).float()
        self.y = torch.from_numpy(y.astype(np.int64)).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_torch_model(model_name, model, x_train, y_train, x_test, y_test, args, note):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    t0 = perf_counter()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    train_ds = WindowDataset(x_train, y_train)
    test_ds = WindowDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_state = None
    best_acc = -1.0
    patience_left = args.patience
    for _epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _yb in test_loader:
                preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
        pred = np.concatenate(preds)
        acc = accuracy_score(y_test, pred)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    logits = []
    with torch.no_grad():
        for xb, _yb in test_loader:
            logits.append(model(xb.to(device)).cpu().numpy())
    scores = np.vstack(logits)
    pred = scores.argmax(axis=1)
    row = score_predictions(
        model_name,
        y_test,
        pred,
        perf_counter() - t0,
        {
            "n_train": int(len(y_train)),
            "params": count_params(model),
            "top5_accuracy": float(top_k_accuracy_score(y_test, scores, k=5, labels=np.arange(len(np.unique(y_train))))),
            "notes": f"{note}; device={device}; epochs={args.epochs}",
        },
    )
    return row


def run_resnet1d(x_train, y_train, x_test, y_test, args):
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, inc, outc):
            super().__init__()
            self.conv1 = nn.Conv1d(inc, outc, 7, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(outc)
            self.conv2 = nn.Conv1d(outc, outc, 5, padding=2, bias=False)
            self.bn2 = nn.BatchNorm1d(outc)
            self.short = nn.Identity() if inc == outc else nn.Sequential(nn.Conv1d(inc, outc, 1, bias=False), nn.BatchNorm1d(outc))

        def forward(self, x):
            y = F.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            return F.relu(y + self.short(x))

    class ResNetSmall(nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.net = nn.Sequential(Block(6, 64), Block(64, 96), Block(96, 128))
            self.head = nn.Linear(128, classes)

        def forward(self, x):
            y = self.net(x).mean(dim=-1)
            return self.head(y)

    return train_torch_model("ResNet1D public", ResNetSmall(len(np.unique(y_train))), x_train, y_train, x_test, y_test, args, "standard local ResNet1D")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--models", default="mblite,minirocket,resnet1d")
    parser.add_argument("--seed", type=int, default=SEED)
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
    x_train, y_train = load_split(args.root, "train")
    x_test, y_test = load_split(args.root, "test")
    x_train, y_train = maybe_subsample(x_train, y_train, args.max_train, args.seed)
    x_test, y_test = maybe_subsample(x_test, y_test, args.max_test, args.seed + 1)
    print(
        f"[Data] whuGAIT Dataset #3 train={x_train.shape} test={x_test.shape} "
        f"classes={len(np.unique(y_train))} labels={int(y_train.min()) + 1}-{int(y_train.max()) + 1}",
        flush=True,
    )

    selected = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    rows = []
    if "mblite" in selected:
        print("[Run] MB-Lite public fusion", flush=True)
        rows.append(run_mblite(x_train, y_train, x_test, y_test, args))
    if "minirocket" in selected:
        print("[Run] MiniROCKET", flush=True)
        rows.append(run_minirocket(x_train, y_train, x_test, y_test, args.seed, args.minirocket_kernels))
    if "multirocket" in selected:
        print("[Run] MultiROCKET", flush=True)
        rows.append(run_multirocket(x_train, y_train, x_test, y_test, args.seed, args.multirocket_kernels))
    if "resnet1d" in selected:
        print("[Run] ResNet1D", flush=True)
        rows.append(run_resnet1d(x_train, y_train, x_test, y_test, args))
    if "mha_cnn" in selected or "mha-cnn" in selected:
        raise ValueError(
            "mha_cnn was a local style approximation, not released author code. "
            "Use exact_sota_trunksway_benchmark.py or an official implementation instead."
        )

    rows = sorted(rows, key=lambda r: r["accuracy"], reverse=True)
    fields = [
        "model",
        "accuracy",
        "macro_f1",
        "top5_accuracy",
        "n_train",
        "n_test",
        "feature_count",
        "params",
        "elapsed_seconds",
        "notes",
    ]
    csv_path = args.out_dir / "whugait_dataset3_public_benchmark.csv"
    json_path = args.out_dir / "whugait_dataset3_public_benchmark.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args) | {"root": str(args.root), "out_dir": str(args.out_dir)}, "rows": rows}, f, indent=2)
    print(f"[Saved] {csv_path}", flush=True)
    for i, row in enumerate(rows, 1):
        print(f"{i}. {row['model']}: acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} top5={row['top5_accuracy']}", flush=True)


if __name__ == "__main__":
    main()
