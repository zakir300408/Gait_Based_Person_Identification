"""
PelvisKernelDuel: compact pelvis-only competing-kernel model.

This is a lightweight random-kernel dictionary inspired by recent HYDRA-style
time-series classifiers: groups of short multivariate kernels compete over
time, and each window is represented by winner counts plus peak responses.
It uses the same closed-set protocol as MB-3.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pelvis_harmonic_covariance_model import H5_PATH, KYOTO_ROOT, featurize as phcf_featurize, load_kyoto, load_original_h5

OUT_DIR = Path("baselines_out/pelvis_kernel_duel")
SEED = 42


def stack(windows):
    return np.stack(windows, axis=0).astype(np.float32)


def augment_windows(x, y, copies, seed):
    if copies <= 0:
        return x, y
    rng = np.random.default_rng(seed)
    xs = [x]
    ys = [y]
    t = np.linspace(0.0, 1.0, x.shape[2], dtype=np.float32)
    for _ in range(copies):
        xa = x.copy()
        xa *= rng.uniform(0.85, 1.15, size=(x.shape[0], x.shape[1], 1)).astype(np.float32)
        xa += rng.normal(0, rng.uniform(0.01, 0.04), size=xa.shape).astype(np.float32)
        for i in range(xa.shape[0]):
            xa[i] = np.roll(xa[i], int(rng.integers(-14, 15)), axis=1)
            if rng.random() < 0.55:
                width = int(rng.integers(8, 28))
                start = int(rng.integers(0, max(1, xa.shape[2] - width + 1)))
                xa[i, :, start : start + width] = 0.0
            if rng.random() < 0.65:
                for block in (slice(0, 3), slice(3, 6)):
                    mix = np.eye(3, dtype=np.float32) + rng.normal(0, 0.04, size=(3, 3)).astype(np.float32)
                    xa[i, block, :] = mix @ xa[i, block, :]
            if rng.random() < 0.75:
                freq = rng.uniform(0.45, 1.6)
                phase = rng.uniform(0, 2 * np.pi)
                wave = np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
                amp = rng.uniform(0.04, 0.22)
                xa[i, 1, :] += amp * wave
                xa[i, 4, :] += amp * rng.uniform(0.5, 1.8) * wave
        xs.append(xa)
        ys.append(y.copy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def add_pelvis_invariants(x):
    acc = x[:, :3, :]
    gyr = x[:, 3:6, :]
    acc_norm = np.sqrt(np.sum(acc ** 2, axis=1, keepdims=True))
    gyr_norm = np.sqrt(np.sum(gyr ** 2, axis=1, keepdims=True))
    acc_xy = np.sqrt(np.sum(acc[:, :2, :] ** 2, axis=1, keepdims=True))
    gyr_xy = np.sqrt(np.sum(gyr[:, :2, :] ** 2, axis=1, keepdims=True))
    jerk = np.diff(acc, axis=2, prepend=acc[:, :, :1])
    ang_jerk = np.diff(gyr, axis=2, prepend=gyr[:, :, :1])
    jerk_norm = np.sqrt(np.sum(jerk ** 2, axis=1, keepdims=True))
    ang_jerk_norm = np.sqrt(np.sum(ang_jerk ** 2, axis=1, keepdims=True))
    derived = np.concatenate([acc_norm, gyr_norm, acc_xy, gyr_xy, jerk_norm, ang_jerk_norm], axis=1)
    derived = (derived - derived.mean(axis=2, keepdims=True)) / (derived.std(axis=2, keepdims=True) + 1e-8)
    return np.concatenate([x, derived.astype(np.float32)], axis=1)


def phase_cycle_features_one(win, bins=16, fs_hz=100):
    x = np.asarray(win, dtype=np.float32)
    svm = np.sqrt(np.sum(x[:3] ** 2, axis=0))
    n = svm.shape[0]
    freqs = np.fft.rfftfreq(n, 1.0 / fs_hz)
    spec = np.fft.rfft((svm - svm.mean()) * np.hanning(n))
    mask = (freqs >= 0.5) & (freqs <= 3.0)
    if np.any(mask):
        idxs = np.where(mask)[0]
        dom = idxs[int(np.argmax(np.abs(spec[idxs])))]
        phase0 = np.angle(spec[dom])
        phase = (2 * np.pi * freqs[dom] * np.arange(n) / fs_hz + phase0) % (2 * np.pi)
    else:
        phase = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bin_id = np.minimum((phase / (2 * np.pi) * bins).astype(int), bins - 1)
    dx = np.diff(x, axis=1, prepend=x[:, :1])
    feats = []
    centers = []
    for b in range(bins):
        m = bin_id == b
        if not np.any(m):
            feats.extend([0.0] * (x.shape[0] * 4))
            centers.append(np.zeros(x.shape[0], dtype=np.float32))
            continue
        xb = x[:, m]
        db = dx[:, m]
        mu = xb.mean(axis=1)
        centers.append(mu)
        feats.extend(mu.tolist())
        feats.extend(xb.std(axis=1).tolist())
        feats.extend(db.mean(axis=1).tolist())
        feats.extend(db.std(axis=1).tolist())
    centers = np.stack(centers, axis=0)
    diffs = np.diff(np.vstack([centers, centers[:1]]), axis=0)
    feats.extend(diffs.reshape(-1).tolist())
    # Low-order closed-cycle Fourier descriptors of the phase-normalized orbit.
    for ch in range(x.shape[0]):
        curve = centers[:, ch]
        fft = np.fft.rfft(curve - curve.mean())
        for k in range(1, min(5, len(fft))):
            feats.extend([float(fft[k].real), float(fft[k].imag), float(np.abs(fft[k]))])
    out = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def phase_cycle_featurize(windows):
    return np.vstack([phase_cycle_features_one(w) for w in windows]).astype(np.float32)


def canonicalize_pelvis_windows(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        acc = x[i, :3, :]
        gyr = x[i, 3:6, :]
        cov = acc @ acc.T / max(1, acc.shape[1] - 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        basis = vecs[:, order].astype(np.float32)
        for j in range(3):
            pivot = int(np.argmax(np.abs(basis[:, j])))
            if basis[pivot, j] < 0:
                basis[:, j] *= -1
        if np.linalg.det(basis) < 0:
            basis[:, -1] *= -1
        rot = basis.T
        out[i, :3, :] = rot @ acc
        out[i, 3:6, :] = rot @ gyr
    return out


def remove_slow_lateral_sway(x, width=51):
    out = x.copy()
    kernel = np.ones(width, dtype=np.float32) / float(width)
    pad = width // 2
    for ch in (1, 4):
        padded = np.pad(out[:, ch, :], ((0, 0), (pad, pad)), mode="reflect")
        slow = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
        out[:, ch, :] = out[:, ch, :] - slow
    out = (out - out.mean(axis=2, keepdims=True)) / (out.std(axis=2, keepdims=True) + 1e-8)
    return out.astype(np.float32)


class PelvisKernelDuelFeatures:
    def __init__(self, groups=96, kernels_per_group=8, kernel_size=9, dilations=(1, 2, 4, 8), seed=42, batch_size=256):
        self.groups = groups
        self.kernels_per_group = kernels_per_group
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.seed = seed
        self.batch_size = batch_size
        self.weights = []

    def fit(self, x):
        rng = np.random.default_rng(self.seed)
        self.weights = []
        out_ch = self.groups * self.kernels_per_group
        for _ in self.dilations:
            w = rng.normal(0, 1, size=(out_ch, x.shape[1], self.kernel_size)).astype(np.float32)
            # Randomly drop some channels per kernel so kernels specialize.
            mask = rng.random(size=(out_ch, x.shape[1], 1)) < rng.uniform(0.35, 0.9, size=(out_ch, 1, 1))
            w *= mask.astype(np.float32)
            w -= w.mean(axis=(1, 2), keepdims=True)
            w /= np.linalg.norm(w.reshape(out_ch, -1), axis=1, keepdims=True).reshape(out_ch, 1, 1) + 1e-8
            self.weights.append(torch.from_numpy(w))
        return self

    def _features_batch(self, xb, device):
        feats = []
        xt = torch.from_numpy(xb).to(device)
        for dilation, weight_cpu in zip(self.dilations, self.weights):
            weight = weight_cpu.to(device)
            pad = dilation * (self.kernel_size // 2)
            resp = F.conv1d(xt, weight, padding=pad, dilation=dilation)
            resp = resp[:, :, : xt.shape[-1]]
            resp = resp.view(resp.shape[0], self.groups, self.kernels_per_group, resp.shape[-1])

            # Positive and negative competitions capture shape and inverse shape.
            for signed in (resp, -resp):
                winners = signed.argmax(dim=2)  # N, groups, time
                counts = []
                for k in range(self.kernels_per_group):
                    counts.append((winners == k).float().mean(dim=2))
                counts = torch.stack(counts, dim=2).reshape(resp.shape[0], -1)
                peaks = signed.amax(dim=3).reshape(resp.shape[0], -1)
                feats.extend([counts, peaks])
        return torch.cat(feats, dim=1).cpu().numpy().astype(np.float32)

    def transform(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        outs = []
        for start in range(0, len(x), self.batch_size):
            outs.append(self._features_batch(x[start : start + self.batch_size], device))
        return np.vstack(outs)


class CosinePrototypeClassifier:
    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        self.classes_ = np.asarray(sorted(set(y.tolist())))
        centroids = []
        for cls in self.classes_:
            c = x[y == cls].mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-12)
            centroids.append(c)
        self.centroids_ = np.vstack(centroids).astype(np.float32)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        scores = x @ self.centroids_.T
        return self.classes_[np.argmax(scores, axis=1)]


class DiagMahalanobisPrototypeClassifier:
    def __init__(self, shrink=0.6):
        self.shrink = shrink

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        self.classes_ = np.asarray(sorted(set(y.tolist())))
        global_var = x.var(axis=0) + 1e-4
        means = []
        inv_vars = []
        for cls in self.classes_:
            xc = x[y == cls]
            means.append(xc.mean(axis=0))
            cls_var = xc.var(axis=0) + 1e-4
            var = self.shrink * global_var + (1.0 - self.shrink) * cls_var
            inv_vars.append(1.0 / var)
        self.means_ = np.vstack(means).astype(np.float32)
        self.inv_vars_ = np.vstack(inv_vars).astype(np.float32)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        preds = []
        for start in range(0, len(x), 256):
            xb = x[start : start + 256]
            diff = xb[:, None, :] - self.means_[None, :, :]
            score = -np.mean(diff * diff * self.inv_vars_[None, :, :], axis=2)
            preds.extend(self.classes_[np.argmax(score, axis=1)].tolist())
        return np.asarray(preds)


def trial_metrics(y_true, y_pred, metas):
    by_trial_pred = defaultdict(list)
    by_trial_true = {}
    by_trial_subject = {}
    for yt, yp, meta in zip(y_true, y_pred, metas):
        key = f"{meta['subject']}|{meta['condition']}|{meta['trial']}"
        by_trial_pred[key].append(int(yp))
        by_trial_true[key] = int(yt)
        by_trial_subject[key] = meta["subject"]
    ys, ps, subjects = [], [], []
    for key, preds in by_trial_pred.items():
        ys.append(by_trial_true[key])
        ps.append(Counter(preds).most_common(1)[0][0])
        subjects.append(by_trial_subject[key])
    misid = int(sum(int(y != p) for y, p in zip(ys, ps)))
    subjects_mis = len({s for s, y, p in zip(subjects, ys, ps) if y != p})
    return (
        float(accuracy_score(ys, ps)),
        float(f1_score(ys, ps, average="macro")),
        len(ys),
        misid,
        subjects_mis,
    )


def evaluate_model(name, clf, x_train_feat, y_train, x_test_by_cond, y_test_by_cond, meta_test_by_cond):
    t0 = perf_counter()
    clf.fit(x_train_feat, y_train)
    rows = []
    for cond in sorted(x_test_by_cond):
        pred = clf.predict(x_test_by_cond[cond])
        y = y_test_by_cond[cond]
        trial_acc, trial_f1, n_trials, misid, subjects_mis = trial_metrics(y, pred, meta_test_by_cond[cond])
        rows.append({
            "model": name,
            "condition": cond,
            "window_accuracy": float(accuracy_score(y, pred)),
            "window_macro_f1": float(f1_score(y, pred, average="macro")),
            "trial_accuracy": trial_acc,
            "trial_macro_f1": trial_f1,
            "n_windows": int(len(y)),
            "n_trials": int(n_trials),
            "misid_trials": int(misid),
            "subjects_with_misid": int(subjects_mis),
            "elapsed_seconds": float(perf_counter() - t0),
        })
    return rows


def overall_row(rows, model):
    model_rows = [r for r in rows if r["model"] == model and r["condition"] != "OVERALL_MEAN"]
    out = {"model": model, "condition": "OVERALL_MEAN"}
    for key in ("window_accuracy", "window_macro_f1", "trial_accuracy", "trial_macro_f1"):
        out[key] = float(np.mean([r[key] for r in model_rows]))
    out["n_windows"] = int(sum(r["n_windows"] for r in model_rows))
    out["n_trials"] = int(sum(r["n_trials"] for r in model_rows))
    out["misid_trials"] = int(sum(r["misid_trials"] for r in model_rows))
    out["subjects_with_misid"] = ""
    out["elapsed_seconds"] = float(max(r["elapsed_seconds"] for r in model_rows))
    return out


def classifiers(seed):
    return {
        "pkd_ridge": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", RidgeClassifierCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
        ]),
        "pkd_linsvm": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LinearSVC(C=0.05, dual="auto", max_iter=8000, random_state=seed)),
        ]),
        "pkd_logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(C=0.5, max_iter=3000, random_state=seed, n_jobs=-1)),
        ]),
        "pkd_cosine_ridge": Pipeline([
            ("norm", Normalizer()),
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", RidgeClassifierCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
        ]),
        "pkd_proto_cosine": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("norm", Normalizer()),
            ("clf", CosinePrototypeClassifier()),
        ]),
        "pkd_proto_diag": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", DiagMahalanobisPrototypeClassifier(shrink=0.75)),
        ]),
    }


def run_dataset(data, args):
    out_dir = args.out_dir / data["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train = stack(data["x_train"])
    y_train = data["y_train"]
    x_train, y_train = augment_windows(x_train, y_train, args.augment_copies, args.seed + 123)
    if args.view == "canonical":
        x_train_base = canonicalize_pelvis_windows(x_train)
    elif args.view == "sway_residual":
        x_train_base = remove_slow_lateral_sway(x_train)
    else:
        x_train_base = x_train
    x_train_pkd = add_pelvis_invariants(x_train_base) if args.derived_channels else x_train_base
    x_test_raw = {cond: stack(wins) for cond, wins in data["x_test_by_cond"].items()}
    if args.view == "canonical":
        x_test_base = {cond: canonicalize_pelvis_windows(x) for cond, x in x_test_raw.items()}
    elif args.view == "sway_residual":
        x_test_base = {cond: remove_slow_lateral_sway(x) for cond, x in x_test_raw.items()}
    else:
        x_test_base = x_test_raw
    x_test_pkd_raw = {cond: add_pelvis_invariants(x) if args.derived_channels else x for cond, x in x_test_base.items()}
    print(f"[Data] {data['dataset']} subjects={len(data['subjects'])} train_windows={len(y_train)}")

    extractor = PelvisKernelDuelFeatures(
        groups=args.groups,
        kernels_per_group=args.kernels_per_group,
        kernel_size=args.kernel_size,
        dilations=tuple(int(x) for x in args.dilations.split(",")),
        seed=args.seed,
        batch_size=args.batch_size,
    ).fit(x_train_pkd)
    print("[Features] train", flush=True)
    x_train_feat = extractor.transform(x_train_pkd)
    if args.add_phcf:
        print("[Features] train PHCF fusion", flush=True)
        x_train_feat = np.hstack([x_train_feat, phcf_featurize(list(x_train_base))])
    if args.add_phase:
        print("[Features] train phase-cycle fusion", flush=True)
        x_train_feat = np.hstack([x_train_feat, phase_cycle_featurize(list(x_train_base))])
    print(f"[Features] train shape={x_train_feat.shape}", flush=True)
    x_test_feat = {}
    for cond, x in sorted(x_test_raw.items()):
        print(f"[Features] {cond} n={len(x)}", flush=True)
        feat = extractor.transform(x_test_pkd_raw[cond])
        if args.add_phcf:
            feat = np.hstack([feat, phcf_featurize(list(x_test_base[cond]))])
        if args.add_phase:
            feat = np.hstack([feat, phase_cycle_featurize(list(x_test_base[cond]))])
        x_test_feat[cond] = feat

    rows = []
    clfs = classifiers(args.seed)
    if args.models:
        keep = set(args.models.split(","))
        clfs = {k: v for k, v in clfs.items() if k in keep}
    for name, clf in clfs.items():
        print(f"[Fit/Eval] {name}", flush=True)
        rows.extend(evaluate_model(
            name,
            clf,
            x_train_feat,
            y_train,
            x_test_feat,
            data["y_test_by_cond"],
            data["meta_test_by_cond"],
        ))
        rows.append(overall_row(rows, name))

    fields = [
        "model",
        "condition",
        "window_accuracy",
        "window_macro_f1",
        "trial_accuracy",
        "trial_macro_f1",
        "n_windows",
        "n_trials",
        "misid_trials",
        "subjects_with_misid",
        "elapsed_seconds",
    ]
    suffix = "_phcf" if args.add_phcf else ""
    csv_path = out_dir / f"pkd_g{args.groups}_k{args.kernels_per_group}_seed{args.seed}{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ranking = sorted(
        [r for r in rows if r["condition"] == "OVERALL_MEAN"],
        key=lambda r: (r["trial_accuracy"], r["window_accuracy"]),
        reverse=True,
    )
    payload = {
        "dataset": data["dataset"],
        "model_family": "PelvisKernelDuel",
        "subjects": data["subjects"],
        "train_windows": int(len(y_train)),
        "feature_dim": int(x_train_feat.shape[1]),
        "args": vars(args) | {"out_dir": str(args.out_dir), "h5": str(args.h5), "kyoto_root": str(args.kyoto_root)},
        "ranking_overall_mean": ranking,
        "results": rows,
    }
    json_path = csv_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"dataset": data["dataset"], "ranking_overall_mean": ranking, "csv": str(csv_path)}, indent=2))
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["original", "kyoto", "both"], default="both")
    parser.add_argument("--h5", type=Path, default=H5_PATH)
    parser.add_argument("--kyoto_root", type=Path, default=KYOTO_ROOT)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--groups", type=int, default=96)
    parser.add_argument("--kernels_per_group", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--dilations", default="1,2,4,8,16")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--add_phcf", action="store_true")
    parser.add_argument("--models", default="")
    parser.add_argument("--augment_copies", type=int, default=0)
    parser.add_argument("--derived_channels", action="store_true")
    parser.add_argument("--view", choices=["raw", "canonical", "sway_residual"], default="raw")
    parser.add_argument("--add_phase", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("original", "both"):
        run_dataset(load_original_h5(args.h5), args)
    if args.dataset in ("kyoto", "both"):
        run_dataset(load_kyoto(args.kyoto_root), args)


if __name__ == "__main__":
    main()
