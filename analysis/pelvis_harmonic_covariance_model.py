"""
Pelvis-only harmonic/covariance gait fingerprint experiments.

Goal: test a lightweight, non-generic alternative to MB-3 on both the original
treadmill H5 and Kyoto overground CSV datasets, using the same closed-set
baseline-train / altered-condition-test protocol.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC, SVC

H5_PATH = Path("gait_id_pelvis_62.h5")
KYOTO_ROOT = Path("data/kyoto_overground_dataset")
OUT_DIR = Path("baselines_out/pelvis_harmonic_covariance")
SEED = 42

FS_HZ = 100
WIN_S = 2.0
WIN = int(WIN_S * FS_HZ)
HOP = WIN // 2
ACC_STD_MIN = 0.4

PELVIS_COLS = [
    "AccelX_PELVIS (m/s^2)",
    "AccelY_PELVIS (m/s^2)",
    "AccelZ_PELVIS (m/s^2)",
    "GyroX_PELVIS (deg/s)",
    "GyroY_PELVIS (deg/s)",
    "GyroZ_PELVIS (deg/s)",
]


def canonical_condition(name):
    return re.sub(r"\d+$", "", name)


def zscore(x_cxt):
    mu = x_cxt.mean(axis=1, keepdims=True)
    sd = x_cxt.std(axis=1, keepdims=True) + 1e-8
    return (x_cxt - mu) / sd


def gate_ok(seg_txc):
    svm = np.sqrt((seg_txc[:, :3] ** 2).sum(axis=1))
    return np.std(svm) >= ACC_STD_MIN


def window_matrix(mat_txc, meta_base):
    wins = []
    metas = []
    for start in range(0, mat_txc.shape[0] - WIN + 1, HOP):
        seg = mat_txc[start : start + WIN]
        if not gate_ok(seg):
            continue
        wins.append(zscore(seg.T.astype(np.float32, copy=False)))
        metas.append({**meta_base, "s0": int(start), "s1": int(start + WIN)})
    return wins, metas


def load_original_h5(path):
    def dec(arr):
        return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr[()]]

    x_train, train_meta = [], []
    x_test_by_cond = defaultdict(list)
    meta_test_by_cond = defaultdict(list)
    with h5py.File(path, "r") as h5:
        idx = h5["index"]
        subjects = dec(idx["subject"])
        conditions = dec(idx["condition"])
        trials = dec(idx["trial_id"])
        groups = dec(idx["h5_group"])
        has_base = sorted({s for s, c in zip(subjects, conditions) if c.lower() == "baseline"})
        label2id = {s: i for i, s in enumerate(has_base)}
        for subject, condition, trial, group in zip(subjects, conditions, trials, groups):
            if subject not in label2id:
                continue
            mat = h5[group]["imu"][()]
            cols = json.loads(h5[group].attrs["columns_json"])
            name2idx = {name: i for i, name in enumerate(cols)}
            out = np.zeros((mat.shape[0], len(PELVIS_COLS)), dtype=np.float32)
            for j, col in enumerate(PELVIS_COLS):
                out[:, j] = mat[:, name2idx[col]].astype(np.float32, copy=False)
            base = {
                "subject": subject,
                "condition": condition,
                "condition_raw": condition,
                "trial": trial,
                "group": group,
            }
            wins, metas = window_matrix(out, base)
            if condition.lower() == "baseline":
                x_train.extend(wins)
                train_meta.extend(metas)
            else:
                x_test_by_cond[condition].extend(wins)
                meta_test_by_cond[condition].extend(metas)
    y_train = np.asarray([label2id[m["subject"]] for m in train_meta], dtype=np.int32)
    y_test_by_cond = {
        cond: np.asarray([label2id[m["subject"]] for m in metas], dtype=np.int32)
        for cond, metas in meta_test_by_cond.items()
    }
    return {
        "dataset": "original_treadmill",
        "subjects": has_base,
        "label2id": label2id,
        "x_train": x_train,
        "y_train": y_train,
        "train_meta": train_meta,
        "x_test_by_cond": dict(x_test_by_cond),
        "y_test_by_cond": y_test_by_cond,
        "meta_test_by_cond": dict(meta_test_by_cond),
    }


def iter_kyoto_trials(root):
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cond_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            imu_path = cond_dir / "Trial" / "imu.csv"
            if imu_path.exists():
                yield subject_dir.name, cond_dir.name, canonical_condition(cond_dir.name), imu_path


def load_kyoto(root):
    trials = list(iter_kyoto_trials(root))
    subjects = sorted({s for s, _, cond, _ in trials if cond == "baseline"})
    label2id = {s: i for i, s in enumerate(subjects)}
    x_train, train_meta = [], []
    x_test_by_cond = defaultdict(list)
    meta_test_by_cond = defaultdict(list)
    skipped = []
    for subject, raw_cond, cond, path in trials:
        if subject not in label2id:
            skipped.append({"subject": subject, "condition_raw": raw_cond, "path": str(path), "reason": "no_baseline"})
            continue
        mat = pd.read_csv(path, usecols=PELVIS_COLS).to_numpy(dtype=np.float32, copy=True)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        base = {
            "subject": subject,
            "condition": cond,
            "condition_raw": raw_cond,
            "trial": raw_cond,
            "group": str(path),
        }
        wins, metas = window_matrix(mat, base)
        if cond == "baseline":
            x_train.extend(wins)
            train_meta.extend(metas)
        else:
            x_test_by_cond[cond].extend(wins)
            meta_test_by_cond[cond].extend(metas)
    y_train = np.asarray([label2id[m["subject"]] for m in train_meta], dtype=np.int32)
    y_test_by_cond = {
        cond: np.asarray([label2id[m["subject"]] for m in metas], dtype=np.int32)
        for cond, metas in meta_test_by_cond.items()
    }
    return {
        "dataset": "kyoto_overground",
        "subjects": subjects,
        "label2id": label2id,
        "x_train": x_train,
        "y_train": y_train,
        "train_meta": train_meta,
        "x_test_by_cond": dict(x_test_by_cond),
        "y_test_by_cond": y_test_by_cond,
        "meta_test_by_cond": dict(meta_test_by_cond),
        "skipped": skipped,
    }


def vectorize_upper(mat):
    iu = np.triu_indices(mat.shape[0])
    return mat[iu]


def safe_corrcoef(x):
    corr = np.corrcoef(x)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def log_cov_features(x):
    cov = np.cov(x) + np.eye(x.shape[0], dtype=np.float64) * 1e-4
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-8, None)
    log_cov = (vecs * np.log(vals)) @ vecs.T
    return vectorize_upper(log_cov)


def spectral_features(x):
    feats = []
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / FS_HZ)
    spec = np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)
    amp = np.abs(spec)
    power = amp ** 2
    harmonic_mask = (freqs >= 0.5) & (freqs <= 8.0)
    bands = [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 15.0), (15.0, 25.0)]
    for ch in range(x.shape[0]):
        p = power[ch] + 1e-12
        hp = p[harmonic_mask]
        hf = freqs[harmonic_mask]
        feats.extend(np.log1p(amp[ch, harmonic_mask]).tolist())
        feats.append(float((freqs * p).sum() / p.sum()))
        feats.append(float(-np.sum((p / p.sum()) * np.log(p / p.sum()))))
        if hp.size:
            top = np.argsort(hp)[-3:][::-1]
            for idx in top:
                feats.extend([float(hf[idx]), float(np.log1p(hp[idx]))])
        else:
            feats.extend([0.0] * 6)
        total = p.sum()
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            feats.append(float(np.log1p(p[mask].sum())))
            feats.append(float(p[mask].sum() / total))

    # Relative phase of each channel to pelvis acceleration magnitude's dominant gait bin.
    svm = np.sqrt((x[:3] ** 2).sum(axis=0))
    svm_spec = np.fft.rfft(svm * np.hanning(len(svm)))
    gait_mask = (freqs >= 0.5) & (freqs <= 3.0)
    if np.any(gait_mask):
        gait_bins = np.where(gait_mask)[0]
        dom_bin = gait_bins[np.argmax(np.abs(svm_spec[gait_bins]))]
        ref_phase = np.angle(svm_spec[dom_bin])
        for ch in range(x.shape[0]):
            delta = np.angle(spec[ch, dom_bin]) - ref_phase
            feats.extend([float(np.sin(delta)), float(np.cos(delta)), float(np.log1p(amp[ch, dom_bin]))])
    else:
        feats.extend([0.0] * (x.shape[0] * 3))
    return np.asarray(feats, dtype=np.float32)


def morphology_features(x):
    feats = []
    t = np.linspace(-1.0, 1.0, x.shape[1])
    for ch in range(x.shape[0]):
        sig = x[ch]
        feats.extend([
            float(sig.mean()),
            float(sig.std()),
            float(np.sqrt(np.mean(sig ** 2))),
            float(np.median(sig)),
            float(np.percentile(sig, 10)),
            float(np.percentile(sig, 25)),
            float(np.percentile(sig, 75)),
            float(np.percentile(sig, 90)),
            float(sig.max() - sig.min()),
            float(np.mean(np.abs(np.diff(sig)))),
            float(np.polyfit(t, sig, deg=1)[0]),
        ])
        centered = sig - sig.mean()
        sd = sig.std() + 1e-8
        feats.append(float(np.mean((centered / sd) ** 3)))
        feats.append(float(np.mean((centered / sd) ** 4)))
        for lag in (10, 20, 33, 50, 67):
            if lag < len(sig):
                a = centered[:-lag]
                b = centered[lag:]
                feats.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            else:
                feats.append(0.0)
    svm = np.sqrt((x[:3] ** 2).sum(axis=0))
    for lag in (10, 20, 33, 50, 67):
        centered = svm - svm.mean()
        a = centered[:-lag]
        b = centered[lag:]
        feats.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return np.asarray(feats, dtype=np.float32)


def phcf_features_one(win):
    raw = np.asarray(win, dtype=np.float32)
    d1 = np.diff(raw, axis=1, prepend=raw[:, :1])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    aug = np.vstack([raw, d1, d2]).astype(np.float64)
    feats = [
        morphology_features(raw),
        spectral_features(raw),
        vectorize_upper(safe_corrcoef(raw)).astype(np.float32),
        vectorize_upper(safe_corrcoef(np.vstack([raw, d1]))).astype(np.float32),
        log_cov_features(aug).astype(np.float32),
    ]
    out = np.concatenate(feats)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def featurize(windows):
    return np.vstack([phcf_features_one(w) for w in windows]).astype(np.float32)


class CentroidCosineClassifier:
    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        self.classes_ = np.asarray(sorted(set(y.tolist())))
        centroids = []
        for cls in self.classes_:
            c = x[y == cls].mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-12)
            centroids.append(c)
        self.centroids_ = np.vstack(centroids)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        scores = x @ self.centroids_.T
        return self.classes_[np.argmax(scores, axis=1)]


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


def evaluate_model(name, clf, x_train, y_train, x_test_by_cond, y_test_by_cond, meta_test_by_cond):
    t0 = perf_counter()
    clf.fit(x_train, y_train)
    rows = []
    for cond in sorted(x_test_by_cond):
        pred = clf.predict(x_test_by_cond[cond])
        y_test = y_test_by_cond[cond]
        trial_acc, trial_f1, n_trials, misid, subjects_mis = trial_metrics(y_test, pred, meta_test_by_cond[cond])
        rows.append({
            "model": name,
            "condition": cond,
            "window_accuracy": float(accuracy_score(y_test, pred)),
            "window_macro_f1": float(f1_score(y_test, pred, average="macro")),
            "trial_accuracy": trial_acc,
            "trial_macro_f1": trial_f1,
            "n_windows": int(len(y_test)),
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


def build_classifiers(seed):
    return {
        "phcf_centroid_cosine": Pipeline([
            ("scaler", StandardScaler()),
            ("norm", Normalizer()),
            ("clf", CentroidCosineClassifier()),
        ]),
        "phcf_ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifierCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
        ]),
        "phcf_logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=3000, random_state=seed, n_jobs=-1)),
        ]),
        "phcf_linsvm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(C=0.08, dual="auto", random_state=seed, max_iter=8000)),
        ]),
        "phcf_rbf_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=6.0, gamma="scale", kernel="rbf")),
        ]),
        "phcf_extratrees": ExtraTreesClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        ),
        "phcf_histgb": HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=seed,
        ),
    }


def run_dataset(data, args):
    out_dir = args.out_dir / data["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Data] {data['dataset']} subjects={len(data['subjects'])} train_windows={len(data['y_train'])}")
    print("[Feature] train", flush=True)
    x_train = featurize(data["x_train"])
    print(f"[Feature] train shape={x_train.shape}", flush=True)
    x_test_by_cond = {}
    for cond, wins in sorted(data["x_test_by_cond"].items()):
        print(f"[Feature] {cond} n={len(wins)}", flush=True)
        x_test_by_cond[cond] = featurize(wins)

    rows = []
    classifiers = build_classifiers(args.seed)
    if args.models:
        keep = set(args.models.split(","))
        classifiers = {k: v for k, v in classifiers.items() if k in keep}

    for name, clf in classifiers.items():
        print(f"[Fit/Eval] {data['dataset']} {name}", flush=True)
        model_rows = evaluate_model(
            name,
            clf,
            x_train,
            data["y_train"],
            x_test_by_cond,
            data["y_test_by_cond"],
            data["meta_test_by_cond"],
        )
        rows.extend(model_rows)
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
    csv_path = out_dir / "phcf_results.csv"
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
        "subjects": data["subjects"],
        "train_windows": int(len(data["y_train"])),
        "test_windows_by_condition": {k: int(len(v)) for k, v in data["y_test_by_cond"].items()},
        "feature_dim": int(x_train.shape[1]),
        "feature": "pelvis harmonic-covariance fingerprint: morphology, gait harmonics, phase, correlations, log covariance of raw/diff/diff2",
        "ranking_overall_mean": ranking,
        "results": rows,
    }
    json_path = out_dir / "phcf_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"dataset": data["dataset"], "ranking_overall_mean": ranking[:5], "csv": str(csv_path)}, indent=2))
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["original", "kyoto", "both"], default="original")
    parser.add_argument("--h5", type=Path, default=H5_PATH)
    parser.add_argument("--kyoto_root", type=Path, default=KYOTO_ROOT)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--models", default="", help="Comma-separated subset of classifier names")
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("original", "both"):
        run_dataset(load_original_h5(args.h5), args)
    if args.dataset in ("kyoto", "both"):
        run_dataset(load_kyoto(args.kyoto_root), args)


if __name__ == "__main__":
    main()
