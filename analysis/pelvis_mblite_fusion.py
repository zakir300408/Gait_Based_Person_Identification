"""
MB-Lite Fusion: compressed multi-branch pelvis kernels + gait fingerprints.

This keeps the useful raw/diff/diff2 idea from MB-3, but uses far fewer
MiniROCKET kernels and fuses compact pelvis-specific PKD/PHCF descriptors.
The goal is a pelvis-only model that is lighter than 28k-kernel MB-3 while
beating it on both original treadmill and Kyoto overground protocols.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pelvis_harmonic_covariance_model import H5_PATH, KYOTO_ROOT, featurize as phcf_featurize, load_kyoto, load_original_h5
from pelvis_kernel_duel_model import PelvisKernelDuelFeatures

OUT_DIR = Path("baselines_out/pelvis_mblite_fusion")
SEED = 42


def stack(windows):
    return np.ascontiguousarray(np.stack(windows, axis=0).astype(np.float64))


def branches():
    return {
        "raw": lambda x: x,
        "diff": lambda x: np.diff(x, axis=2),
        "diff2": lambda x: np.diff(x, n=2, axis=2),
    }


def import_minirocket():
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    return MiniRocketMultivariate


def fit_mblite(x_train, y_train, kernels, seed):
    MiniRocket = import_minirocket()
    models = {}
    feats = []
    for i, (name, fn) in enumerate(branches().items()):
        n_k = int(kernels[name])
        mr = MiniRocket(num_kernels=n_k, random_state=seed + i, n_jobs=-1)
        xb = np.ascontiguousarray(fn(x_train), dtype=np.float64)
        mr.fit(xb, y_train)
        fb = mr.transform(xb)
        models[name] = mr
        feats.append(fb)
    return models, np.concatenate(feats, axis=1)


def transform_mblite(models, x):
    feats = []
    for name, fn in branches().items():
        xb = np.ascontiguousarray(fn(x), dtype=np.float64)
        feats.append(models[name].transform(xb))
    return np.concatenate(feats, axis=1)


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


def evaluate_model(name, clf, x_train_feat, y_train, x_test_feat_by_cond, data):
    t0 = perf_counter()
    clf.fit(x_train_feat, y_train)
    rows = []
    for cond in sorted(x_test_feat_by_cond):
        pred = clf.predict(x_test_feat_by_cond[cond])
        y = data["y_test_by_cond"][cond]
        trial_acc, trial_f1, n_trials, misid, subjects_mis = trial_metrics(y, pred, data["meta_test_by_cond"][cond])
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


def run_dataset(data, args):
    out_dir = args.out_dir / data["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Data] {data['dataset']} subjects={len(data['subjects'])} train_windows={len(data['y_train'])}")
    x_train = stack(data["x_train"])
    y_train = data["y_train"]
    x_test_raw = {cond: stack(wins) for cond, wins in data["x_test_by_cond"].items()}

    kernels = {"raw": args.raw_kernels, "diff": args.diff_kernels, "diff2": args.diff2_kernels}
    print(f"[Fit] MB-lite kernels={kernels}", flush=True)
    mb_models, x_train_mb = fit_mblite(x_train, y_train, kernels, args.seed)
    x_test_mb = {cond: transform_mblite(mb_models, x) for cond, x in x_test_raw.items()}

    feature_blocks = [x_train_mb]
    test_blocks = {cond: [feat] for cond, feat in x_test_mb.items()}
    feature_dims = {"mblite": int(x_train_mb.shape[1])}

    if args.add_pkd:
        print("[Fit] PKD compact block", flush=True)
        pkd = PelvisKernelDuelFeatures(
            groups=args.pkd_groups,
            kernels_per_group=args.pkd_kernels_per_group,
            kernel_size=args.pkd_kernel_size,
            dilations=tuple(int(x) for x in args.pkd_dilations.split(",")),
            seed=args.seed,
            batch_size=args.batch_size,
        ).fit(x_train.astype(np.float32))
        x_train_pkd = pkd.transform(x_train.astype(np.float32))
        feature_blocks.append(x_train_pkd)
        feature_dims["pkd"] = int(x_train_pkd.shape[1])
        for cond, x in x_test_raw.items():
            test_blocks[cond].append(pkd.transform(x.astype(np.float32)))

    if args.add_phcf:
        print("[Feature] PHCF block", flush=True)
        x_train_phcf = phcf_featurize(data["x_train"])
        feature_blocks.append(x_train_phcf)
        feature_dims["phcf"] = int(x_train_phcf.shape[1])
        for cond, wins in data["x_test_by_cond"].items():
            test_blocks[cond].append(phcf_featurize(wins))

    x_train_feat = np.hstack(feature_blocks).astype(np.float32)
    x_test_feat = {cond: np.hstack(blocks).astype(np.float32) for cond, blocks in test_blocks.items()}
    print(f"[Info] total_feature_dim={x_train_feat.shape[1]} dims={feature_dims}", flush=True)

    clfs = {
        "mblite_fusion_ridge": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", RidgeClassifierCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
        ]),
        "mblite_fusion_cosine_ridge": Pipeline([
            ("norm", Normalizer()),
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", RidgeClassifierCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
        ]),
    }
    rows = []
    for name, clf in clfs.items():
        print(f"[Fit/Eval] {name}", flush=True)
        rows.extend(evaluate_model(name, clf, x_train_feat, y_train, x_test_feat, data))
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
    suffix = f"r{args.raw_kernels}_d{args.diff_kernels}_d2{args.diff2_kernels}_pkd{int(args.add_pkd)}_phcf{int(args.add_phcf)}"
    csv_path = out_dir / f"mblite_fusion_{suffix}.csv"
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
        "model_family": "MB-Lite Fusion",
        "subjects": data["subjects"],
        "train_windows": int(len(y_train)),
        "feature_dims": feature_dims,
        "total_feature_dim": int(x_train_feat.shape[1]),
        "mb3_reference_kernels": 28000,
        "args": vars(args) | {"out_dir": str(args.out_dir), "h5": str(args.h5), "kyoto_root": str(args.kyoto_root)},
        "ranking_overall_mean": ranking,
        "results": rows,
    }
    with open(csv_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"dataset": data["dataset"], "ranking_overall_mean": ranking, "csv": str(csv_path), "feature_dims": feature_dims}, indent=2))
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["original", "kyoto", "both"], default="both")
    parser.add_argument("--h5", type=Path, default=H5_PATH)
    parser.add_argument("--kyoto_root", type=Path, default=KYOTO_ROOT)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--raw_kernels", type=int, default=3000)
    parser.add_argument("--diff_kernels", type=int, default=3000)
    parser.add_argument("--diff2_kernels", type=int, default=2000)
    parser.add_argument("--add_pkd", action="store_true")
    parser.add_argument("--pkd_groups", type=int, default=48)
    parser.add_argument("--pkd_kernels_per_group", type=int, default=6)
    parser.add_argument("--pkd_kernel_size", type=int, default=9)
    parser.add_argument("--pkd_dilations", default="1,2,4,8,16")
    parser.add_argument("--add_phcf", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Optional test-condition subset; baseline training data are unchanged.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("original", "both"):
        data = load_original_h5(args.h5)
        if args.conditions:
            keep = set(args.conditions)
            for key in ("x_test_by_cond", "y_test_by_cond", "meta_test_by_cond"):
                data[key] = {name: value for name, value in data[key].items() if name in keep}
        run_dataset(data, args)
    if args.dataset in ("kyoto", "both"):
        run_dataset(load_kyoto(args.kyoto_root), args)


if __name__ == "__main__":
    main()
