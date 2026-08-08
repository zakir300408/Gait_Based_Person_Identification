"""MB-Lite closed-set and open-set verification metrics.

Outputs score-level EER/AUC summaries for MB-Lite on private treadmill and
Tantian data. This complements closed-set identification accuracy tables.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pelvis_harmonic_covariance_model import H5_PATH, featurize as phcf_featurize, load_original_h5
from pelvis_kernel_duel_model import PelvisKernelDuelFeatures
from pelvis_mblite_fusion import fit_mblite, transform_mblite
try:
    from tantian_mblite_benchmark import ROOT as TANTIAN_ROOT, load_tantian
except ImportError:
    TANTIAN_ROOT = Path("public_datasets/tantian")
    load_tantian = None

OUT_DIR = Path("baselines_out/mblite_verification")
TRUNK_NAMES = {"trunk_sway", "trunk_sway_large", "trunk_sway_small"}


def stack(windows):
    return np.ascontiguousarray(np.stack(windows, axis=0).astype(np.float64))


def make_classifier(cosine=False):
    steps = []
    if cosine:
        steps.append(("norm", Normalizer()))
    steps.extend(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", RidgeClassifierCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
        ]
    )
    return Pipeline(steps)


def score_matrix(clf, x):
    scores = clf.decision_function(x)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return np.asarray(scores, dtype=np.float32)


def compute_auc_eer(genuine, impostor):
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    y = np.concatenate([np.ones(len(genuine), dtype=int), np.zeros(len(impostor), dtype=int)])
    s = np.concatenate([genuine, impostor])
    fpr, tpr, thr = roc_curve(y, s)
    roc_auc = float(auc(fpr, tpr))
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return roc_auc, eer, float(thr[idx]), fpr, tpr


def group_name(condition):
    if condition in TRUNK_NAMES:
        return "trunk_sway_only"
    return "non_sway"


def fit_features(x_train, y_train, x_blocks, args):
    kernels = {"raw": args.raw_kernels, "diff": args.diff_kernels, "diff2": args.diff2_kernels}
    models, train_mb = fit_mblite(x_train, y_train, kernels, args.seed)
    train_blocks = [train_mb.astype(np.float32)]
    test_blocks = {name: [transform_mblite(models, x).astype(np.float32)] for name, x in x_blocks.items()}
    dims = {"mblite_rocket": int(train_mb.shape[1])}

    if args.add_pkd:
        pkd = PelvisKernelDuelFeatures(
            groups=args.pkd_groups,
            kernels_per_group=args.pkd_kernels_per_group,
            kernel_size=9,
            dilations=(1, 2, 4, 8),
            seed=args.seed,
            batch_size=args.batch_size,
        ).fit(x_train.astype(np.float32))
        train_pkd = pkd.transform(x_train.astype(np.float32))
        train_blocks.append(train_pkd)
        dims["pkd"] = int(train_pkd.shape[1])
        for name, x in x_blocks.items():
            test_blocks[name].append(pkd.transform(x.astype(np.float32)))

    if args.add_phcf:
        # x_train is N,C,T, while phcf_featurize accepts iterable windows.
        train_phcf = phcf_featurize([w.astype(np.float32) for w in x_train])
        train_blocks.append(train_phcf)
        dims["phcf"] = int(train_phcf.shape[1])
        for name, x in x_blocks.items():
            test_blocks[name].append(phcf_featurize([w.astype(np.float32) for w in x]))

    xtr = np.hstack(train_blocks).astype(np.float32)
    xout = {name: np.hstack(blocks).astype(np.float32) for name, blocks in test_blocks.items()}
    return xtr, xout, dims


def closed_identification_rows(dataset, clf, x_by_cond, y_by_cond, meta_by_cond, elapsed):
    rows = []
    for cond, x in sorted(x_by_cond.items()):
        y = y_by_cond[cond]
        pred = clf.predict(x)
        rows.append(
            {
                "dataset": dataset,
                "protocol": "closed_set_identification",
                "group": group_name(cond),
                "condition": cond,
                "level": "window",
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro")),
                "n": int(len(y)),
                "elapsed_seconds": float(elapsed),
            }
        )
        # Trial majority vote.
        by_trial = defaultdict(list)
        truth = {}
        for yt, yp, meta in zip(y, pred, meta_by_cond[cond]):
            key = f"{meta['subject']}|{meta['condition']}|{meta['trial']}"
            by_trial[key].append(int(yp))
            truth[key] = int(yt)
        ty, py = [], []
        for key, preds in by_trial.items():
            vals, counts = np.unique(preds, return_counts=True)
            ty.append(truth[key])
            py.append(int(vals[np.argmax(counts)]))
        rows.append(
            {
                "dataset": dataset,
                "protocol": "closed_set_identification",
                "group": group_name(cond),
                "condition": cond,
                "level": "trial",
                "accuracy": float(accuracy_score(ty, py)),
                "macro_f1": float(f1_score(ty, py, average="macro")),
                "n": int(len(ty)),
                "elapsed_seconds": float(elapsed),
            }
        )
    return rows


def verification_scores_from_scores(scores, y, metas, labels, impostor_mode="all_nontrue"):
    label_to_col = {int(label): i for i, label in enumerate(labels)}
    genuine_w, impostor_w = defaultdict(list), defaultdict(list)
    for row, yt, meta in zip(scores, y, metas):
        cond = meta["condition"]
        true_col = label_to_col[int(yt)]
        genuine_w["overall"].append(float(row[true_col]))
        genuine_w[group_name(cond)].append(float(row[true_col]))
        if impostor_mode == "max_nontrue":
            imp_vals = [float(row[j]) for j in range(row.shape[0]) if j != true_col]
            vals = [max(imp_vals)]
        else:
            vals = [float(row[j]) for j in range(row.shape[0]) if j != true_col]
        impostor_w["overall"].extend(vals)
        impostor_w[group_name(cond)].extend(vals)

    # Trial-level average score vectors.
    by_trial = defaultdict(list)
    y_trial = {}
    cond_trial = {}
    for row, yt, meta in zip(scores, y, metas):
        key = f"{meta['subject']}|{meta['condition']}|{meta['trial']}"
        by_trial[key].append(row)
        y_trial[key] = int(yt)
        cond_trial[key] = meta["condition"]
    genuine_t, impostor_t = defaultdict(list), defaultdict(list)
    for key, rows in by_trial.items():
        mean_scores = np.vstack(rows).mean(axis=0)
        true_col = label_to_col[y_trial[key]]
        cond = cond_trial[key]
        genuine_t["overall"].append(float(mean_scores[true_col]))
        genuine_t[group_name(cond)].append(float(mean_scores[true_col]))
        if impostor_mode == "max_nontrue":
            vals = [max(float(mean_scores[j]) for j in range(mean_scores.shape[0]) if j != true_col)]
        else:
            vals = [float(mean_scores[j]) for j in range(mean_scores.shape[0]) if j != true_col]
        impostor_t["overall"].extend(vals)
        impostor_t[group_name(cond)].extend(vals)
    return (genuine_w, impostor_w), (genuine_t, impostor_t)


def trial_score_rows(dataset, model, fold, kind, condition, scores, metas, labels, y=None):
    """Retain trial-level claim scores for DET and fixed-threshold analysis."""
    by_trial = defaultdict(list)
    trial_meta = {}
    trial_truth = {}
    for index, (row, meta) in enumerate(zip(scores, metas)):
        key = f"{meta['subject']}|{meta['condition']}|{meta['trial']}"
        by_trial[key].append(row)
        trial_meta[key] = meta
        if y is not None:
            trial_truth[key] = int(y[index])

    label_to_col = {int(label): index for index, label in enumerate(labels)}
    rows = []
    for key, score_rows in by_trial.items():
        mean_scores = np.vstack(score_rows).mean(axis=0)
        meta = trial_meta[key]
        base = {
            "dataset": dataset,
            "model": model,
            "fold": int(fold),
            "group": group_name(condition),
            "subject": meta["subject"],
            "condition": condition,
            "trial": meta["trial"],
        }
        if kind == "genuine":
            claim = trial_truth[key]
            rows.append({**base, "claim": claim, "genuine": 1, "score": float(mean_scores[label_to_col[claim]])})
        else:
            for column, claim in enumerate(labels):
                rows.append({**base, "claim": int(claim), "genuine": 0, "score": float(mean_scores[column])})
    return rows


def verification_rows(dataset, protocol, score_sets, fold=""):
    rows = []
    for level, (genuine, impostor) in score_sets.items():
        for group in ("overall", "trunk_sway_only", "non_sway"):
            if len(genuine[group]) == 0 or len(impostor[group]) == 0:
                continue
            roc_auc, eer, thr, _fpr, _tpr = compute_auc_eer(genuine[group], impostor[group])
            rows.append(
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "fold": fold,
                    "group": group,
                    "level": level,
                    "n_genuine": int(len(genuine[group])),
                    "n_impostor": int(len(impostor[group])),
                    "auc": roc_auc,
                    "eer": eer,
                    "eer_percent": eer * 100.0,
                    "threshold_at_eer": thr,
                }
            )
    return rows


def fit_closed_dataset(dataset, data, args):
    t0 = perf_counter()
    x_train = stack(data["x_train"])
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    x_test_raw = {cond: stack(wins) for cond, wins in data["x_test_by_cond"].items()}
    x_train_feat, x_test_feat, dims = fit_features(x_train, y_train, x_test_raw, args)
    clf = make_classifier(cosine=args.cosine)
    clf.fit(x_train_feat, y_train)
    elapsed = perf_counter() - t0
    id_rows = closed_identification_rows(dataset, clf, x_test_feat, data["y_test_by_cond"], data["meta_test_by_cond"], elapsed)

    all_scores = []
    all_y = []
    all_meta = []
    for cond, feat in sorted(x_test_feat.items()):
        all_scores.append(score_matrix(clf, feat))
        all_y.append(np.asarray(data["y_test_by_cond"][cond], dtype=np.int64))
        all_meta.extend(data["meta_test_by_cond"][cond])
    scores = np.vstack(all_scores)
    y = np.concatenate(all_y)
    labels = np.asarray(clf.named_steps["ridge"].classes_, dtype=np.int64)
    win_scores, trial_scores = verification_scores_from_scores(scores, y, all_meta, labels, args.impostor_mode)
    ver_rows = verification_rows(
        dataset,
        "closed_set_verification",
        {"window": win_scores, "trial": trial_scores},
    )
    return id_rows, ver_rows, dims


def subset_by_subject(data, train_subjects, impostor_subjects):
    train_subjects = set(train_subjects)
    impostor_subjects = set(impostor_subjects)
    label2id = {s: i for i, s in enumerate(sorted(train_subjects))}
    x_train, y_train = [], []
    genuine_by_cond = defaultdict(list)
    yg_by_cond = defaultdict(list)
    mg_by_cond = defaultdict(list)
    impostor_by_cond = defaultdict(list)
    mi_by_cond = defaultdict(list)

    for w, meta in zip(data["x_train"], data["train_meta"]):
        s = meta["subject"]
        if s in train_subjects:
            x_train.append(w)
            y_train.append(label2id[s])
        elif s in impostor_subjects:
            impostor_by_cond["baseline"].append(w)
            mi_by_cond["baseline"].append(meta)
    for cond, wins in data["x_test_by_cond"].items():
        metas = data["meta_test_by_cond"][cond]
        for w, meta in zip(wins, metas):
            s = meta["subject"]
            if s in train_subjects:
                genuine_by_cond[cond].append(w)
                yg_by_cond[cond].append(label2id[s])
                mg_by_cond[cond].append(meta)
            elif s in impostor_subjects:
                impostor_by_cond[cond].append(w)
                mi_by_cond[cond].append(meta)
    return {
        "label2id": label2id,
        "x_train": x_train,
        "y_train": np.asarray(y_train, dtype=np.int64),
        "genuine_by_cond": dict(genuine_by_cond),
        "yg_by_cond": {k: np.asarray(v, dtype=np.int64) for k, v in yg_by_cond.items()},
        "mg_by_cond": dict(mg_by_cond),
        "impostor_by_cond": dict(impostor_by_cond),
        "mi_by_cond": dict(mi_by_cond),
    }


def open_set_rows(dataset, data, args):
    subjects = sorted(data["subjects"])
    n_splits = min(args.open_folds, len(subjects))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    rows = []
    score_rows = []
    fold_summaries = []
    for fold_idx, (tr_idx, imp_idx) in enumerate(kf.split(subjects), 1):
        t0 = perf_counter()
        train_subjects = [subjects[i] for i in tr_idx]
        impostor_subjects = [subjects[i] for i in imp_idx]
        split = subset_by_subject(data, train_subjects, impostor_subjects)
        if len(split["x_train"]) == 0:
            continue
        x_train = stack(split["x_train"])
        y_train = split["y_train"]
        x_blocks = {}
        meta_blocks = {}
        y_blocks = {}
        for cond, wins in split["genuine_by_cond"].items():
            if wins:
                name = f"genuine::{cond}"
                x_blocks[name] = stack(wins)
                meta_blocks[name] = split["mg_by_cond"][cond]
                y_blocks[name] = split["yg_by_cond"][cond]
        for cond, wins in split["impostor_by_cond"].items():
            if wins:
                name = f"impostor::{cond}"
                x_blocks[name] = stack(wins)
                meta_blocks[name] = split["mi_by_cond"][cond]
        if not x_blocks:
            continue
        x_train_feat, x_feat, _dims = fit_features(x_train, y_train, x_blocks, args)
        clf = make_classifier(cosine=args.cosine)
        clf.fit(x_train_feat, y_train)
        labels = np.asarray(clf.named_steps["ridge"].classes_, dtype=np.int64)

        gw, iw = defaultdict(list), defaultdict(list)
        gt, it = defaultdict(list), defaultdict(list)

        # Genuine enrolled windows/trials.
        for key, feat in x_feat.items():
            kind, cond = key.split("::", 1)
            scores = score_matrix(clf, feat)
            metas = meta_blocks[key]
            if kind == "genuine":
                y = y_blocks[key]
                score_rows.extend(
                    trial_score_rows(dataset, "PRISM-Gait", fold_idx, kind, cond, scores, metas, labels, y)
                )
                (g_w, _i_w), (g_t, _i_t) = verification_scores_from_scores(scores, y, metas, labels, "max_nontrue")
                for group in g_w:
                    gw[group].extend(g_w[group])
                for group in g_t:
                    gt[group].extend(g_t[group])
            else:
                score_rows.extend(
                    trial_score_rows(dataset, "PRISM-Gait", fold_idx, kind, cond, scores, metas, labels)
                )
                # For unseen impostors, every enrolled claim is an impostor.
                group = group_name(cond)
                iw["overall"].extend(scores.ravel().astype(float).tolist())
                iw[group].extend(scores.ravel().astype(float).tolist())
                by_trial = defaultdict(list)
                for row, meta in zip(scores, metas):
                    tkey = f"{meta['subject']}|{meta['condition']}|{meta['trial']}"
                    by_trial[tkey].append(row)
                for _tkey, trial_rows in by_trial.items():
                    vals = np.vstack(trial_rows).mean(axis=0).astype(float).tolist()
                    it["overall"].extend(vals)
                    it[group].extend(vals)

        fold_rows = verification_rows(
            dataset,
            "open_set_subject_heldout_verification",
            {"window": (gw, iw), "trial": (gt, it)},
            fold=str(fold_idx),
        )
        for r in fold_rows:
            r["elapsed_seconds"] = float(perf_counter() - t0)
            r["n_enrolled_subjects"] = int(len(train_subjects))
            r["n_impostor_subjects"] = int(len(impostor_subjects))
        rows.extend(fold_rows)
        fold_summaries.append({"fold": fold_idx, "enrolled": len(train_subjects), "impostors": len(impostor_subjects)})
        print(f"[OpenSet] {dataset} fold {fold_idx}/{n_splits} done in {perf_counter() - t0:.1f}s", flush=True)
    return rows, fold_summaries, score_rows


def load_dataset(name, args):
    if name == "original_treadmill":
        return load_original_h5(args.h5)
    if name == "tantian_grf_waist":
        if load_tantian is None:
            raise ImportError("Tantian support requires tantian_mblite_benchmark.py")
        return load_tantian(args.tantian_root, args.win, args.hop, args.acc_std_min, args.max_windows_per_file)
    raise ValueError(f"Unknown dataset: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="original_treadmill")
    parser.add_argument("--h5", type=Path, default=H5_PATH)
    parser.add_argument("--tantian-root", type=Path, default=TANTIAN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--raw-kernels", type=int, default=3000)
    parser.add_argument("--diff-kernels", type=int, default=3000)
    parser.add_argument("--diff2-kernels", type=int, default=2000)
    parser.add_argument("--add-pkd", action="store_true")
    parser.add_argument("--add-phcf", action="store_true")
    parser.add_argument("--pkd-groups", type=int, default=48)
    parser.add_argument("--pkd-kernels-per-group", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--impostor-mode", choices=("all_nontrue", "max_nontrue"), default="all_nontrue")
    parser.add_argument("--skip-open-set", action="store_true")
    parser.add_argument("--open-folds", type=int, default=5)
    parser.add_argument("--win", type=int, default=200)
    parser.add_argument("--hop", type=int, default=100)
    parser.add_argument("--acc-std-min", type=float, default=0.4)
    parser.add_argument("--max-windows-per-file", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_id_rows = []
    all_ver_rows = []
    all_score_rows = []
    payload = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "datasets": {}}
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"[Dataset] {dataset}", flush=True)
        data = load_dataset(dataset, args)
        id_rows, closed_rows, dims = fit_closed_dataset(dataset, data, args)
        all_id_rows.extend(id_rows)
        all_ver_rows.extend(closed_rows)
        payload["datasets"][dataset] = {"feature_dims": dims, "subjects": data["subjects"]}
        if not args.skip_open_set:
            open_rows, fold_summaries, score_rows = open_set_rows(dataset, data, args)
            all_ver_rows.extend(open_rows)
            all_score_rows.extend(score_rows)
            payload["datasets"][dataset]["open_set_folds"] = fold_summaries

    id_path = args.out_dir / "mblite_closed_set_identification.csv"
    ver_path = args.out_dir / "mblite_verification_metrics.csv"
    with open(id_path, "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "protocol", "group", "condition", "level", "accuracy", "macro_f1", "n", "elapsed_seconds"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_id_rows)
    ver_fields = [
        "dataset",
        "protocol",
        "fold",
        "group",
        "level",
        "n_genuine",
        "n_impostor",
        "auc",
        "eer",
        "eer_percent",
        "threshold_at_eer",
        "elapsed_seconds",
        "n_enrolled_subjects",
        "n_impostor_subjects",
    ]
    with open(ver_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ver_fields)
        writer.writeheader()
        writer.writerows(all_ver_rows)
    with open(ver_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        payload["identification_rows"] = all_id_rows
        payload["verification_rows"] = all_ver_rows
        json.dump(payload, f, indent=2)
    score_path = args.out_dir / "open_set_trial_scores.csv"
    score_fields = ["dataset", "model", "fold", "group", "subject", "condition", "trial", "claim", "genuine", "score"]
    with open(score_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=score_fields)
        writer.writeheader()
        writer.writerows(all_score_rows)
    print(f"[Saved] {id_path}")
    print(f"[Saved] {ver_path}")
    print(f"[Saved] {score_path}")


if __name__ == "__main__":
    main()
