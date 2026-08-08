"""PRISM-Gait component and window-duration ablation on original treadmill data.

This study is meant to replace the older MB-3 branch/window ablation figure
with the new PRISM-Gait configuration:

  1. Component ablation at the selected 2 s window.
  2. Window-duration sweep for the selected full PRISM-Gait configuration.

Outputs:
  baselines_out/prism_gait_ablation/component_ablation.csv
  baselines_out/prism_gait_ablation/window_duration_sweep.csv
  baselines_out/prism_gait_ablation/subject_condition_accuracy.csv
  baselines_out/prism_gait_ablation/fig_prism_gait_ablation.png/.pdf
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from analysis.prism_gait.hcgf import H5_PATH, PELVIS_COLS, featurize as hcgf_featurize
from analysis.prism_gait.cdkc import CDKCFeatures

OUT_DIR = Path("baselines_out/prism_gait_ablation")
SEED = 42
FS_HZ = 100
ACC_STD_MIN = 0.4

EVAL_CONDITIONS = [
    "fpa_large",
    "fpa_small",
    "step_width_large",
    "step_width_small",
    "trunk_sway_large",
    "trunk_sway_small",
    "walk_speed_large",
    "walk_speed_small",
    "dual_task",
]

TRUNK_CONDITIONS = ("trunk_sway_large", "trunk_sway_small")

COMPONENT_VARIANTS = [
    {
        "variant": "R",
        "label": "Raw",
        "branches": ("raw",),
        "pkd": False,
        "phcf": False,
    },
    {
        "variant": "R+D",
        "label": "Raw+Diff",
        "branches": ("raw", "diff"),
        "pkd": False,
        "phcf": False,
    },
    {
        "variant": "Lite core",
        "label": "Raw+Diff+Diff2",
        "branches": ("raw", "diff", "diff2"),
        "pkd": False,
        "phcf": False,
    },
    {
        "variant": "Core+PKD",
        "label": "+PKD",
        "branches": ("raw", "diff", "diff2"),
        "pkd": True,
        "phcf": False,
    },
    {
        "variant": "Core+PHCF",
        "label": "+PHCF",
        "branches": ("raw", "diff", "diff2"),
        "pkd": False,
        "phcf": True,
    },
    {
        "variant": "PRISM-Gait",
        "label": "PRISM-Gait",
        "branches": ("raw", "diff", "diff2"),
        "pkd": True,
        "phcf": True,
    },
]

BRANCH_KERNELS = {"raw": 3000, "diff": 3000, "diff2": 2000}


def set_seed(seed):
    np.random.seed(seed)


def decode(arr):
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr[()]]


def zscore_window(x_cxt):
    return ((x_cxt - x_cxt.mean(axis=1, keepdims=True)) / (x_cxt.std(axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def gate_ok(seg_txc, acc_std_min):
    svm = np.sqrt((seg_txc[:, :3] ** 2).sum(axis=1))
    return float(np.std(svm)) >= float(acc_std_min)


def window_matrix(mat_txc, meta_base, win_s, acc_std_min):
    win = int(round(float(win_s) * FS_HZ))
    hop = win // 2
    wins = []
    metas = []
    for start in range(0, mat_txc.shape[0] - win + 1, hop):
        seg = mat_txc[start : start + win]
        if not gate_ok(seg, acc_std_min):
            continue
        wins.append(zscore_window(seg.T.astype(np.float32, copy=False)))
        metas.append({**meta_base, "s0": int(start), "s1": int(start + win), "win_s": float(win_s)})
    return wins, metas


def load_original_h5_windowed(path, win_s, acc_std_min):
    x_train, train_meta = [], []
    x_test_by_cond = defaultdict(list)
    meta_test_by_cond = defaultdict(list)
    with h5py.File(path, "r") as h5:
        idx = h5["index"]
        subjects = decode(idx["subject"])
        conditions = decode(idx["condition"])
        trials = decode(idx["trial_id"])
        groups = decode(idx["h5_group"])
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
            wins, metas = window_matrix(out, base, win_s, acc_std_min)
            if condition.lower() == "baseline":
                x_train.extend(wins)
                train_meta.extend(metas)
            else:
                x_test_by_cond[condition].extend(wins)
                meta_test_by_cond[condition].extend(metas)
    y_train = np.asarray([label2id[m["subject"]] for m in train_meta], dtype=np.int64)
    y_test_by_cond = {
        cond: np.asarray([label2id[m["subject"]] for m in metas], dtype=np.int64)
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
        "win_s": float(win_s),
    }


def stack(windows):
    return np.ascontiguousarray(np.stack(windows, axis=0).astype(np.float64))


def branch_transform(name, x):
    if name == "raw":
        return x
    if name == "diff":
        return np.diff(x, axis=2)
    if name == "diff2":
        return np.diff(x, n=2, axis=2)
    raise ValueError(name)


def import_minirocket():
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    return MiniRocketMultivariate


def fit_branch_features(x_train, y_train, x_test_raw, branches, seed):
    MiniRocket = import_minirocket()
    models = {}
    train_blocks = []
    test_blocks = {cond: [] for cond in x_test_raw}
    for i, branch in enumerate(branches):
        kernels = int(BRANCH_KERNELS[branch])
        rocket = MiniRocket(num_kernels=kernels, random_state=seed + i, n_jobs=-1)
        xb = np.ascontiguousarray(branch_transform(branch, x_train), dtype=np.float64)
        rocket.fit(xb, y_train)
        train_blocks.append(np.asarray(rocket.transform(xb), dtype=np.float32))
        models[branch] = rocket
        for cond, x in x_test_raw.items():
            xt = np.ascontiguousarray(branch_transform(branch, x), dtype=np.float64)
            test_blocks[cond].append(np.asarray(rocket.transform(xt), dtype=np.float32))
    x_train_feat = np.hstack(train_blocks).astype(np.float32)
    x_test_feat = {cond: np.hstack(blocks).astype(np.float32) for cond, blocks in test_blocks.items()}
    feature_dims = {branch: int(block.shape[1]) for branch, block in zip(branches, train_blocks)}
    return x_train_feat, x_test_feat, feature_dims


def add_pkd_features(x_train_feat, x_test_feat, x_train_raw, x_test_raw, seed, args):
    pkd = CDKCFeatures(
        groups=args.pkd_groups,
        kernels_per_group=args.pkd_kernels_per_group,
        kernel_size=9,
        dilations=tuple(args.pkd_dilations),
        seed=seed,
        batch_size=args.batch_size,
    ).fit(x_train_raw.astype(np.float32))
    train_pkd = pkd.transform(x_train_raw.astype(np.float32))
    x_train_feat = np.hstack([x_train_feat, train_pkd]).astype(np.float32)
    x_test_feat = {
        cond: np.hstack([feat, pkd.transform(x.astype(np.float32))]).astype(np.float32)
        for cond, feat, x in ((cond, x_test_feat[cond], x_test_raw[cond]) for cond in x_test_feat)
    }
    return x_train_feat, x_test_feat, int(train_pkd.shape[1])


def add_phcf_features(x_train_feat, x_test_feat, data):
    train_phcf = hcgf_featurize([w.astype(np.float32) for w in data["x_train"]])
    x_train_feat = np.hstack([x_train_feat, train_phcf]).astype(np.float32)
    out = {}
    for cond, feat in x_test_feat.items():
        test_phcf = hcgf_featurize([w.astype(np.float32) for w in data["x_test_by_cond"][cond]])
        out[cond] = np.hstack([feat, test_phcf]).astype(np.float32)
    return x_train_feat, out, int(train_phcf.shape[1])


def make_classifier():
    return Pipeline(
        [
            ("norm", Normalizer()),
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", RidgeClassifierCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))),
        ]
    )


def trial_metrics(y_true, y_pred, metas):
    by_trial_pred = defaultdict(list)
    truth = {}
    subject = {}
    for yt, yp, meta in zip(y_true, y_pred, metas):
        key = f"{meta['subject']}|{meta['condition']}|{meta['trial']}"
        by_trial_pred[key].append(int(yp))
        truth[key] = int(yt)
        subject[key] = meta["subject"]
    ty, py, subjects = [], [], []
    for key, preds in by_trial_pred.items():
        vals, counts = np.unique(preds, return_counts=True)
        ty.append(truth[key])
        py.append(int(vals[np.argmax(counts)]))
        subjects.append(subject[key])
    misid = int(sum(y != p for y, p in zip(ty, py)))
    return {
        "trial_accuracy": float(accuracy_score(ty, py)),
        "trial_macro_f1": float(f1_score(ty, py, average="macro")),
        "n_trials": int(len(ty)),
        "misid_trials": misid,
        "subjects_with_misid": int(len({s for s, y, p in zip(subjects, ty, py) if y != p})),
    }


def subject_condition_rows(study, variant, win_s, condition, y_true, y_pred, metas):
    rows = []
    by_subject = defaultdict(list)
    for yt, yp, meta in zip(y_true, y_pred, metas):
        by_subject[meta["subject"]].append(int(yt == yp))
    for subject, correct in by_subject.items():
        rows.append(
            {
                "study": study,
                "variant": variant,
                "win_s": float(win_s),
                "condition": condition,
                "subject": subject,
                "subject_window_accuracy": float(np.mean(correct)),
                "n_windows": int(len(correct)),
            }
        )
    return rows


def evaluate_variant(data, variant_cfg, args):
    t0 = perf_counter()
    x_train_raw = stack(data["x_train"])
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    x_test_raw = {cond: stack(data["x_test_by_cond"][cond]) for cond in sorted(data["x_test_by_cond"])}
    x_train_feat, x_test_feat, feature_dims = fit_branch_features(
        x_train_raw, y_train, x_test_raw, variant_cfg["branches"], args.seed
    )
    if variant_cfg["pkd"]:
        x_train_feat, x_test_feat, dim = add_pkd_features(x_train_feat, x_test_feat, x_train_raw, x_test_raw, args.seed, args)
        feature_dims["pkd"] = dim
    if variant_cfg["phcf"]:
        x_train_feat, x_test_feat, dim = add_phcf_features(x_train_feat, x_test_feat, data)
        feature_dims["phcf"] = dim

    clf = make_classifier()
    clf.fit(x_train_feat, y_train)
    elapsed = perf_counter() - t0

    rows = []
    subject_rows = []
    for cond in sorted(x_test_feat):
        y = np.asarray(data["y_test_by_cond"][cond], dtype=np.int64)
        pred = clf.predict(x_test_feat[cond])
        trial = trial_metrics(y, pred, data["meta_test_by_cond"][cond])
        row = {
            "study": "",
            "variant": variant_cfg["variant"],
            "label": variant_cfg["label"],
            "win_s": float(data["win_s"]),
            "condition": cond,
            "window_accuracy": float(accuracy_score(y, pred)),
            "window_macro_f1": float(f1_score(y, pred, average="macro")),
            "n_windows": int(len(y)),
            "feature_dim": int(x_train_feat.shape[1]),
            "elapsed_seconds": float(elapsed),
            "branches": "+".join(variant_cfg["branches"]),
            "pkd": int(variant_cfg["pkd"]),
            "phcf": int(variant_cfg["phcf"]),
        }
        row.update(trial)
        rows.append(row)
        subject_rows.extend(subject_condition_rows("", variant_cfg["variant"], data["win_s"], cond, y, pred, data["meta_test_by_cond"][cond]))

    overall = {"condition": "OVERALL_MEAN", "n_windows": int(sum(r["n_windows"] for r in rows)), "n_trials": int(sum(r["n_trials"] for r in rows))}
    for key in ("window_accuracy", "window_macro_f1", "trial_accuracy", "trial_macro_f1"):
        overall[key] = float(np.mean([r[key] for r in rows]))
    overall.update(
        {
            "study": "",
            "variant": variant_cfg["variant"],
            "label": variant_cfg["label"],
            "win_s": float(data["win_s"]),
            "feature_dim": int(x_train_feat.shape[1]),
            "elapsed_seconds": float(elapsed),
            "branches": "+".join(variant_cfg["branches"]),
            "pkd": int(variant_cfg["pkd"]),
            "phcf": int(variant_cfg["phcf"]),
            "misid_trials": int(sum(r["misid_trials"] for r in rows)),
            "subjects_with_misid": "",
        }
    )
    rows.append(overall)
    return rows, subject_rows, feature_dims


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(x):
    return float(x) * 100.0


def load_or_run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    comp_csv = args.out_dir / "component_ablation.csv"
    win_csv = args.out_dir / "window_duration_sweep.csv"
    subj_csv = args.out_dir / "subject_condition_accuracy.csv"
    meta_path = args.out_dir / "prism_gait_ablation_meta.json"

    if args.skip_run and comp_csv.exists() and win_csv.exists() and subj_csv.exists():
        return comp_csv, win_csv, subj_csv

    all_component = []
    all_window = []
    all_subject = []
    meta = {"args": vars(args), "component_feature_dims": {}, "window_feature_dims": {}}

    data_2s = load_original_h5_windowed(args.h5, 2.0, args.acc_std_min)
    print(f"[Data] 2s train_windows={len(data_2s['x_train'])} subjects={len(data_2s['subjects'])}", flush=True)
    for variant_cfg in COMPONENT_VARIANTS:
        print(f"[Component] {variant_cfg['variant']}", flush=True)
        rows, subject_rows, dims = evaluate_variant(data_2s, variant_cfg, args)
        for row in rows:
            row["study"] = "component"
        for row in subject_rows:
            row["study"] = "component"
        all_component.extend(rows)
        all_subject.extend(subject_rows)
        meta["component_feature_dims"][variant_cfg["variant"]] = dims
        write_csv(comp_csv, all_component)
        write_csv(subj_csv, all_subject)

    full_cfg = next(v for v in COMPONENT_VARIANTS if v["variant"] == "PRISM-Gait")
    for win_s in args.windows:
        data = data_2s if abs(win_s - 2.0) < 1e-6 else load_original_h5_windowed(args.h5, win_s, args.acc_std_min)
        print(f"[Window] {win_s:.1f}s train_windows={len(data['x_train'])}", flush=True)
        rows, subject_rows, dims = evaluate_variant(data, full_cfg, args)
        for row in rows:
            row["study"] = "window_duration"
        for row in subject_rows:
            row["study"] = "window_duration"
        all_window.extend(rows)
        all_subject.extend(subject_rows)
        meta["window_feature_dims"][str(win_s)] = dims
        write_csv(win_csv, all_window)
        write_csv(subj_csv, all_subject)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return comp_csv, win_csv, subj_csv


def plot_study(comp_csv, win_csv, subj_csv, out_dir):
    import pandas as pd

    comp = pd.read_csv(comp_csv)
    win = pd.read_csv(win_csv)
    subj = pd.read_csv(subj_csv)

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    red = "#E64B35"
    blue = "#4A7FA5"
    tan = "#B07A52"
    slate = "#8795B8"
    green = "#3C8D5A"
    purple = "#7E62A3"
    grid = "#DDDDDD"
    variants = [v["variant"] for v in COMPONENT_VARIANTS]
    labels = [v["label"] for v in COMPONENT_VARIANTS]
    colors = [slate, slate, slate, green, purple, red]

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.15, 1.0], hspace=0.50, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # Panel A: component ablation at selected 2s, trunk-sway large/small.
    y_pos = np.arange(len(variants))
    comp_trunk = comp[(comp["condition"].isin(TRUNK_CONDITIONS)) & (comp["win_s"].round(3) == 2.0)]
    large_vals = []
    small_vals = []
    for variant in variants:
        large_vals.append(pct(comp_trunk[(comp_trunk["variant"] == variant) & (comp_trunk["condition"] == "trunk_sway_large")]["trial_accuracy"].iloc[0]))
        small_vals.append(pct(comp_trunk[(comp_trunk["variant"] == variant) & (comp_trunk["condition"] == "trunk_sway_small")]["trial_accuracy"].iloc[0]))
    ax_a.hlines(y_pos, large_vals, small_vals, color="#BBBBBB", linewidth=1.2, zorder=1)
    for idx, (variant, lv, sv) in enumerate(zip(variants, large_vals, small_vals)):
        c = red if variant == "PRISM-Gait" else colors[idx]
        ax_a.scatter(lv, idx, marker="o", s=38 if variant != "PRISM-Gait" else 55, color=c, edgecolor="white", linewidth=0.6, zorder=3)
        ax_a.scatter(sv, idx, marker="D", s=38 if variant != "PRISM-Gait" else 55, color=c, edgecolor="white", linewidth=0.6, zorder=3)
        if variant == "PRISM-Gait":
            ax_a.text(lv - 1.3, idx, f"{lv:.1f}", ha="right", va="center", color=red, fontsize=7, fontweight="bold")
            ax_a.text(sv + 1.3, idx, f"{sv:.1f}", ha="left", va="center", color=red, fontsize=7, fontweight="bold")
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(labels)
    for tick, variant in zip(ax_a.get_yticklabels(), variants):
        if variant == "PRISM-Gait":
            tick.set_fontweight("bold")
    ax_a.invert_yaxis()
    ax_a.set_xlim(78, 101)
    ax_a.set_xlabel("Trial accuracy at 2 s (%)\n(circle: large; diamond: small)")
    ax_a.set_title("A  Component ablation", loc="left", fontweight="bold")
    ax_a.grid(axis="x", color=grid, linewidth=0.6, alpha=0.8)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Panel B: overall component ablation.
    overall = comp[comp["condition"] == "OVERALL_MEAN"].set_index("variant").loc[variants]
    vals = overall["trial_accuracy"].to_numpy(dtype=float) * 100.0
    bars = ax_b.bar(np.arange(len(variants)), vals, color=colors, edgecolor="black", linewidth=0.55)
    for bar, val, variant in zip(bars, vals, variants):
        ax_b.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold" if variant == "PRISM-Gait" else "normal")
    ax_b.set_xticks(np.arange(len(variants)))
    ax_b.set_xticklabels(labels, rotation=35, ha="right")
    ax_b.set_ylim(max(80, vals.min() - 3), 100.5)
    ax_b.set_ylabel("Mean trial accuracy (%)")
    ax_b.set_title("B  Overall gain", loc="left", fontweight="bold")
    ax_b.grid(axis="y", color=grid, linewidth=0.6, alpha=0.8)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Panel C: window duration variability, full PRISM-Gait only.
    win_order = [1.0, 2.0, 3.0]
    cond_names = ["trunk_sway_large", "trunk_sway_small"]
    cond_labels = {"trunk_sway_large": "Trunk sway large", "trunk_sway_small": "Trunk sway small"}
    cond_colors = {"trunk_sway_large": blue, "trunk_sway_small": tan}
    offsets = {"trunk_sway_large": -0.18, "trunk_sway_small": 0.18}
    base_pos = np.arange(len(win_order), dtype=float) + 1
    win_subj = subj[(subj["study"] == "window_duration") & (subj["variant"] == "PRISM-Gait")]
    win_rows = win[(win["condition"].isin(cond_names)) & (win["variant"] == "PRISM-Gait")]
    for cond in cond_names:
        distributions = []
        positions = []
        for pos, win_s in zip(base_pos, win_order):
            values = win_subj[(win_subj["condition"] == cond) & np.isclose(win_subj["win_s"], win_s)]["subject_window_accuracy"].to_numpy(dtype=float) * 100.0
            distributions.append(values)
            positions.append(pos + offsets[cond])
        bp = ax_c.boxplot(distributions, positions=positions, widths=0.30, patch_artist=True, showfliers=False, whis=1.5)
        for box in bp["boxes"]:
            box.set(facecolor=cond_colors[cond], alpha=0.35, edgecolor=cond_colors[cond], linewidth=0.9, hatch="///" if cond.endswith("small") else "")
        for median in bp["medians"]:
            median.set(color=cond_colors[cond], linewidth=1.2)
        for obj in bp["whiskers"] + bp["caps"]:
            obj.set(color=cond_colors[cond], linewidth=0.8, alpha=0.65)
        for pos, win_s in zip(base_pos, win_order):
            mean_val = pct(win_rows[(win_rows["condition"] == cond) & np.isclose(win_rows["win_s"], win_s)]["trial_accuracy"].iloc[0])
            ax_c.text(pos + offsets[cond], 102.0, f"{mean_val:.1f}", ha="center", va="bottom", fontsize=6.5, color=red if win_s == 2.0 else cond_colors[cond], fontweight="bold" if win_s == 2.0 else "normal")
    ax_c.set_xlim(0.45, len(win_order) + 0.55)
    ax_c.set_ylim(-4, 108)
    ax_c.set_xticks(base_pos)
    ax_c.set_xticklabels(["1 s", "2 s\n(selected)", "3 s"])
    for tick, win_s in zip(ax_c.get_xticklabels(), win_order):
        if win_s == 2.0:
            tick.set_fontweight("bold")
    ax_c.set_ylabel("Subject window accuracy (%)")
    ax_c.set_title("C  Window-duration sweep for full PRISM-Gait", loc="left", fontweight="bold")
    ax_c.text(
        0.99,
        1.03,
        "top labels: trial accuracy",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#555555",
    )
    ax_c.grid(axis="y", color=grid, linewidth=0.6, alpha=0.8)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.legend(
        handles=[
            Patch(facecolor=blue, alpha=0.35, edgecolor=blue, label=cond_labels["trunk_sway_large"]),
            Patch(facecolor=tan, alpha=0.35, edgecolor=tan, hatch="///", label=cond_labels["trunk_sway_small"]),
        ],
        frameon=False,
        fontsize=7,
        loc="lower right",
        ncol=2,
    )

    fig.suptitle("PRISM-Gait configuration ablation on original treadmill pelvis IMU", fontsize=10, fontweight="bold", y=0.985)
    fig.savefig(out_dir / "fig_prism_gait_ablation.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / "fig_prism_gait_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=H5_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--acc-std-min", type=float, default=ACC_STD_MIN)
    parser.add_argument("--windows", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--pkd-groups", type=int, default=48)
    parser.add_argument("--pkd-kernels-per-group", type=int, default=6)
    parser.add_argument("--pkd-dilations", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    comp_csv, win_csv, subj_csv = load_or_run(args)
    plot_study(comp_csv, win_csv, subj_csv, args.out_dir)
    print(f"[Saved] {args.out_dir / 'fig_prism_gait_ablation.png'}", flush=True)
    print(f"[Saved] {args.out_dir / 'fig_prism_gait_ablation.pdf'}", flush=True)


if __name__ == "__main__":
    main()
