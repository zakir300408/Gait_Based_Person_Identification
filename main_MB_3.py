# mb3_minirocket_final.py
# Multi-branch MiniROCKET for person identification with pelvis IMU
#
# IMPORTANT: This script now supports two DISTINCT evaluation modes:
#   1) CLOSED-SET (default): uses 100% of subjects for training (Baseline) and
#      tests the SAME subjects under non-Baseline conditions. This is the run
#      you should use to produce the condition-wise identification table.
#   2) OPEN-SET (--open_set_mode): holds out a fraction of subjects entirely
#      from training (e.g., 20%) and uses them only as UNSEEN IMPOSTORS for
#      verification. In this mode, identification tables are not exported to
#      avoid mixing results with the closed-set report.
#
# Example:
#   Closed-set (100% subjects; use for Table 1):
#       python mb3_minirocket_final.py
#   Open-set (80/20 verification with unseen impostors):
#       python mb3_minirocket_final.py --open_set_mode --holdout_frac 0.2 --holdout_seed 42
#
# Both runs write to separate OUT_DIRs (suffix: -closed or -openset).

import os
import json
import csv
import random
import hashlib
import platform
import sys
import argparse
from collections import defaultdict, Counter
from typing import List, Tuple, Callable, Dict, Any

import numpy as np
import h5py
from time import perf_counter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================== CONFIG (defaults) ========================
H5_PATH = r"gait_id_pelvis_62.h5"
BASE_OUT_DIR = r"baselines_out\mb3_minirocket_final"
SEED = 42

FS_HZ = 100
WIN_S = 2.0
WIN = int(WIN_S * FS_HZ)
HOP = WIN // 2
ACC_STD_MIN = 0.4

BRANCH_LIBRARY: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "raw":   lambda X: X,
    "diff":  lambda X: np.diff(X, axis=2),
    "diff2": lambda X: np.diff(X, n=2, axis=2),
}
DEFAULT_KERNELS = {"raw": 10000, "diff": 10000, "diff2": 8000}
SEED_OFFSETS = {"raw": 0, "diff": 1, "diff2": 2}
N_JOBS_ROCKET = -1
SAVE_MODEL = True

PELVIS_ACCEL = ["AccelX_PELVIS (m/s^2)", "AccelY_PELVIS (m/s^2)", "AccelZ_PELVIS (m/s^2)"]
PELVIS_GYRO   = ["GyroX_PELVIS (deg/s)", "GyroY_PELVIS (deg/s)", "GyroZ_PELVIS (deg/s)"]
PELVIS_ALLOWED = PELVIS_ACCEL + PELVIS_GYRO

# ======================== LaTeX helpers ========================
PRETTY_COND = {
    "fpa_large": "FPA (Large)",
    "fpa_small": "FPA (Small)",
    "step_width_large": "Step Width (Large)",
    "step_width_small": "Step Width (Small)",
    "trunk_sway_large": "Trunk Sway (Large)",
    "trunk_sway_small": "Trunk Sway (Small)",
    "walk_speed_large": "Walk Speed (Large)",
    "walk_speed_small": "Walk Speed (Small)",
    "dual_task": "Dual Task",
}
LATEX_TABLE_BASENAME = "table_mb3_best_preproc_condition.tex"

def _pct_or_nan(x: float) -> str:
    return "nan" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x*100:.1f}"

def write_condition_table(out_dir: str, rows: list) -> str:
    lines = []
    lines.append("\\begin{table*}")
    lines.append("\t\\caption{MB--3 condition wise performance for the best preprocessing. Values in percent. MisID trials are trials where majority vote prediction mismatched the true subject.}")
    lines.append("\t\\label{tab:best_preproc_condition}")
    lines.append("\t\\centering")
    lines.append("\t\\footnotesize")
    lines.append("\t\\renewcommand{\\arraystretch}{1.12}")
    lines.append("\t\\setlength{\\tabcolsep}{4pt}")
    lines.append("\t\\begin{tabular}{L{3.6cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm} C{2.2cm}}")
    lines.append("\t\t\\toprule")
    lines.append("\t\t\\textbf{Condition} & \\textbf{Acc. trial} & \\textbf{F1 trial} & \\textbf{Acc. window} & \\textbf{F1 window} & \\textbf{MisID trials} & \\textbf{Subjects w misID} \\\\")
    lines.append("\t\t\\midrule")
    for r in rows:
        lines.append(
            f"\t\t{r['pretty']} & {_pct_or_nan(r['acc_trial'])} & {_pct_or_nan(r['f1_trial'])} & "
            f"{_pct_or_nan(r['acc_window'])} & {_pct_or_nan(r['f1_window'])} & "
            f"{r['misid_trials']} & {r['subjects_mis']}/{r['subjects_total']} \\\\"
        )
    lines.append("\t\t\\bottomrule")
    lines.append("\t\\end{tabular}")
    lines.append("\\end{table*}")
    tex = "\n".join(lines)
    out_path = os.path.join(out_dir, LATEX_TABLE_BASENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
    return out_path

# ======================== SIMPLE LOGGER ========================
class RunLogger:
    def __init__(self, out_dir: str):
        self.lines: List[str] = []
        self.out_txt = os.path.join(out_dir, "mb3_log.txt")
        self.out_json = os.path.join(out_dir, "mb3_meta.json")
        self.meta: Dict[str, Any] = {"notes": [], "checks": {}, "train": {}, "test": {}, "env": {}, "model": {}}

    def log(self, msg: str):
        self.lines.append(msg)
        print(msg)

    def add_meta(self, key: str, val: Any):
        self.meta[key] = val

    def write_all(self):
        with open(self.out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        def _clean(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (list, tuple)): return [_clean(x) for x in obj]
            if isinstance(obj, dict): return {str(k): _clean(v) for k, v in obj.items()}
            return obj
        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(_clean(self.meta), f, indent=2)

# ======================== IMPORT MiniROCKET ========================
def _import_minirocket():
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        return MiniRocketMultivariate
    except Exception as e:
        raise ImportError(
            "Requires sktime>=0.38,<0.40. Install: pip install 'sktime>=0.38,<0.40' numpy scikit-learn h5py matplotlib"
        ) from e

# ======================== UTILS ========================
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)

def ensure_outdir(p):
    os.makedirs(p, exist_ok=True)

def list_index(h5: h5py.File):
    idx = h5["index"]
    dec = lambda arr: [x.decode("utf-8") for x in arr[()]]
    return dict(
        subject=dec(idx["subject"]),
        condition=dec(idx["condition"]),
        trial_id=dec(idx["trial_id"]),
        h5_group=dec(idx["h5_group"]),
        n_samples=idx["n_samples"][()].tolist(),
        n_channels=idx["n_channels"][()].tolist(),
        had_time=idx["had_time"][()].tolist(),
    )

def get_cols(h5: h5py.File, gp: str) -> List[str]:
    return json.loads(h5[gp].attrs["columns_json"])

def build_pelvis_canonical(h5: h5py.File, baseline_groups: List[str]) -> List[str]:
    from collections import Counter
    have = Counter()
    for gp in baseline_groups:
        cols = set(get_cols(h5, gp))
        for name in PELVIS_ALLOWED:
            if name in cols:
                have[name] += 1
    if baseline_groups:
        thresh = int(np.ceil(0.90 * len(baseline_groups)))
        keep = [n for n, c in have.items() if c >= thresh]
        if not keep:
            col_sets = [set(get_cols(h5, gp)) for gp in baseline_groups]
            inter = set.intersection(*col_sets) if col_sets else set()
            keep = [n for n in PELVIS_ALLOWED if n in inter]
        if keep:
            return [n for n in PELVIS_ALLOWED if n in keep]
    return PELVIS_ALLOWED.copy()

def load_pelvis_aligned_TxC(h5: h5py.File, gp: str, canon_cols: List[str]) -> np.ndarray:
    mat = h5[gp]["imu"][()]
    cols = get_cols(h5, gp)
    name2idx = {n: i for i, n in enumerate(cols)}
    T = mat.shape[0]
    C = len(canon_cols)
    out = np.zeros((T, C), dtype=np.float32)
    for j, name in enumerate(canon_cols):
        i = name2idx.get(name, None)
        if i is not None:
            out[:, j] = mat[:, i].astype(np.float32, copy=False)
    return out

# ======================== SEGMENTATION and NORMALIZATION ========================
def _svm(acc_xyz: np.ndarray) -> np.ndarray:
    return np.sqrt((acc_xyz ** 2).sum(axis=1))

def window_and_gate_with_meta(mat_TxC: np.ndarray, win: int, hop: int,
                              subj: str, cond: str, trial: str, h5_group: str):
    T = mat_TxC.shape[0]
    out_wins = []
    out_meta = []
    s = 0
    while s + win <= T:
        seg = mat_TxC[s:s + win, :]
        svm = _svm(seg[:, :3])
        if np.std(svm) >= ACC_STD_MIN:
            out_wins.append(seg.T.astype(np.float32, copy=False))
            out_meta.append({"subject": subj, "condition": cond, "trial": trial,
                             "group": h5_group, "s0": int(s), "s1": int(s + win)})
        s += hop
    return out_wins, out_meta

def zscore_per_window(x_CxT: np.ndarray) -> np.ndarray:
    mu = x_CxT.mean(axis=1, keepdims=True)
    sd = x_CxT.std(axis=1, keepdims=True) + 1e-8
    return (x_CxT - mu) / sd

def _as_f64_c(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)

def stack_windows_NCT(wins: List[np.ndarray]) -> np.ndarray:
    if not wins:
        return np.zeros((0, 0, 0), dtype=np.float64)
    X = np.stack(wins, axis=0).astype(np.float64, copy=False)
    return np.ascontiguousarray(X)

# ======================== Trial utilities ========================
def trial_level_from_windows(y_true_win: List[int], y_pred_win: List[int],
                             metas: List[Dict[str, Any]]
                             ) -> Tuple[List[int], List[int], List[str], List[str]]:
    by_trial_preds = defaultdict(list)
    by_trial_true = {}
    by_trial_subj = {}
    by_trial_id = {}
    for yt, yp, m in zip(y_true_win, y_pred_win, metas):
        tid = f"{m['subject']}|{m['condition']}|{m['trial']}"
        by_trial_preds[tid].append(yp)
        by_trial_true[tid] = yt
        by_trial_subj[tid] = m["subject"]
        by_trial_id[tid] = tid
    y_true_t, y_pred_t, t_subjs, t_ids = [], [], [], []
    for tid, preds in by_trial_preds.items():
        maj = Counter(preds).most_common(1)[0][0]
        y_pred_t.append(maj)
        y_true_t.append(by_trial_true[tid])
        t_subjs.append(by_trial_subj[tid])
        t_ids.append(by_trial_id[tid])
    return y_true_t, y_pred_t, t_subjs, t_ids

def subject_trial_accuracy(y_true_t: List[int], y_pred_t: List[int],
                           t_subjs: List[str]) -> Dict[str, float]:
    by_s_corr = defaultdict(list)
    for yt, yp, s in zip(y_true_t, y_pred_t, t_subjs):
        by_s_corr[s].append(1.0 if yp == yt else 0.0)
    return {s: float(np.mean(v)) for s, v in by_s_corr.items()}

def median_iqr(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    q1 = float(np.percentile(arr, 25.0))
    q3 = float(np.percentile(arr, 75.0))
    return med, q1, q3

def bootstrap_ci(values: List[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = SEED,
                 stat: str = "median") -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    n = len(v)
    stats = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        sample = v[idx]
        stats.append(np.mean(sample) if stat == "mean" else np.median(sample))
    lo = float(np.percentile(stats, 100.0 * alpha / 2.0))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi

def confusion_matrix_trial(y_true_t: List[int], y_pred_t: List[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for yt, yp in zip(y_true_t, y_pred_t):
        cm[yt, yp] += 1
    return cm

# ======================== Verification helpers ========================
def compute_roc_eer(genuine: np.ndarray, impostor: np.ndarray) -> Dict[str, Any]:
    y = np.concatenate([np.ones_like(genuine, dtype=int), np.zeros_like(impostor, dtype=int)])
    s = np.concatenate([genuine.astype(float), impostor.astype(float)])
    fpr, tpr, thr = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return {"fpr": fpr, "tpr": tpr, "thr": thr, "auc": float(roc_auc), "eer": eer, "eer_thr": float(thr[idx]), "eer_idx": int(idx)}

def save_roc_csv_png(basepath: str, roc: Dict[str, Any], title: str):
    csv_path = basepath + ".csv"
    png_path = basepath + ".png"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["FPR", "TPR", "Threshold"])
        for fp, tp, th in zip(roc["fpr"], roc["tpr"], roc["thr"]):
            w.writerow([f"{fp:.8f}", f"{tp:.8f}", f"{th:.8f}"])
        w.writerow([])
        w.writerow(["AUC", f"{roc['auc']:.6f}"])
        w.writerow(["EER", f"{roc['eer']:.6f}"])
        w.writerow(["EER_threshold", f"{roc['eer_thr']:.6f}"])
    plt.figure(figsize=(5, 5))
    plt.plot(roc["fpr"], roc["tpr"], linewidth=2, label=f"AUC = {roc['auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    i = roc["eer_idx"]
    plt.scatter([roc["fpr"][i]], [roc["tpr"][i]], s=35, zorder=5, label=f"EER = {100*roc['eer']:.2f}%")
    plt.xlabel("False Acceptance Rate")
    plt.ylabel("True Acceptance Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

# ======================== Hashing and checks ========================
def sha1_of_strings(strings: List[str]) -> str:
    h = hashlib.sha1()
    for s in strings:
        h.update((s + "\n").encode("utf-8"))
    return h.hexdigest()

def win_id(meta: Dict[str, Any]) -> str:
    return f"{meta['subject']}|{meta['condition']}|{meta['trial']}|{meta['group']}|{meta['s0']}|{meta['s1']}"

def summarize_counts(metas: List[Dict[str, Any]], by: str) -> Dict[str, int]:
    cnt = Counter([m[by] for m in metas])
    return dict(sorted(cnt.items(), key=lambda kv: kv[0]))

# ======================== MAIN ========================
def main():
    # --------- CLI args ---------
    parser = argparse.ArgumentParser(description="MB-3 MiniROCKET closed-set vs open-set evaluation.")
    parser.add_argument("--h5", default=H5_PATH, help="Path to H5 file")
    parser.add_argument("--out_dir", default=BASE_OUT_DIR, help="Base output directory")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--open_set_mode", action="store_true",
                        help="Enable open-set verification with unseen impostors (subject holdout).")
    parser.add_argument("--holdout_frac", type=float, default=0.2,
                        help="Fraction of subjects to hold out as unseen impostors (open-set mode only).")
    parser.add_argument("--holdout_seed", type=int, default=42,
                        help="Seed for selecting held-out subjects (open-set mode only).")
    args = parser.parse_args()

    # Resolve OUT_DIR based on mode
    if args.open_set_mode:
        OUT_DIR = os.path.join(args.out_dir + "-openset")
    else:
        OUT_DIR = os.path.join(args.out_dir + "-closed")

    # --------- setup ---------
    set_seed(args.seed)
    ensure_outdir(OUT_DIR)
    L = RunLogger(OUT_DIR)

    L.add_meta("task", "person identification from pelvis mounted IMU during walking with verification")
    L.add_meta("method_tag", "P7_accgyro_zwin_gate0.4_win2.0")
    L.add_meta("mode", "open-set" if args.open_set_mode else "closed-set")
    L.add_meta("preprocessing", {
        "channels": "acc+gyro", "fs_hz": FS_HZ,
        "gate": "std SVM >= 0.4 m/s^2",
        "window_s": WIN_S, "hop_s": HOP / FS_HZ,
        "normalization": "per window z score"
    })
    L.add_meta("seed", args.seed)
    L.add_meta("branches", list(BRANCH_LIBRARY.keys()))
    L.add_meta("kernels_per_branch", DEFAULT_KERNELS)

    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    try:
        import sklearn
        env["scikit_learn"] = sklearn.__version__
    except Exception:
        env["scikit_learn"] = "unknown"
    try:
        import sktime
        env["sktime"] = sktime.__version__
    except Exception:
        env["sktime"] = "unknown"
    L.add_meta("env", env)

    MiniRocketMultivariate = _import_minirocket()

    # --------- data load & subject selection ---------
    with h5py.File(args.h5, "r") as h5:
        idx = list_index(h5)
        subjects_all = sorted(set(idx["subject"]))
        has_base = {s for s, c in zip(idx["subject"], idx["condition"]) if c.lower() == "baseline"}
        keep_subjects = sorted(has_base)
        L.log(f"[Info] Subjects with baseline: {len(keep_subjects)} of {len(subjects_all)}")
        L.add_meta("subjects_total", len(subjects_all))
        L.add_meta("subjects_with_baseline", len(keep_subjects))
        L.add_meta("subject_list_all", keep_subjects)

        # Canonical pelvis channels from baseline
        base_groups = [gp for s, c, gp in zip(idx["subject"], idx["condition"], idx["h5_group"])
                       if s in has_base and c.lower() == "baseline"]
        canon_cols = build_pelvis_canonical(h5, base_groups)
        with open(os.path.join(OUT_DIR, "pelvis_canonical_cols.json"), "w", encoding="utf-8") as f:
            json.dump(canon_cols, f, indent=2)
        L.log(f"[Info] Pelvis columns: {canon_cols} C={len(canon_cols)}")
        L.add_meta("pelvis_columns", canon_cols)

        # --------- split by mode ---------
        if args.open_set_mode:
            rng = np.random.default_rng(args.holdout_seed)
            n_hold = max(1, int(len(keep_subjects) * float(args.holdout_frac)))
            holdout_subjects = set(rng.choice(keep_subjects, size=n_hold, replace=False))
            train_subjects = [s for s in keep_subjects if s not in holdout_subjects]
            L.log(f"[Split] OPEN-SET mode: enrolled={len(train_subjects)} unseen_impostors={len(holdout_subjects)}")
        else:
            holdout_subjects = set()
            train_subjects = keep_subjects.copy()
            L.log(f"[Split] CLOSED-SET mode: enrolled={len(train_subjects)} (100% subjects)")

        L.add_meta("split", {
            "mode": "open-set" if args.open_set_mode else "closed-set",
            "n_enrolled_subjects": len(train_subjects),
            "n_unseen_subjects": len(holdout_subjects),
            "unseen_subjects": sorted(list(holdout_subjects)),
            "holdout_frac": float(args.holdout_frac) if args.open_set_mode else 0.0,
            "holdout_seed": int(args.holdout_seed) if args.open_set_mode else None
        })

        # Labels are ONLY for enrolled subjects
        label2id = {s: i for i, s in enumerate(sorted(train_subjects))}
        id2label = {i: s for s, i in label2id.items()}
        L.add_meta("label2id", label2id)

        # --------- TRAIN windows: baseline of enrolled ---------
        X_train_win, train_meta = [], []
        for i, (s, c) in enumerate(zip(idx["subject"], idx["condition"])):
            if s in train_subjects and c.lower() == "baseline":
                gp = idx["h5_group"][i]
                trid = idx["trial_id"][i]
                mat_TxC = load_pelvis_aligned_TxC(h5, gp, canon_cols)
                wins, metas = window_and_gate_with_meta(
                    mat_TxC, WIN, HOP, subj=s, cond="baseline", trial=trid, h5_group=gp
                )
                for w, m in zip(wins, metas):
                    X_train_win.append(zscore_per_window(w))
                    train_meta.append(m)

        # --------- TEST sets: build per condition ---------
        conds_present = sorted({c for c in idx["condition"] if c.lower() != "baseline"})
        seen_test_sets: Dict[str, List[np.ndarray]] = {}
        seen_test_meta: Dict[str, List[Dict[str, Any]]] = {}
        unseen_test_sets: Dict[str, List[np.ndarray]] = {}
        unseen_test_meta: Dict[str, List[Dict[str, Any]]] = {}

        for cond in conds_present:
            X_seen, M_seen = [], []
            X_unseen, M_unseen = [], []
            for i, (s, c) in enumerate(zip(idx["subject"], idx["condition"])):
                if c == cond:
                    gp = idx["h5_group"][i]
                    trid = idx["trial_id"][i]
                    mat_TxC = load_pelvis_aligned_TxC(h5, gp, canon_cols)
                    wins, metas = window_and_gate_with_meta(
                        mat_TxC, WIN, HOP, subj=s, cond=cond, trial=trid, h5_group=gp
                    )
                    for w, m in zip(wins, metas):
                        w = zscore_per_window(w)
                        if s in train_subjects:
                            X_seen.append(w); M_seen.append(m)
                        elif s in holdout_subjects:
                            X_unseen.append(w); M_unseen.append(m)
            if X_seen:
                seen_test_sets[cond] = X_seen
                seen_test_meta[cond] = M_seen
            if args.open_set_mode and X_unseen:
                unseen_test_sets[cond] = X_unseen
                unseen_test_meta[cond] = M_unseen

    # --------- Prepare TRAIN panel & leakage check (seen only) ---------
    L.log("[Info] Preparing panels")
    X_train_nct = stack_windows_NCT(X_train_win)
    y_train_arr = np.asarray([label2id[m["subject"]] for m in train_meta], dtype=np.int32)
    L.log(f"[Info] Train windows={X_train_nct.shape[0]} CxT=({X_train_nct.shape[1]},{X_train_nct.shape[2]}) classes={len(set(y_train_arr.tolist()))}")
    assert X_train_nct.dtype == np.float64 and X_train_nct.flags['C_CONTIGUOUS']
    assert y_train_arr.dtype == np.int32

    train_ids = [win_id(m) for m in train_meta]
    L.add_meta("train", {
        "n_windows": len(train_meta),
        "per_subject_counts": summarize_counts(train_meta, "subject"),
        "per_trial_counts": summarize_counts(train_meta, "trial"),
        "checksum_sha1": sha1_of_strings(train_ids)
    })
    for cond, metas in seen_test_meta.items():
        test_ids = [win_id(m) for m in metas]
        inter = set(test_ids).intersection(train_ids)
        L.meta["checks"].setdefault("leakage", {})[cond] = {"test_n_windows": len(test_ids), "intersection_with_train": len(inter)}
        if len(inter) != 0:
            raise RuntimeError(f"[LEAKAGE] Found {len(inter)} overlapping window IDs between TRAIN and TEST {cond}")
    L.log("[Check] No train–test window ID intersections detected (seen/enrolled set)")

    # --------- Fit MB-3 branch transformers ---------
    MiniRocketMultivariate = _import_minirocket()
    branch_models: Dict[str, Any] = {}
    L.log("[Info] Fitting branch transformers")
    for bname, pre_fn in BRANCH_LIBRARY.items():
        nk = DEFAULT_KERNELS[bname]
        mr = MiniRocketMultivariate(
            num_kernels=int(nk), random_state=args.seed + int(SEED_OFFSETS[bname]), n_jobs=N_JOBS_ROCKET
        )
        Xb_train = pre_fn(_as_f64_c(X_train_nct))
        L.log(f"  [Fit] branch={bname} kernels={nk} Xb_train={Xb_train.shape}")
        t0 = perf_counter()
        mr.fit(Xb_train, y_train_arr)
        L.log(f"  [Done] branch={bname} fit in {perf_counter() - t0:.1f}s")
        branch_models[bname] = mr

    # --------- Transform TRAIN to features & fit classifier ---------
    L.log("[Info] Transforming TRAIN features")
    feats_train = []
    feat_dims = {}
    for bname, pre_fn in BRANCH_LIBRARY.items():
        Xb = pre_fn(_as_f64_c(X_train_nct))
        Fb = branch_models[bname].transform(Xb)
        feats_train.append(Fb)
        feat_dims[bname] = int(Fb.shape[1])
        L.log(f"  [Cache] branch={bname} F_train={Fb.shape[1]}")
    X_train_feat = np.concatenate(feats_train, axis=1)
    total_F = int(sum(feat_dims.values()))
    L.log(f"[Info] Total features {total_F}")
    L.meta["model"]["feature_dims_per_branch"] = feat_dims
    L.meta["model"]["total_features"] = total_F

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("ridge", RidgeClassifierCV(alphas=(0.1, 1.0, 10.0)))
    ])
    L.log("[Info] Fitting classifier")
    t0 = perf_counter()
    clf.fit(X_train_feat, y_train_arr)
    L.log(f"[Done] Classifier fit in {perf_counter() - t0:.1f}s")

    # --------- Outputs ---------
    ensure_outdir(OUT_DIR)
    out_csv_legacy  = os.path.join(OUT_DIR, "results_mb3_minirocket.csv")
    out_csv_summary = os.path.join(OUT_DIR, "results_mb3_condition_summary.csv")
    out_csv_subjects= os.path.join(OUT_DIR, "results_mb3_subject_by_condition.csv")

    # In OPEN-SET mode we avoid exporting the condition LaTeX to keep closed-set table pristine
    export_table = (not args.open_set_mode)

    if not args.open_set_mode:
        with open(out_csv_legacy, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["transform", "branches", "kernels", "condition",
                        "acc_window", "f1_window", "acc_trial", "f1_trial",
                        "win_s", "seed", "train_windows", "total_features"])

        with open(out_csv_summary, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "condition", "n_subjects", "n_trials",
                "acc_window", "f1_window",
                "acc_trial_overall", "f1_trial_overall",
                "misid_trials", "subjects_with_misid"
            ])

        with open(out_csv_subjects, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["condition", "subject", "n_trials", "subject_acc", "subject_misid"])

    kernels_desc = ";".join([f"{b}:{DEFAULT_KERNELS[b]}" for b in BRANCH_LIBRARY])
    overall_results = []

    # Accumulators for CLOSED-SET verification (seen/enrolled only)
    all_win_scores = []
    all_win_true   = []
    all_win_metas  = []
    by_trial_scores = defaultdict(list)
    by_trial_true   = {}

    latex_rows = []

    # --------- Evaluate per condition on SEEN (enrolled) subjects ---------
    for cond, Xc_win in seen_test_sets.items():
        metas = seen_test_meta[cond]
        if not Xc_win:
            continue
        Xc_nct = stack_windows_NCT(Xc_win)
        y_true_win = np.asarray([label2id[m["subject"]] for m in metas], dtype=np.int32)

        feats_c = []
        for bname, pre_fn in BRANCH_LIBRARY.items():
            Xb = pre_fn(_as_f64_c(Xc_nct))
            Fb = branch_models[bname].transform(Xb)
            feats_c.append(Fb)
        Xc_feat = np.concatenate(feats_c, axis=1)

        y_pred_win = clf.predict(Xc_feat).tolist()
        scores_win = clf.decision_function(Xc_feat)

        all_win_scores.append(scores_win)
        all_win_true.append(y_true_win)
        all_win_metas.extend(metas)

        for yt, sw, m in zip(y_true_win, scores_win, metas):
            tid = f"{m['subject']}|{m['condition']}|{m['trial']}"
            by_trial_scores[tid].append(sw)
            by_trial_true[tid] = int(yt)

        acc_w = accuracy_score(y_true_win, y_pred_win) if len(y_true_win) else float("nan")
        try:
            f1_w = f1_score(y_true_win, y_pred_win, average="macro")
        except Exception:
            f1_w = float("nan")

        y_true_t, y_pred_t, t_subjs, t_ids = trial_level_from_windows(y_true_win, y_pred_win, metas)
        acc_t = accuracy_score(y_true_t, y_pred_t) if len(y_true_t) else float("nan")
        try:
            f1_t = f1_score(y_true_t, y_pred_t, average="macro")
        except Exception:
            f1_t = float("nan")

        # Export per-subject rows only in CLOSED-SET mode
        if not args.open_set_mode:
            per_subj_acc = subject_trial_accuracy(y_true_t, y_pred_t, t_subjs)
            subj_list = sorted(per_subj_acc.keys())
            with open(out_csv_subjects, "a", newline="", encoding="utf-8") as fsub:
                wsub = csv.writer(fsub)
                trials_per_subj = Counter(t_subjs)
                for s in subj_list:
                    wsub.writerow([cond, s, trials_per_subj.get(s, 0), f"{per_subj_acc[s]:.6f}", f"{1.0 - per_subj_acc[s]:.6f}"])

        misid_trials = int(np.sum([int(yp != yt) for yt, yp in zip(y_true_t, y_pred_t)]))
        subj_mis_set = set([s for (s, yt, yp) in zip(t_subjs, y_true_t, y_pred_t) if yp != yt])
        subjects_total = len(set(t_subjs))

        if not args.open_set_mode:
            with open(out_csv_summary, "a", newline="", encoding="utf-8") as fs:
                ws = csv.writer(fs)
                ws.writerow([
                    cond, subjects_total, len(y_true_t),
                    f"{acc_w:.6f}", f"{f1_w:.6f}",
                    f"{acc_t:.6f}", f"{f1_t:.6f}",
                    misid_trials, len(subj_mis_set)
                ])

            with open(out_csv_legacy, "a", newline="", encoding="utf-8") as f_legacy:
                wl = csv.writer(f_legacy)
                wl.writerow(["MB-3", "+".join(BRANCH_LIBRARY.keys()), kernels_desc, cond,
                             f"{acc_w:.4f}", f"{f1_w:.4f}", f"{acc_t:.4f}", f"{f1_t:.4f}",
                             f"{WIN_S:.1f}", SEED, X_train_nct.shape[0], total_F])

        L.log(f"  MB-3 | {cond:20s} win-acc={acc_w:.3f} win-f1={f1_w:.3f} trial-acc={acc_t:.3f} trial-f1={f1_t:.3f} misID_trials={misid_trials} subjects_mis={len(subj_mis_set)}/{subjects_total}")

        if export_table:
            latex_rows.append({
                "cond": cond,
                "pretty": PRETTY_COND.get(cond, cond),
                "acc_trial": acc_t, "f1_trial": f1_t,
                "acc_window": acc_w, "f1_window": f1_w,
                "misid_trials": misid_trials,
                "subjects_mis": len(subj_mis_set),
                "subjects_total": subjects_total,
            })

    # OVERALL means (seen/enrolled), closed-set only
    if (not args.open_set_mode) and len(overall_results) == 0 and seen_test_sets:
        # overall_results list wasn't filled above; compute from accumulated metrics
        pass  # kept for compatibility if you decide to store per-cond tuples again

    if export_table and latex_rows:
        latex_rows_sorted = sorted(latex_rows, key=lambda r: PRETTY_COND.get(r["cond"], r["cond"]))
        tex_path = write_condition_table(OUT_DIR, latex_rows_sorted)
        L.log(f"[Saved] LaTeX condition table: {tex_path}")

    # --------- CLOSED-SET verification (windows & trials) ---------
    if all_win_scores:
        all_scores = np.vstack(all_win_scores)               # [Nwin, C_seen]
        all_true   = np.concatenate(all_win_true).astype(int)
        genuine_win = all_scores[np.arange(all_scores.shape[0]), all_true]
        mask = np.ones_like(all_scores, dtype=bool)
        mask[np.arange(all_scores.shape[0]), all_true] = False
        impostor_win_enrolled = all_scores[mask].ravel()
        roc_win = compute_roc_eer(genuine_win, impostor_win_enrolled)
        L.log(f"[Verification CLOSED-SET] WINDOWS AUC={roc_win['auc']:.4f} EER={100*roc_win['eer']:.2f}% thr={roc_win['eer_thr']:.4f}")
        save_roc_csv_png(os.path.join(OUT_DIR, "roc_overall_windows_closed"), roc_win, "ROC Windows (enrolled impostors)")

    if by_trial_scores:
        trial_ids = sorted(by_trial_scores.keys())
        trial_scores = []
        trial_true = []
        for tid in trial_ids:
            S = np.vstack(by_trial_scores[tid])  # [Nwin_trial, C_seen]
            trial_scores.append(S.mean(axis=0))
            trial_true.append(by_trial_true[tid])
        trial_scores = np.vstack(trial_scores)
        trial_true = np.asarray(trial_true, dtype=int)
        genuine_t = trial_scores[np.arange(trial_scores.shape[0]), trial_true]
        mask_t = np.ones_like(trial_scores, dtype=bool)
        mask_t[np.arange(trial_scores.shape[0]), trial_true] = False
        impostor_t_enrolled = trial_scores[mask_t].ravel()
        roc_t = compute_roc_eer(genuine_t, impostor_t_enrolled)
        L.log(f"[Verification CLOSED-SET] TRIALS AUC={roc_t['auc']:.4f} EER={100*roc_t['eer']:.2f}% thr={roc_t['eer_thr']:.4f}")
        save_roc_csv_png(os.path.join(OUT_DIR, "roc_overall_trials_closed"), roc_t, "ROC Trials (enrolled impostors)")

    # --------- OPEN-SET extension: add unseen-subject impostors (separate run only) ---------
    if args.open_set_mode and unseen_test_sets and all_win_scores:
        unseen_win_scores = []
        for cond, Xc_win_unseen in unseen_test_sets.items():
            if not Xc_win_unseen:
                continue
            X_unseen_nct = stack_windows_NCT(Xc_win_unseen)
            feats_unseen = []
            for bname, pre_fn in BRANCH_LIBRARY.items():
                Xb = pre_fn(_as_f64_c(X_unseen_nct))
                Fb = branch_models[bname].transform(Xb)
                feats_unseen.append(Fb)
            X_unseen_feat = np.concatenate(feats_unseen, axis=1)
            scores_unseen = clf.decision_function(X_unseen_feat)  # [Nunseen, C_seen]
            unseen_win_scores.append(scores_unseen)

        if unseen_win_scores:
            all_scores_seen = np.vstack(all_win_scores)
            all_true_seen = np.concatenate(all_win_true).astype(int)
            genuine = all_scores_seen[np.arange(all_scores_seen.shape[0]), all_true_seen]
            mask_seen = np.ones_like(all_scores_seen, dtype=bool)
            mask_seen[np.arange(all_scores_seen.shape[0]), all_true_seen] = False
            impostor_enrolled = all_scores_seen[mask_seen].ravel()

            impostor_unseen = np.vstack(unseen_win_scores).ravel()
            impostor_open = np.concatenate([impostor_enrolled, impostor_unseen])

            roc_open = compute_roc_eer(genuine, impostor_open)
            L.log(f"[Verification OPEN-SET] WINDOWS AUC={roc_open['auc']:.4f} EER={100*roc_open['eer']:.2f}% thr={roc_open['eer_thr']:.4f}")
            save_roc_csv_png(os.path.join(OUT_DIR, "roc_overall_windows_openset"), roc_open, "ROC Windows (enrolled + unseen impostors)")

    # --------- Save model package ---------
    if SAVE_MODEL:
        try:
            import joblib
            package = {
                "branches": list(BRANCH_LIBRARY.keys()),
                "kernels": DEFAULT_KERNELS,
                "seed": args.seed,
                "seed_offsets": SEED_OFFSETS,
                "branch_models": branch_models,
                "classifier": clf,
                "label2id": label2id,
                "id2label": id2label,
                "pelvis_columns": canon_cols,
                "preprocessing": {
                    "fs_hz": FS_HZ,
                    "win_s": WIN_S,
                    "hop_s": HOP / FS_HZ,
                    "energy_gate_std_svm_min": ACC_STD_MIN,
                    "normalization": "per window z score"
                },
                "split": {
                    "mode": "open-set" if args.open_set_mode else "closed-set",
                    "enrolled_subjects": sorted(train_subjects),
                    "unseen_subjects": sorted(list(holdout_subjects))
                }
            }
            model_path = os.path.join(OUT_DIR, "mb3_model.joblib")
            joblib.dump(package, model_path)
            L.log(f"[Saved] Model package: {model_path}")
        except Exception as e:
            L.log(f"[Warn] Could not save model package joblib {type(e).__name__}: {e}")

    # --------- Final logs ---------
    L.write_all()
    print(f"\n[Done] Saved:")
    if not args.open_set_mode:
        print(f" - CSV legacy        {out_csv_legacy}")
        print(f" - CSV summary       {out_csv_summary}")
        print(f" - CSV by subject    {out_csv_subjects}")
        print(f" - LaTeX table       {os.path.join(OUT_DIR, LATEX_TABLE_BASENAME)}")
        print(f" - ROC windows (closed) {os.path.join(OUT_DIR, 'roc_overall_windows_closed.csv')} and .png")
        print(f" - ROC trials (closed)  {os.path.join(OUT_DIR, 'roc_overall_trials_closed.csv')} and .png")
    else:
        print(f" - ROC windows (closed)  {os.path.join(OUT_DIR, 'roc_overall_windows_closed.csv')} and .png")
        print(f" - ROC trials (closed)   {os.path.join(OUT_DIR, 'roc_overall_trials_closed.csv')} and .png")
        print(f" - ROC windows (open-set){os.path.join(OUT_DIR, 'roc_overall_windows_openset.csv')} and .png")
    print(f" - LOG               {L.out_txt}")
    print(f" - META              {L.out_json}")
    if SAVE_MODEL:
        print(f" - MODEL             {os.path.join(OUT_DIR, 'mb3_model.joblib')}")
    print("\nNotes")
    if not args.open_set_mode:
        print(" - CLOSED-SET: 100% of subjects are enrolled (train on Baseline, test on their altered conditions).")
        print(" - Use this run for condition-wise identification tables.")
    else:
        print(" - OPEN-SET: ~80/20 split by subjects (configurable); held-out subjects are UNSEEN impostors for verification.")
        print(" - Identification tables are intentionally NOT exported in open-set mode.")
    print(" - Zero train–test window overlap is enforced via hashed window IDs.")
    print(" - Verification uses classifier decision scores; trial-level uses window-averaged scores.")
    print(" - Per-window z-score normalization has no leakage.")

if __name__ == "__main__":
    main()
