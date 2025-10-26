import os, json, csv, random, hashlib, platform, sys
from collections import defaultdict, Counter
from typing import List, Tuple, Callable, Dict, Any

import numpy as np
import h5py
from time import perf_counter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

H5_PATH = r"gait_id_pelvis_62.h5"
OUT_DIR = r"baselines_out/mb_minirocket_ablation_mb3"
SEED = 42

BOOTSTRAP_REPS = 2000
BOOTSTRAP_ALPHA = 0.05

EVAL_CONDITIONS = [
    "fpa_large", "fpa_small",
    "step_width_large", "step_width_small",
    "trunk_sway_large", "trunk_sway_small",
    "walk_speed_large", "walk_speed_small",
    "dual_task"
]

BRANCH_LIBRARY: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "raw": lambda X: X,
    "diff": lambda X: np.diff(X, axis=2),
    "diff2": lambda X: np.diff(X, n=2, axis=2),
}

DEFAULT_KERNELS = {"raw": 10000, "diff": 10000, "diff2": 8000}

BRANCH_EXPERIMENTS = [
    {"name": "raw", "branches": ["raw"]},
    {"name": "diff", "branches": ["diff"]},
    {"name": "diff2", "branches": ["diff2"]},
    {"name": "raw+diff", "branches": ["raw", "diff"]},
    {"name": "raw+diff2", "branches": ["raw", "diff2"]},
    {"name": "diff+diff2", "branches": ["diff", "diff2"]},
    {"name": "MB-3_raw+diff+diff2", "branches": ["raw", "diff", "diff2"]},
]

PREPROC_EXPERIMENTS = [
    {"name": "W1_accgyro_zwin_gate0.4_win1.0",
     "channels": "acc+gyro", "norm": "zscore_window", "gate_std_svm_min": 0.4, "win_s": 1.0},
    {"name": "W2_accgyro_zwin_gate0.4_win2.0",
     "channels": "acc+gyro", "norm": "zscore_window", "gate_std_svm_min": 0.4, "win_s": 2.0},
    {"name": "W3_accgyro_zwin_gate0.4_win3.0",
     "channels": "acc+gyro", "norm": "zscore_window", "gate_std_svm_min": 0.4, "win_s": 3.0},
]

N_JOBS_ROCKET = -1
DEBUG_FAST_HEAD = False

PELVIS_ACCEL = ["AccelX_PELVIS (m/s^2)", "AccelY_PELVIS (m/s^2)", "AccelZ_PELVIS (m/s^2)"]
PELVIS_GYRO = ["GyroX_PELVIS (deg/s)", "GyroY_PELVIS (deg/s)", "GyroZ_PELVIS (deg/s)"]
PELVIS_ALLOWED = PELVIS_ACCEL + PELVIS_GYRO


class RunLogger:
    def __init__(self, out_dir: str):
        self.lines: List[str] = []
        self.out_txt = os.path.join(out_dir, "ablation_log.txt")
        self.out_json = os.path.join(out_dir, "ablation_meta.json")
        self.meta: Dict[str, Any] = {"notes": [], "checks": {}, "train": {}, "test": {}, "env": {}, "experiments": []}

    def log(self, msg: str):
        self.lines.append(msg)
        print(msg)

    def add_meta(self, key: str, val: Any):
        self.meta[key] = val

    def add_exp_meta(self, exp_meta: Dict[str, Any]):
        self.meta["experiments"].append(exp_meta)

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


def _import_minirocket():
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        return MiniRocketMultivariate
    except Exception as e:
        raise ImportError(
            "Requires sktime>=0.38. Try: pip install -U 'sktime>=0.38,<0.40' numpy scikit-learn h5py"
        ) from e


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


def load_pelvis_aligned_TxC(h5: h5py.File, gp: str, canon_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
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
    return out, canon_cols


def _svm(acc_xyz: np.ndarray) -> np.ndarray:
    return np.sqrt((acc_xyz ** 2).sum(axis=1))


def _norm_apply(x_CxT: np.ndarray, mode: str, mu=None, sd=None) -> np.ndarray:
    if mode == "none":
        return x_CxT
    if mode == "demean_window":
        mu = x_CxT.mean(axis=1, keepdims=True)
        return x_CxT - mu
    if mode == "zscore_window":
        mu = x_CxT.mean(axis=1, keepdims=True)
        sd = x_CxT.std(axis=1, keepdims=True) + 1e-8
        return (x_CxT - mu) / sd
    if mode == "zscore_trial":
        assert mu is not None and sd is not None
        return (x_CxT - mu) / (sd + 1e-8)
    raise ValueError(f"Unknown normalization mode: {mode}")


def segment_windows_with_preproc(
    mat_TxC: np.ndarray,
    cols: List[str],
    subj: str, cond: str, trial: str, h5_group: str,
    pre: Dict[str, Any]
):
    fs = 100
    win = int(float(pre["win_s"]) * fs)
    hop = win // 2
    gate_thr = pre["gate_std_svm_min"]
    use_acc = pre["channels"] in ("acc", "acc+gyro")
    use_gyro = pre["channels"] in ("gyro", "acc+gyro")
    acc_idx = [cols.index(n) for n in PELVIS_ACCEL if n in cols]
    gyr_idx = [cols.index(n) for n in PELVIS_GYRO if n in cols]
    keep_idx = []
    if use_acc:
        keep_idx += acc_idx
    if use_gyro:
        keep_idx += gyr_idx
    if not keep_idx:
        return [], []
    mu_trial = None
    sd_trial = None
    if pre["norm"] == "zscore_trial":
        Xtrial = mat_TxC[:, keep_idx].astype(np.float32, copy=False)
        mu_trial = Xtrial.mean(axis=0, keepdims=True).T
        sd_trial = Xtrial.std(axis=0, keepdims=True).T
    T = mat_TxC.shape[0]
    out_wins, out_meta = [], []
    s = 0
    while s + win <= T:
        seg_TxC = mat_TxC[s:s + win, :]
        pass_gate = True
        if gate_thr is not None and len(acc_idx) >= 3:
            svm = _svm(seg_TxC[:, acc_idx])
            pass_gate = (np.std(svm) >= float(gate_thr))
        if pass_gate:
            seg_keep = seg_TxC[:, keep_idx].T.astype(np.float32, copy=False)
            if pre["norm"] == "zscore_trial":
                seg_keep = _norm_apply(seg_keep, "zscore_trial", mu=mu_trial, sd=sd_trial)
            else:
                seg_keep = _norm_apply(seg_keep, pre["norm"])
            out_wins.append(seg_keep)
            out_meta.append({"subject": subj, "condition": cond, "trial": trial,
                             "group": h5_group, "s0": int(s), "s1": int(s + win)})
        s += hop
    return out_wins, out_meta


def _as_f64_c(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def stack_windows_NCT(wins: List[np.ndarray]) -> np.ndarray:
    if not wins:
        return np.zeros((0, 0, 0), dtype=np.float64)
    X = np.stack(wins, axis=0).astype(np.float64, copy=False)
    return np.ascontiguousarray(X)


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


def paired_bootstrap_acc_delta(trial_ok_base: Dict[str, int],
                               trial_ok_alt: Dict[str, int],
                               reps: int = 1000, seed: int = SEED, alpha: float = BOOTSTRAP_ALPHA):
    rng = np.random.default_rng(seed)
    trials = sorted(set(trial_ok_base.keys()) & set(trial_ok_alt.keys()))
    if not trials:
        return float("nan"), (float("nan"), float("nan"))
    n = len(trials)
    base_arr = np.array([trial_ok_base[t] for t in trials], dtype=float)
    alt_arr = np.array([trial_ok_alt[t] for t in trials], dtype=float)
    deltas = []
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        deltas.append((alt_arr[idx].mean() - base_arr[idx].mean()))
    deltas = np.array(deltas)
    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return float(deltas.mean()), (lo, hi)


def _init_mr(num_kernels: int, seed_offset: int, MiniRocketMultivariate, n_jobs: int):
    return MiniRocketMultivariate(
        num_kernels=int(num_kernels),
        random_state=SEED + int(seed_offset),
        n_jobs=n_jobs
    )


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


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def pretty_cond(cond: str) -> str:
    mapping = {
        "fpa_large": "FPA (Large)",
        "fpa_small": "FPA (Small)",
        "step_width_large": "Step Width (Large)",
        "step_width_small": "Step Width (Small)",
        "trunk_sway_large": "Trunk Sway (Large)",
        "trunk_sway_small": "Trunk Sway (Small)",
        "walk_speed_large": "Walk Speed (Large)",
        "walk_speed_small": "Walk Speed (Small)",
        "dual_task": "Dual Task",
        "baseline": "Baseline",
    }
    return mapping.get(cond, cond.replace("_", " ").title())


def pct(x):
    try:
        return f"{100.0 * float(x):.1f}"
    except:
        return ""


def main():
    set_seed(SEED)
    ensure_outdir(OUT_DIR)
    L = RunLogger(OUT_DIR)
    L.add_meta("task", "person identification from pelvis-mounted IMU during walking (subject classification)")
    L.add_meta("seed", SEED)
    L.add_meta("branch_experiments", BRANCH_EXPERIMENTS)
    L.add_meta("preproc_experiments", PREPROC_EXPERIMENTS)
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
    with h5py.File(H5_PATH, "r") as h5:
        idx = list_index(h5)
        subjects_all = sorted(set(idx["subject"]))
        has_base = {s for s, c in zip(idx["subject"], idx["condition"]) if c.lower() == "baseline"}
        keep_subjects = sorted(has_base)
        L.log(f"[Info] Subjects with baseline: {len(keep_subjects)} / total {len(subjects_all)}")
        L.add_meta("subjects_total", len(subjects_all))
        L.add_meta("subjects_with_baseline", len(keep_subjects))
        L.add_meta("subject_list", keep_subjects)
        base_groups = [gp for s, c, gp in zip(idx["subject"], idx["condition"], idx["h5_group"])
                       if s in has_base and c.lower() == "baseline"]
        canon_cols = build_pelvis_canonical(h5, base_groups)
        with open(os.path.join(OUT_DIR, "pelvis_canonical_cols.json"), "w", encoding="utf-8") as f:
            json.dump(canon_cols, f, indent=2)
        L.log(f"[Info] Pelvis columns: {canon_cols} (C={len(canon_cols)})")
        L.add_meta("pelvis_columns", canon_cols)
        label2id = {s: i for i, s in enumerate(keep_subjects)}
        id2label = {i: s for s, i in label2id.items()}
        L.add_meta("label2id", label2id)
        rows_by = defaultdict(list)
        for i in range(len(idx["subject"])):
            rows_by[(idx["subject"][i], idx["condition"][i])].append(i)
        ensure_outdir(OUT_DIR)
        out_summary = os.path.join(OUT_DIR, "results_ablation_summary.csv")
        out_trials = os.path.join(OUT_DIR, "results_ablation_trial_preds.csv")
        out_subject = os.path.join(OUT_DIR, "results_ablation_subject_by_condition.csv")
        out_tex_all = os.path.join(OUT_DIR, "ablation_tables.tex")
        with open(out_summary, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "preproc", "branches", "kernels", "condition",
                "acc_window", "f1_window", "acc_trial", "f1_trial",
                "win_s", "seed", "train_windows", "total_features", "overall_flag"
            ])
        with open(out_trials, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "preproc", "branches", "condition", "trial_id", "subject_true", "subject_pred", "correct"
            ])
        with open(out_subject, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "preproc", "branches", "condition", "subject", "n_trials", "subject_acc", "subject_misid"
            ])
        main_rows_cond = []
        branch_rows_W2 = []
        window_rows_MB3 = []
        baseline_key = None
        perexp_trial_ok: Dict[str, Dict[str, int]] = {}
        for pre in PREPROC_EXPERIMENTS:
            pre_name = pre["name"]
            fs = 100
            win = int(float(pre["win_s"]) * fs)
            hop = win // 2
            L.log(f"\n[Preproc] {pre_name} | channels={pre['channels']} norm={pre['norm']} gate={pre['gate_std_svm_min']} win_s={pre['win_s']}")
            X_train_win, train_meta = [], []
            for s in keep_subjects:
                for i in rows_by[(s, "baseline")]:
                    gp = idx["h5_group"][i]
                    trid = idx["trial_id"][i]
                    mat_TxC, cols = load_pelvis_aligned_TxC(h5, gp, canon_cols)
                    wins, metas = segment_windows_with_preproc(
                        mat_TxC, cols, subj=s, cond="baseline", trial=trid, h5_group=gp, pre=pre
                    )
                    X_train_win.extend(wins)
                    train_meta.extend(metas)
            X_train_nct = stack_windows_NCT(X_train_win)
            y_train_arr = np.asarray([label2id[m["subject"]] for m in train_meta], dtype=np.int32)
            L.log(f"[Train] windows={X_train_nct.shape[0]} | (C,T)=({X_train_nct.shape[1]},{X_train_nct.shape[2]})")
            conds_present = sorted(set([c for c in idx["condition"] if c.lower() != "baseline"]))
            eval_conds = [c for c in EVAL_CONDITIONS if c in conds_present] if EVAL_CONDITIONS else conds_present
            test_sets = {}
            test_meta_by_cond: Dict[str, List[Dict[str, Any]]] = {}
            for cond in eval_conds:
                Xc, metas_c = [], []
                for i, (s, c) in enumerate(zip(idx["subject"], idx["condition"])):
                    if s in keep_subjects and c == cond:
                        gp = idx["h5_group"][i]
                        trid = idx["trial_id"][i]
                        mat_TxC, cols = load_pelvis_aligned_TxC(h5, gp, canon_cols)
                        wins, metas = segment_windows_with_preproc(
                            mat_TxC, cols, subj=s, cond=cond, trial=trid, h5_group=gp, pre=pre
                        )
                        Xc.extend(wins)
                        metas_c.extend(metas)
                if Xc:
                    test_sets[cond] = Xc
                    test_meta_by_cond[cond] = metas_c
            train_ids = [win_id(m) for m in train_meta]
            for cond, metas in test_meta_by_cond.items():
                test_ids = [win_id(m) for m in metas]
                inter = set(test_ids).intersection(train_ids)
                L.meta["checks"].setdefault("leakage", {})[f"{pre_name}|{cond}"] = {
                    "test_n_windows": len(test_ids), "intersection_with_train": len(inter)
                }
                if len(inter) != 0:
                    raise RuntimeError(f"[LEAKAGE] Preproc {pre_name}: {len(inter)} overlapping window IDs ({cond}).")
            L.log("[Check] No train/test window-ID intersections detected.")
            MiniRocketMultivariate = _import_minirocket()
            seed_offsets = {"raw": 0, "diff": 1, "diff2": 2}
            branch_models = {}
            for bname, pre_fn in BRANCH_LIBRARY.items():
                num_k = DEFAULT_KERNELS[bname]
                mr = _init_mr(num_k, seed_offsets[bname], MiniRocketMultivariate, N_JOBS_ROCKET)
                Xb_train = pre_fn(_as_f64_c(X_train_nct))
                L.log(f"  [Fit MR] branch={bname} kernels={num_k} | Xb_train={Xb_train.shape}")
                t0 = perf_counter()
                mr.fit(Xb_train, y_train_arr)
                L.log(f"  [Done] branch={bname} fit in {perf_counter() - t0:.1f}s")
                branch_models[bname] = mr
            train_feats_by_branch: Dict[str, np.ndarray] = {}
            for bname, pre_fn in BRANCH_LIBRARY.items():
                Xb_train = pre_fn(_as_f64_c(X_train_nct))
                Fb = branch_models[bname].transform(Xb_train)
                train_feats_by_branch[bname] = Fb
                L.log(f"  [Cache TRAIN] {bname} -> F={Fb.shape[1]}")
            test_feats_by_cond_branch: Dict[str, Dict[str, np.ndarray]] = {cond: {} for cond in test_sets}
            test_truth_by_cond: Dict[str, np.ndarray] = {}
            test_metas_by_cond: Dict[str, List[Dict[str, Any]]] = {}
            for cond, Xc_win in test_sets.items():
                metas = test_meta_by_cond[cond]
                Xc_nct = stack_windows_NCT(Xc_win)
                test_truth_by_cond[cond] = np.asarray([label2id[m["subject"]] for m in metas], dtype=np.int32)
                test_metas_by_cond[cond] = metas
                for bname, pre_fn in BRANCH_LIBRARY.items():
                    Xb = pre_fn(_as_f64_c(Xc_nct))
                    Fb = branch_models[bname].transform(Xb)
                    test_feats_by_cond_branch[cond][bname] = Fb
                L.log(f"  [Cache TEST] {cond} n={len(metas)}")
            with open(out_summary, "a", newline="", encoding="utf-8") as fsum, \
                 open(out_trials, "a", newline="", encoding="utf-8") as ftri, \
                 open(out_subject, "a", newline="", encoding="utf-8") as fsub:
                ws = csv.writer(fsum)
                wt = csv.writer(ftri)
                wsub = csv.writer(fsub)
                exp_trial_ok: Dict[str, Dict[str, int]] = {}
                for exp in BRANCH_EXPERIMENTS:
                    name = exp["name"]
                    branches = exp["branches"]
                    X_train_feat_list = [train_feats_by_branch[b] for b in branches]
                    X_train_feat = np.concatenate(X_train_feat_list, axis=1)
                    total_F = int(sum(train_feats_by_branch[b].shape[1] for b in branches))
                    kernels_desc = ";".join([f"{b}:{DEFAULT_KERNELS[b]}" for b in branches])
                    L.log(f"\n[Exp] {pre_name} | {name} | branches={branches} | total_F={total_F}")
                    clf = Pipeline([
                        ("scaler", StandardScaler(with_mean=False)),
                        ("ridge", RidgeClassifier(alpha=1.0)) if DEBUG_FAST_HEAD else ("ridge", RidgeClassifierCV(alphas=(0.1, 1.0, 10.0)))
                    ])
                    clf.fit(X_train_feat, y_train_arr)
                    per_cond_stats = []
                    mis_trials_sum = 0
                    mis_subjects_set = set()
                    pertrial_ok: Dict[str, int] = {}
                    for cond in test_sets:
                        feats_c_list = [test_feats_by_cond_branch[cond][b] for b in branches]
                        Xc_feat = np.concatenate(feats_c_list, axis=1)
                        metas = test_metas_by_cond[cond]
                        y_true_win = test_truth_by_cond[cond]
                        y_pred_win = clf.predict(Xc_feat).tolist()
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
                        for yt, yp, tid in zip(y_true_t, y_pred_t, t_ids):
                            wt.writerow([pre_name, name, cond, tid, id2label[yt], id2label[yp], int(1 if yt == yp else 0)])
                        per_subj_acc = subject_trial_accuracy(y_true_t, y_pred_t, t_subjs)
                        trials_per_subj = Counter(t_subjs)
                        for s_name, acc in per_subj_acc.items():
                            wsub.writerow([pre_name, name, cond, s_name, trials_per_subj.get(s_name, 0), f"{acc:.6f}", f"{1.0 - acc:.6f}"])
                        for yt, yp, tid, subj in zip(y_true_t, y_pred_t, t_ids, t_subjs):
                            ok = (1 if yt == yp else 0)
                            pertrial_ok[tid] = ok
                            if not ok:
                                mis_trials_sum += 1
                                mis_subjects_set.add(subj)
                        per_cond_stats.append((acc_w, f1_w, acc_t, f1_t))
                        ws.writerow([pre_name, name, kernels_desc, cond,
                                     f"{acc_w:.4f}", f"{f1_w:.4f}", f"{acc_t:.4f}", f"{f1_t:.4f}",
                                     f"{float(pre['win_s']):.1f}", SEED, X_train_nct.shape[0], total_F, ""])
                        L.log(f"  {name:28s} | {cond:20s} acc_w={acc_w:.3f} f1_w={f1_w:.3f} | acc_t={acc_t:.3f} f1_t={f1_t:.3f}")
                    if per_cond_stats:
                        means = np.nanmean(np.array(per_cond_stats), axis=0)
                        ws.writerow([pre_name, name, kernels_desc, "OVERALL_MEAN",
                                     f"{means[0]:.4f}", f"{means[1]:.4f}", f"{means[2]:.4f}", f"{means[3]:.4f}",
                                     f"{float(pre['win_s']):.1f}", SEED, X_train_nct.shape[0], total_F, "overall"])
                        L.log(f"  -> OVERALL MEAN: acc_w={means[0]:.3f} f1_w={means[1]:.3f} | acc_t={means[2]:.3f} f1_t={means[3]:.3f}")
                        if pre_name.startswith("W2_"):
                            branch_rows_W2.append({
                                "branches": name.replace("MB-3_", "MB-3 "),
                                "acc_t": pct(means[2]), "f1_t": pct(means[3]),
                                "acc_w": pct(means[0]), "f1_w": pct(means[1]),
                                "features": str(total_F)
                            })
                        if name.startswith("MB-3"):
                            window_rows_MB3.append({
                                "win": f"{float(pre['win_s']):.1f}",
                                "acc_t": pct(means[2]), "f1_t": pct(means[3]),
                                "acc_w": pct(means[0]), "f1_w": pct(means[1]),
                                "mis_trials": str(mis_trials_sum),
                                "mis_subjects": f"{len(mis_subjects_set)}/{len(keep_subjects)}"
                            })
                        if pre_name.startswith("W2_") and name.startswith("MB-3"):
                            for cond in test_sets:
                                feats_c_list = [test_feats_by_cond_branch[cond][b] for b in BRANCH_EXPERIMENTS[-1]["branches"]]
                                Xc_feat = np.concatenate(feats_c_list, axis=1)
                                metas = test_metas_by_cond[cond]
                                y_true_win = test_truth_by_cond[cond]
                                y_pred_win = clf.predict(Xc_feat).tolist()
                                acc_w_c = accuracy_score(y_true_win, y_pred_win) if len(y_true_win) else float("nan")
                                try:
                                    f1_w_c = f1_score(y_true_win, y_pred_win, average="macro")
                                except Exception:
                                    f1_w_c = float("nan")
                                y_true_t, y_pred_t, t_subjs, t_ids = trial_level_from_windows(y_true_win, y_pred_win, metas)
                                acc_t_c = accuracy_score(y_true_t, y_pred_t) if len(y_true_t) else float("nan")
                                try:
                                    f1_t_c = f1_score(y_true_t, y_pred_t, average="macro")
                                except Exception:
                                    f1_t_c = float("nan")
                                mis_trials = sum(1 for yt, yp in zip(y_true_t, y_pred_t) if yp != yt)
                                mis_subjects = {s for yt, yp, s in zip(y_true_t, y_pred_t, t_subjs) if yp != yt}
                                main_rows_cond.append({
                                    "cond": pretty_cond(cond),
                                    "acc_t": pct(acc_t_c), "f1_t": pct(f1_t_c),
                                    "acc_w": pct(acc_w_c), "f1_w": pct(f1_w_c),
                                    "mis_trials": str(mis_trials),
                                    "mis_subjects": f"{len(mis_subjects)}/{len(set(t_subjs))}"
                                })
                    exp_key = f"{pre_name}|{name}"
                    exp_trial_ok[exp_key] = pertrial_ok
                    L.add_exp_meta({
                        "preproc": pre_name, "exp": name, "branches": branches,
                        "kernels": {b: DEFAULT_KERNELS[b] for b in branches},
                        "total_features": total_F, "win_s": float(pre["win_s"])
                    })
                perexp_trial_ok.update(exp_trial_ok)
                if baseline_key is None:
                    baseline_key = f"{PREPROC_EXPERIMENTS[0]['name']}|{BRANCH_EXPERIMENTS[0]['name']}"
        if BOOTSTRAP_REPS > 0 and baseline_key in perexp_trial_ok:
            L.log(f"\n[Bootstrap] Trial-accuracy deltas vs baseline '{baseline_key}' (reps={BOOTSTRAP_REPS})")
            base_ok = perexp_trial_ok[baseline_key]
            deltas = {}
            for exp_key, alt_ok in perexp_trial_ok.items():
                if exp_key == baseline_key:
                    continue
                mean_delta, (ci_lo, ci_hi) = paired_bootstrap_acc_delta(base_ok, alt_ok, reps=BOOTSTRAP_REPS)
                L.log(f"  {exp_key:48s} Δacc_t={mean_delta:+.4f}  CI95%=({ci_lo:+.4f},{ci_hi:+.4f})")
                deltas[exp_key] = {"delta_mean": mean_delta, "ci95": [ci_lo, ci_hi]}
            L.meta["bootstrap_deltas_vs_baseline"] = {"baseline": baseline_key, "reps": BOOTSTRAP_REPS, "alpha": BOOTSTRAP_ALPHA, "deltas": deltas}
    main_rows_cond_sorted = sorted(main_rows_cond, key=lambda r: r["cond"])
    branch_rows_W2_sorted = branch_rows_W2
    w_order = {"1.0": 0, "2.0": 1, "3.0": 2}
    window_rows_MB3_sorted = sorted(window_rows_MB3, key=lambda r: w_order.get(r["win"], 99))

    def row_main(r):
        return (f"{latex_escape(r['cond'])} & {r['acc_t']} & {r['f1_t']} & "
                f"{r['acc_w']} & {r['f1_w']} & {r['mis_trials']} & {r['mis_subjects']} \\\\")

    def row_branch(r):
        return (f"{latex_escape(r['branches'])} & {r['acc_t']} & {r['f1_t']} & "
                f"{r['acc_w']} & {r['f1_w']} & {r['features']} \\\\")

    def row_window(r):
        return (f"{r['win']} & {r['acc_t']} & {r['f1_t']} & "
                f"{r['acc_w']} & {r['f1_w']} & {r['mis_trials']} & {r['mis_subjects']} \\\\")

    body_main = "\n".join([row_main(r) for r in main_rows_cond_sorted])
    body_branch = "\n".join([row_branch(r) for r in branch_rows_W2_sorted])
    body_window = "\n".join([row_window(r) for r in window_rows_MB3_sorted])

    table_main = r"""
% ===== Table 1: MB-3 (W2 = 2.0 s) condition-wise results =====
\begin{table}[t]
  \caption{MB--3 (raw+diff+diff2) with W2 preprocessing (acc+gyro, per-window $z$-score, gate=0.4, 2.0\,s). Values in percent. MisID trials: trials where majority-vote prediction mismatched the true subject; ``Subjects w/ misID'': unique subjects with at least one misidentified trial in that condition.}
  \label{tab:mb3_w2_condition}
  \centering
  \footnotesize
  \renewcommand{\arraystretch}{1.12}
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{L{3.6cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm} C{2.2cm}}
    \toprule
    \textbf{Condition} & \textbf{Acc. (trial)} & \textbf{F1 (trial)} & \textbf{Acc. (window)} & \textbf{F1 (window)} & \textbf{MisID trials} & \textbf{Subjects w/ misID} \\
    \midrule
""" + body_main + r"""
    \bottomrule
  \end{tabular}
\end{table}
""".strip()

    table_branch = r"""
% ===== Table 2: Branch ablation at W2 (2.0 s) =====
\begin{table}[t]
  \caption{Branch ablation at W2 (2.0\,s). Reported are overall means across all non-baseline conditions.}
  \label{tab:branch_ablation_w2}
  \centering
  \footnotesize
  \renewcommand{\arraystretch}{1.12}
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{L{3.2cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.8cm}}
    \toprule
    \textbf{Branches} & \textbf{Acc. (trial)} & \textbf{F1 (trial)} & \textbf{Acc. (window)} & \textbf{F1 (window)} & \textbf{Features} \\
    \midrule
""" + body_branch + r"""
    \bottomrule
  \end{tabular}
\end{table}
""".strip()

    table_window = r"""
% ===== Table 3: Window-size ablation for MB-3 (W1/W2/W3) =====
\begin{table}[t]
  \caption{Window-size ablation for MB--3 (acc+gyro, per-window $z$-score, gate=0.4). Overall means across conditions; percent values.}
  \label{tab:window_ablation_mb3}
  \centering
  \footnotesize
  \renewcommand{\arraystretch}{1.12}
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{C{1.2cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm} C{2.2cm}}
    \toprule
    \textbf{Win (s)} & \textbf{Acc. (trial)} & \textbf{F1 (trial)} & \textbf{Acc. (window)} & \textbf{F1 (window)} & \textbf{MisID trials} & \textbf{Subjects w/ misID} \\
    \midrule
""" + body_window + r"""
    \bottomrule
  \end{tabular}
\end{table}
""".strip()

    with open(out_tex_all, "w", encoding="utf-8") as ftex:
        ftex.write(table_main + "\n\n" + table_branch + "\n\n" + table_window + "\n")
    print(f"\n[Saved] LaTeX tables -> {out_tex_all}")
    L.write_all()
    print(f"\n[Done] Saved to: {OUT_DIR}")
    print(" - results_ablation_summary.csv")
    print(" - results_ablation_trial_preds.csv")
    print(" - results_ablation_subject_by_condition.csv")
    print(" - ablation_tables.tex")
    print(" - ablation_log.txt / ablation_meta.json")


if __name__ == "__main__":
    main()
