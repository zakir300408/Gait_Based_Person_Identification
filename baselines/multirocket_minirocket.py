import os, math, random, json, hashlib, platform, sys
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any

import h5py, numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

H5_PATH = r"gait_id_pelvis_62.h5"
OUT_DIR = r"baselines_out/rocket_consistent_mb3"
SEED = 42

RUN_KINDS = ["minirocket", "multirocket"]

MINIROCKET_NUM_KERNELS = 10000
MULTIROCKET_NUM_KERNELS = 10000
N_JOBS_MINIROCKET = -1
N_JOBS_MULTI = -1

EVAL_CONDITIONS = [
    "fpa_large","fpa_small",
    "step_width_large","step_width_small",
    "trunk_sway_large","trunk_sway_small",
    "walk_speed_large","walk_speed_small",
    "dual_task"
]

FS_HZ = 100
WIN_S = 2.0
WIN = int(WIN_S * FS_HZ)
HOP = WIN // 2
ACC_STD_MIN = 0.4

class RunLogger:
    def __init__(self, out_dir: str):
        self.lines: List[str] = []
        self.out_txt = os.path.join(out_dir, "rocket_log.txt")
        self.out_json = os.path.join(out_dir, "rocket_meta.json")
        self.meta: Dict[str, Any] = {"notes": [], "checks": {}, "train": {}, "test": {}, "env": {}, "model": {}}
    def log(self, msg: str):
        self.lines.append(msg)
        print(msg, flush=True)
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
            if isinstance(obj, dict): return {str(k): _clean(v) for k,v in obj.items()}
            return obj
        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(_clean(self.meta), f, indent=2)

def sha1_of_strings(strings: List[str]) -> str:
    h = hashlib.sha1()
    for s in strings:
        h.update((s+"\n").encode("utf-8"))
    return h.hexdigest()

def summarize_counts(metas: List[Dict[str,Any]], by: str) -> Dict[str,int]:
    cnt = Counter([m[by] for m in metas])
    return dict(sorted(cnt.items(), key=lambda kv: kv[0]))

def _import_rocket():
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate, MultiRocketMultivariate
        return MiniRocketMultivariate, MultiRocketMultivariate
    except Exception as e:
        raise ImportError(
            "Requires sktime>=0.38. Try: pip install -U 'sktime>=0.38,<0.40' numba numpy scikit-learn"
        ) from e

def set_seed(seed:int=SEED):
    random.seed(seed); np.random.seed(seed)

def ensure_outdir(p): os.makedirs(p, exist_ok=True)

def list_index(h5: h5py.File):
    idx = h5["index"]
    dec = lambda arr: [x.decode("utf-8") for x in arr[()]]
    return dict(
        subject   = dec(idx["subject"]),
        condition = dec(idx["condition"]),
        trial_id  = dec(idx["trial_id"]),
        h5_group  = dec(idx["h5_group"]),
        n_samples = idx["n_samples"][()].tolist(),
        n_channels= idx["n_channels"][()].tolist(),
        had_time  = idx["had_time"][()].tolist(),
    )

def get_cols(h5: h5py.File, gp: str) -> List[str]:
    return json.loads(h5[gp].attrs["columns_json"])

PELVIS_ACCEL = ["AccelX_PELVIS (m/s^2)","AccelY_PELVIS (m/s^2)","AccelZ_PELVIS (m/s^2)"]
PELVIS_GYRO  = ["GyroX_PELVIS (deg/s)","GyroY_PELVIS (deg/s)","GyroZ_PELVIS (deg/s)"]
PELVIS_ALLOWED = PELVIS_ACCEL + PELVIS_GYRO

def build_pelvis_canonical(h5: h5py.File, baseline_groups: List[str]) -> List[str]:
    have = Counter()
    for gp in baseline_groups:
        cols = set(get_cols(h5, gp))
        for name in PELVIS_ALLOWED:
            if name in cols:
                have[name] += 1
    if baseline_groups:
        thresh = math.ceil(0.90*len(baseline_groups))
        keep = [n for n,c in have.items() if c>=thresh]
        if not keep:
            col_sets = [set(get_cols(h5,gp)) for gp in baseline_groups]
            inter = set.intersection(*col_sets) if col_sets else set()
            keep = [n for n in PELVIS_ALLOWED if n in inter]
        if keep:
            return [n for n in PELVIS_ALLOWED if n in keep]
    return PELVIS_ALLOWED.copy()

def load_pelvis_aligned_TxC(h5: h5py.File, gp: str, canon_cols: List[str]) -> np.ndarray:
    mat = h5[gp]["imu"][()]
    cols = get_cols(h5, gp)
    name2idx = {n:i for i,n in enumerate(cols)}
    T = mat.shape[0]; C = len(canon_cols)
    out = np.zeros((T, C), dtype=np.float32)
    for j, name in enumerate(canon_cols):
        i = name2idx.get(name, None)
        if i is not None:
            out[:, j] = mat[:, i].astype(np.float32, copy=False)
    return out

def _svm(acc_xyz: np.ndarray) -> np.ndarray:
    return np.sqrt((acc_xyz**2).sum(axis=1))

def win_id(subj: str, cond: str, trial: str, h5_group: str, s0: int, s1: int) -> str:
    return f"{subj}|{cond}|{trial}|{h5_group}|{s0}|{s1}"

def window_and_gate_with_meta(mat_TxC: np.ndarray, win:int, hop:int,
                              subj: str, cond: str, trial: str, h5_group: str):
    T = mat_TxC.shape[0]; out_wins=[]; out_meta=[]
    s=0
    while s+win<=T:
        seg = mat_TxC[s:s+win,:]
        svm = _svm(seg[:,:3])
        if np.std(svm) >= ACC_STD_MIN:
            out_wins.append(seg.T.astype(np.float32, copy=False))
            out_meta.append({"subject":subj, "condition":cond, "trial":trial,
                             "group":h5_group, "s0":int(s), "s1":int(s+win)})
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

def per_trial_metrics(y_true_win, y_pred_win, ids_win):
    if not ids_win:
        return float("nan"), float("nan")
    by_trial = defaultdict(list); lab_trial = {}
    for yt, yp, tid in zip(y_true_win, y_pred_win, ids_win):
        by_trial[tid].append(yp); lab_trial.setdefault(tid, yt)
    y_true_t, y_pred_t = [], []
    for tid, preds in by_trial.items():
        maj = Counter(preds).most_common(1)[0][0]
        y_pred_t.append(maj); y_true_t.append(lab_trial[tid])
    acc_t = accuracy_score(y_true_t, y_pred_t)
    try:
        f1_t = f1_score(y_true_t, y_pred_t, average="macro")
    except Exception:
        f1_t = float("nan")
    return acc_t, f1_t

def make_linear_head():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("ridge", RidgeClassifierCV(alphas=(0.1, 1.0, 10.0)))
    ])

def main():
    set_seed(SEED)
    ensure_outdir(OUT_DIR)
    L = RunLogger(OUT_DIR)
    L.add_meta("task", "person identification from pelvis-mounted IMU during walking subject classification")
    L.add_meta("preprocessing", {
        "channels": "acc+gyro",
        "gate": "energy only std SVM >= 0.4 m/s^2",
        "window_s": WIN_S, "hop_s": HOP/FS_HZ,
        "normalization": "per window z score"
    })
    L.add_meta("seed", SEED)
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    try:
        import sklearn
        env["scikit_learn"] = sklearn.__version__
    except Exception:
        env["scikit_learn"] = "unknown"
    L.add_meta("env", env)

    MiniRocketMultivariate, MultiRocketMultivariate = _import_rocket()

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
                       if s in has_base and c.lower()=="baseline"]
        canon_cols = build_pelvis_canonical(h5, base_groups)
        with open(os.path.join(OUT_DIR, "pelvis_canonical_cols.json"), "w", encoding="utf-8") as f:
            json.dump(canon_cols, f, indent=2)
        L.log(f"[Info] Pelvis columns: {canon_cols} (C={len(canon_cols)})")
        L.add_meta("pelvis_columns", canon_cols)
        label2id = {s:i for i,s in enumerate(keep_subjects)}
        id2label = {i:s for s,i in label2id.items()}
        L.add_meta("label2id", label2id)

        X_train_win, y_train, train_meta = [], [], []
        for i,(s,c) in enumerate(zip(idx["subject"], idx["condition"])):
            if s in has_base and c.lower()=="baseline":
                gp = idx["h5_group"][i]
                trid = idx["trial_id"][i]
                mat = load_pelvis_aligned_TxC(h5, gp, canon_cols)
                T = mat.shape[0]; s0 = 0
                while s0 + WIN <= T:
                    seg = mat[s0:s0+WIN, :]
                    svm = _svm(seg[:,:3])
                    if np.std(svm) >= ACC_STD_MIN:
                        win_CxT = seg.T.astype(np.float32, copy=False)
                        win_CxT = zscore_per_window(win_CxT)
                        X_train_win.append(win_CxT)
                        y_train.append(label2id[s])
                        train_meta.append({"subject":s, "condition":"baseline", "trial":idx["trial_id"][i],
                                           "group":gp, "s0":int(s0), "s1":int(s0+WIN)})
                    s0 += HOP

        conds_present = sorted(set([c for c in idx["condition"] if c.lower()!="baseline"]))
        eval_conds = [c for c in EVAL_CONDITIONS if c in conds_present] if EVAL_CONDITIONS else conds_present
        test_sets = {}
        test_meta_by_cond: Dict[str, List[Dict[str,Any]]] = {}
        for cond in eval_conds:
            Xc, yc, metas_c = [], [], []
            for i,(s,c) in enumerate(zip(idx["subject"], idx["condition"])):
                if s in has_base and c == cond:
                    gp = idx["h5_group"][i]
                    trid = idx["trial_id"][i]
                    mat = load_pelvis_aligned_TxC(h5, gp, canon_cols)
                    T = mat.shape[0]; s0 = 0
                    while s0 + WIN <= T:
                        seg = mat[s0:s0+WIN, :]
                        svm = _svm(seg[:,:3])
                        if np.std(svm) >= ACC_STD_MIN:
                            win_CxT = seg.T.astype(np.float32, copy=False)
                            win_CxT = zscore_per_window(win_CxT)
                            Xc.append(win_CxT); yc.append(label2id[s])
                            metas_c.append({"subject":s, "condition":cond, "trial":trid,
                                            "group":gp, "s0":int(s0), "s1":int(s0+WIN)})
                        s0 += HOP
            if Xc:
                test_sets[cond] = (Xc, yc)
                test_meta_by_cond[cond] = metas_c

    train_ids = [win_id(m["subject"], m["condition"], m["trial"], m["group"], m["s0"], m["s1"]) for m in train_meta]
    L.add_meta("train", {
        "n_windows": len(train_meta),
        "per_subject_counts": summarize_counts(train_meta, "subject"),
        "per_trial_counts": summarize_counts(train_meta, "trial"),
        "checksum_sha1": sha1_of_strings(sorted(train_ids))
    })
    for cond, metas in test_meta_by_cond.items():
        test_ids = [win_id(m["subject"], m["condition"], m["trial"], m["group"], m["s0"], m["s1"]) for m in metas]
        inter = set(test_ids).intersection(train_ids)
        L.meta["checks"].setdefault("leakage", {})[cond] = {
            "test_n_windows": len(test_ids),
            "intersection_with_train": len(inter),
            "test_checksum_sha1": sha1_of_strings(sorted(test_ids))
        }
        if len(inter) != 0:
            raise RuntimeError(f"[LEAKAGE] Found {len(inter)} overlapping window IDs between TRAIN and TEST:{cond}")
    L.log("[Check] No train/test window-ID intersections detected.")

    L.log("[Info] Preparing NumPy panels (N,C,T) for ROCKET...")
    X_train_nct = stack_windows_NCT(X_train_win)
    y_train_arr = np.asarray(y_train, dtype=np.int32)
    n_classes = len(set(y_train_arr.tolist()))
    L.log(f"[Info] Train windows={X_train_nct.shape[0]} | (C,T)=({X_train_nct.shape[1]},{X_train_nct.shape[2]}) | classes={n_classes}")
    assert X_train_nct.dtype == np.float64 and X_train_nct.flags['C_CONTIGUOUS']
    assert y_train_arr.dtype == np.int32

    clf_maker = make_linear_head

    for kind in RUN_KINDS:
        L.log(f"\n========== Training ROCKET baseline: {kind} (CONSISTENT) ==========")
        if kind == "minirocket":
            RocketClass = _import_rocket()[0]
            rocket = RocketClass(num_kernels=MINIROCKET_NUM_KERNELS, random_state=SEED, n_jobs=N_JOBS_MINIROCKET)
            out_csv = os.path.join(OUT_DIR, "results_rocket_minirocket_CONSISTENT.csv")
            nk = MINIROCKET_NUM_KERNELS
        elif kind == "multirocket":
            RocketClass = _import_rocket()[1]
            rocket = RocketClass(num_kernels=MULTIROCKET_NUM_KERNELS, random_state=SEED, n_jobs=N_JOBS_MULTI)
            out_csv = os.path.join(OUT_DIR, "results_rocket_multirocket_CONSISTENT.csv")
            nk = MULTIROCKET_NUM_KERNELS
        else:
            raise ValueError(f"Unknown kind: {kind}")

        X_train_f = _as_f64_c(X_train_nct)
        rocket.fit(X_train_f, y_train_arr)
        X_train_feat = rocket.transform(X_train_f)

        clf = clf_maker()
        clf.fit(X_train_feat, y_train_arr)

        results = []
        for cond, (Xc_win, yc) in test_sets.items():
            metas = test_meta_by_cond[cond]
            ids_trials = [f"{m['subject']}|{m['condition']}|{m['trial']}" for m in metas]
            if not Xc_win:
                continue
            Xc_nct = stack_windows_NCT(Xc_win)
            Xc_feat = rocket.transform(_as_f64_c(Xc_nct))
            y_pred_win = clf.predict(Xc_feat).tolist()
            yc_arr = np.asarray(yc, dtype=np.int32)
            acc_w = accuracy_score(yc_arr, y_pred_win) if len(yc_arr) else float("nan")
            try:
                f1_w = f1_score(yc_arr, y_pred_win, average="macro")
            except Exception:
                f1_w = float("nan")
            acc_t, f1_t = per_trial_metrics(yc_arr, y_pred_win, ids_trials)
            results.append((cond, acc_w, f1_w, acc_t, f1_t))
            L.log(f"  {kind:11s} | {cond:20s} acc_w={acc_w:.3f} f1_w={f1_w:.3f} | acc_t={acc_t:.3f} f1_t={f1_t:.3f}")

        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rocket_kind","condition","acc_window","f1_window","acc_trial","f1_trial",
                        "win_s","num_kernels","seed","anchored_yaw","subject_norm"])
            for cond, accw, f1w, acct, f1t in results:
                w.writerow([kind, cond, f"{accw:.4f}", f"{f1w:.4f}", f"{acct:.4f}", f"{f1t:.4f}",
                            f"{WIN_S:.1f}", nk, SEED, 0, 0])
            if results:
                arr = np.array([[a,b,c,d] for _,a,b,c,d in results], dtype=float)
                means = np.nanmean(arr, axis=0)
                w.writerow([kind, "OVERALL_MEAN", f"{means[0]:.4f}", f"{means[1]:.4f}",
                            f"{means[2]:.4f}", f"{means[3]:.4f}", f"{WIN_S:.1f}", nk, SEED, 0, 0])
                L.log(f"  -> OVERALL MEAN ({kind}): acc_w={means[0]:.3f} f1_w={means[1]:.3f} | acc_t={means[2]:.3f} f1_t={means[3]:.3f}")
        L.log(f"[Saved] {out_csv}")

    L.write_all()
    print(f"\n[Saved]")
    print(f" - LOG:   {L.out_txt}")
    print(f" - META:  {L.out_json}")
    print(f" - CSVs:  {OUT_DIR}/*.csv")

if __name__ == "__main__":
    main()
