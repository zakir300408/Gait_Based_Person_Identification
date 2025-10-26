import os, json, time, random, math, hashlib, platform, sys
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any

import h5py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

H5_PATH = r"gait_id_pelvis_62.h5"
OUT_DIR = r"baselines_out/resnet1d_consistent_mb3"
SEED = 42

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

BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTH = 0.05

class RunLogger:
    def __init__(self, out_dir: str):
        self.lines: List[str] = []
        self.out_txt = os.path.join(out_dir, "resnet_log.txt")
        self.out_json = os.path.join(out_dir, "resnet_meta.json")
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

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

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
    from collections import Counter
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

def zscore_per_window(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True); sd = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mu) / sd

class WindowDataset(Dataset):
    def __init__(self, X, y, ids=None):
        self.X, self.y, self.ids = X, y, ids
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = zscore_per_window(self.X[i])
        xb = torch.from_numpy(x).float()
        yb = torch.tensor(int(self.y[i]), dtype=torch.long)
        if self.ids is None:
            return xb, yb
        else:
            return xb, yb, self.ids[i]

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k1=7, k2=5, k3=3):
        super().__init__()
        pad1, pad2, pad3 = k1//2, k2//2, k3//2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k1, padding=pad1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k2, padding=pad2, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.conv3 = nn.Conv1d(out_ch, out_ch, kernel_size=k3, padding=pad3, bias=False)
        self.bn3   = nn.BatchNorm1d(out_ch)
        self.short = nn.Sequential()
        if in_ch != out_ch:
            self.short = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(out_ch))
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        y = F.relu(y + self.short(x))
        return y

class ResNet1D(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, widths=(64,128,128)):
        super().__init__()
        self.block1 = BasicBlock1D(in_ch, widths[0])
        self.block2 = BasicBlock1D(widths[0], widths[1])
        self.block3 = BasicBlock1D(widths[1], widths[2])
        self.bn     = nn.BatchNorm1d(widths[2])
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Linear(widths[2], n_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.bn(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)

@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    y_true, y_pred, ids_out = [], [], []
    for batch in loader:
        if len(batch) == 3:
            xb, yb, ids = batch
        else:
            xb, yb = batch
            ids = None
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        logits = model(xb)
        pred = logits.argmax(1)
        y_true.extend(yb.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        if ids is not None:
            ids_out.extend(list(ids))
    acc = accuracy_score(y_true, y_pred) if y_true else float("nan")
    try:
        f1 = f1_score(y_true, y_pred, average="macro")
    except Exception:
        f1 = float("nan")
    return acc, f1, y_pred, ids_out

def per_trial_metrics(y_true_win, y_pred_win, ids_win):
    if not ids_win: return float("nan"), float("nan")
    by_trial = defaultdict(list); lab_trial = {}
    for yt, yp, tid in zip(y_true_win, y_pred_win, ids_win):
        by_trial[tid].append(yp); lab_trial.setdefault(tid, yt)
    y_true_t, y_pred_t = [], []
    for tid, preds in by_trial.items():
        maj = Counter(preds).most_common(1)[0][0]
        y_pred_t.append(maj); y_true_t.append(lab_trial[tid])
    acc_t = accuracy_score(y_true_t, y_pred_t)
    try: f1_t = f1_score(y_true_t, y_pred_t, average="macro")
    except Exception: f1_t = float("nan")
    return acc_t, f1_t

def smooth_ce(logits, target, eps=LABEL_SMOOTH):
    nclass = logits.size(1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits).fill_(eps / (nclass - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1 - eps)
    return torch.sum(-true_dist * F.log_softmax(logits, dim=1), dim=1).mean()

def train_resnet(model, train_loader, epochs=EPOCHS):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(1, epochs+1):
        model.train(); running=0.0; nseen=0
        for batch in train_loader:
            xb, yb = batch[:2]
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = smooth_ce(model(xb), yb)
            loss.backward()
            if GRAD_CLIP_NORM:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            running += float(loss.item()) * yb.size(0); nseen += yb.size(0)
        sched.step()
        print(f"[Epoch {ep:02d}] train_loss={running/max(nseen,1):.4f} | lr={opt.param_groups[0]['lr']:.2e}", flush=True)
    return model

def sha1_of_strings(strings: List[str]) -> str:
    h = hashlib.sha1()
    for s in strings:
        h.update((s+"\n").encode("utf-8"))
    return h.hexdigest()

def summarize_counts(metas: List[Dict[str,Any]], by: str) -> Dict[str,int]:
    cnt = Counter([m[by] for m in metas])
    return dict(sorted(cnt.items(), key=lambda kv: kv[0]))

def main():
    t0 = time.time()
    set_seed(SEED); ensure_outdir(OUT_DIR)
    L = RunLogger(OUT_DIR)
    L.add_meta("task", "person identification from pelvis-mounted IMU during walking subject classification")
    L.add_meta("preprocessing", {
        "channels": "acc+gyro",
        "gate": "energy only std SVM >= 0.4 m/s^2",
        "window_s": WIN_S, "hop_s": HOP/FS_HZ,
        "normalization": "per window z score"
    })
    L.add_meta("seed", SEED)
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "torch": torch.__version__}
    try:
        import sklearn
        env["scikit_learn"] = sklearn.__version__
    except Exception:
        env["scikit_learn"] = "unknown"
    L.add_meta("env", env)
    with h5py.File(H5_PATH, "r") as h5:
        idx = list_index(h5)
        subjects_all = sorted(set(idx["subject"]))
        has_base = {s for s, c in zip(idx["subject"], idx["condition"]) if c.lower()=="baseline"}
        keep_subjects = sorted(has_base)
        L.log(f"[Info] Subjects with baseline: {len(keep_subjects)} / total {len(subjects_all)}")
        L.add_meta("subjects_total", len(subjects_all))
        L.add_meta("subjects_with_baseline", len(keep_subjects))
        L.add_meta("subject_list", keep_subjects)
        base_groups = [gp for s,c,gp in zip(idx["subject"], idx["condition"], idx["h5_group"])
                       if s in has_base and c.lower()=="baseline"]
        canon_cols = build_pelvis_canonical(h5, base_groups)
        with open(os.path.join(OUT_DIR, "pelvis_canonical_cols.json"), "w", encoding="utf-8") as f:
            json.dump(canon_cols, f, indent=2)
        L.log(f"[Info] Pelvis columns: {canon_cols} (C={len(canon_cols)})")
        L.add_meta("pelvis_columns", canon_cols)
        label2id = {s:i for i,s in enumerate(keep_subjects)}
        id2label = {i:s for s,i in label2id.items()}
        L.add_meta("label2id", label2id)
        X_train, y_train, train_meta = [], [], []
        for i,(s,c) in enumerate(zip(idx["subject"], idx["condition"])):
            if s in has_base and c.lower()=="baseline":
                gp  = idx["h5_group"][i]
                trid= idx["trial_id"][i]
                mat = load_pelvis_aligned_TxC(h5, gp, canon_cols)
                wins, metas = window_and_gate_with_meta(mat, WIN, HOP, subj=s, cond="baseline", trial=trid, h5_group=gp)
                for w, m in zip(wins, metas):
                    X_train.append(w); y_train.append(label2id[s]); train_meta.append(m)
        conds_present = sorted(set([c for c in idx["condition"] if c.lower()!="baseline"]))
        eval_conds = [c for c in EVAL_CONDITIONS if c in conds_present] if EVAL_CONDITIONS else conds_present
        test_sets = {}
        test_meta_by_cond: Dict[str, List[Dict[str,Any]]] = {}
        for cond in eval_conds:
            Xc, yc, metas_c = [], [], []
            for i,(s,c) in enumerate(zip(idx["subject"], idx["condition"])):
                if s in has_base and c == cond:
                    gp   = idx["h5_group"][i]; trid = idx["trial_id"][i]
                    mat = load_pelvis_aligned_TxC(h5, gp, canon_cols)
                    wins, metas = window_and_gate_with_meta(mat, WIN, HOP, subj=s, cond=cond, trial=trid, h5_group=gp)
                    for w, m in zip(wins, metas):
                        Xc.append(w); yc.append(label2id[s]); metas_c.append(m)
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
    train_loader = DataLoader(WindowDataset(X_train, y_train, None),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    in_ch = X_train[0].shape[0] if X_train else len(canon_cols)
    n_classes = len(set(y_train))
    L.log(f"[Info] ResNet1D: in_ch={in_ch}, classes={n_classes}, train_windows={len(X_train)}")
    model = ResNet1D(in_ch=in_ch, n_classes=n_classes, widths=(64,128,128))
    model = train_resnet(model, train_loader, epochs=EPOCHS)
    model_path = os.path.join(OUT_DIR, "resnet1d_model.pt")
    torch.save({"state_dict": model.state_dict(),
                "in_ch": in_ch, "n_classes": n_classes,
                "arch": {"widths": (64,128,128)}}, model_path)
    L.log(f"[Saved] Model weights -> {model_path}")
    results = []
    for cond, (Xc, yc) in test_sets.items():
        metas = test_meta_by_cond[cond]
        ids = [f"{m['subject']}|{m['condition']}|{m['trial']}" for m in metas]
        loader = DataLoader(WindowDataset(Xc, yc, ids), batch_size=128, shuffle=False, num_workers=0, drop_last=False)
        acc_w, f1_w, y_pred_w, ids_w = evaluate_loader(model, loader)
        acc_t, f1_t = per_trial_metrics(yc, y_pred_w, ids_w)
        results.append((cond, acc_w, f1_w, acc_t, f1_t))
        L.log(f"  RESNET1D | {cond:20s} acc_w={acc_w:.3f} f1_w={f1_w:.3f} | acc_t={acc_t:.3f} f1_t={f1_t:.3f}")
    ensure_outdir(OUT_DIR)
    out_csv = os.path.join(OUT_DIR, "results_resnet1d_CONSISTENT.csv")
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["baseline","condition","acc_window","f1_window","acc_trial","f1_trial",
                    "win_s","seed","epochs","batch","lr","weight_decay","anchored_yaw","subject_norm"])
        for cond, accw, f1w, acct, f1t in results:
            w.writerow(["resnet1d_consistent_mb3", cond, f"{accw:.4f}", f"{f1w:.4f}",
                        f"{acct:.4f}", f"{f1t:.4f}", f"{WIN_S:.1f}", SEED, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, 0, 0])
        if results:
            arr = np.array([[a,b,c,d] for _,a,b,c,d in results], dtype=float)
            means = np.nanmean(arr, axis=0)
            w.writerow(["resnet1d_consistent_mb3", "OVERALL_MEAN",
                        f"{means[0]:.4f}", f"{means[1]:.4f}", f"{means[2]:.4f}", f"{means[3]:.4f}",
                        f"{WIN_S:.1f}", SEED, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, 0, 0])
            L.log(f"  -> OVERALL MEAN: acc_w={means[0]:.3f} f1_w={means[1]:.3f} | acc_t={means[2]:.3f} f1_t={means[3]:.3f}")
    L.add_meta("model", {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
                         "in_ch": in_ch, "n_classes": n_classes})
    L.write_all()
    print(f"\n[Saved]")
    print(f" - CSV:   {out_csv}")
    print(f" - LOG:   {L.out_txt}")
    print(f" - META:  {L.out_json}")
    print(f" - MODEL: {model_path}")

if __name__ == "__main__":
    main()
