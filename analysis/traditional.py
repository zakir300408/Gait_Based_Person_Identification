"""The three fixed-setting 41-feature conventional comparison pipelines."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import platform
from typing import Tuple
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from analysis.data import PELVIS_COLS, sha256
import sklearn
FS_HZ=100
SEED=42
GAIT_BAND=(0.5,3.0)
def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))

def _energy(x: np.ndarray) -> float:
    return float(np.sum(x**2) / max(1, len(x)))

def _corr(x: np.ndarray, y: np.ndarray) -> float:
    xv = x - x.mean(); yv = y - y.mean()
    num = float(np.sum(xv*yv))
    den = float(np.sqrt(np.sum(xv*xv)*np.sum(yv*yv))+1e-12)
    return num/den

def _dominant_peak_ratio(x: np.ndarray, fs: int, fband: Tuple[float,float]) -> float:
    n = len(x)
    if n < 32: return 0.0
    X = np.fft.rfft(x * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    lo, hi = fband
    mask = (freqs>=lo) & (freqs<=hi)
    if not np.any(mask): return 0.0
    pwr = (np.abs(X[mask])**2)
    if pwr.sum() <= 1e-12: return 0.0
    return float(pwr.max()/pwr.sum())

def _dominant_freq(x: np.ndarray, fs: int, fband: Tuple[float,float]) -> float:
    n = len(x)
    if n < 32: return 0.0
    X = np.fft.rfft(x * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    lo, hi = fband
    mask = (freqs>=lo) & (freqs<=hi)
    if not np.any(mask): return 0.0
    pwr = (np.abs(X[mask])**2)
    if pwr.max() <= 1e-12: return 0.0
    return float(freqs[mask][np.argmax(pwr)])

def extract_features_window(x_CxT: np.ndarray, fs:int=FS_HZ) -> np.ndarray:
    Ax, Ay, Az, Gx, Gy, Gz = x_CxT
    feats = []
    for ch in (Ax, Ay, Az, Gx, Gy, Gz):
        feats.extend([float(np.mean(ch)), float(np.std(ch)), _rms(ch), _mad(ch), _energy(ch)])
    acc_horiz = _energy(Ax) + _energy(Ay)
    acc_vert  = _energy(Az) + 1e-12
    gyro_horiz = _energy(Gx) + _energy(Gy)
    gyro_vert  = _energy(Gz) + 1e-12
    feats.extend([acc_horiz/acc_vert, gyro_horiz/gyro_vert])
    feats.extend([
        _corr(Ax, Ay), _corr(Ax, Az), _corr(Ay, Az),
        _corr(Gx, Gy), _corr(Gx, Gz), _corr(Gy, Gz)
    ])
    svm = np.sqrt(Ax**2 + Ay**2 + Az**2)
    feats.extend([
        float(np.std(svm)),
        _dominant_freq(svm, fs, GAIT_BAND),
        _dominant_peak_ratio(svm, fs, GAIT_BAND)
    ])
    return np.asarray(feats, dtype=np.float32)

def per_trial_metrics(y_true_win, y_pred_win, ids_win):
    if not ids_win:
        return float("nan"), float("nan")
    by_trial = defaultdict(list)
    lab_trial = {}
    for yt, yp, tid in zip(y_true_win, y_pred_win, ids_win):
        by_trial[tid].append(yp)
        lab_trial.setdefault(tid, yt)
    y_true_t, y_pred_t = [], []
    for tid, preds in by_trial.items():
        maj = Counter(preds).most_common(1)[0][0]
        y_pred_t.append(maj)
        y_true_t.append(lab_trial[tid])
    acc_t = accuracy_score(y_true_t, y_pred_t)
    f1_t = f1_score(y_true_t, y_pred_t, average="macro")
    return acc_t, f1_t

def make_classifier(kind: str, n_classes: int) -> Pipeline:
    if kind == "lr":
        clf = LogisticRegression(
            penalty="l2", solver="lbfgs", multi_class="auto",
            max_iter=2000, random_state=SEED
        )
    elif kind == "svm_rbf":
        clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=SEED)
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            n_jobs=-1, random_state=SEED
        )
    else:
        raise ValueError(f"Unknown classifier kind: {kind}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
def main():
    root=Path(__file__).resolve().parents[1]
    metadata=pd.read_csv(root/'outputs/data/window_200/windows.csv')
    metadata=metadata.loc[metadata.partition.isin(['enrollment','test'])].copy()
    features=np.empty((len(metadata),41),dtype=np.float32)
    metadata=metadata.reset_index(drop=True)
    with h5py.File(root/'data/altergait/gait_id_pelvis_62.h5','r') as h5:
        for group,rows in metadata.groupby('recording',sort=False):
            node=h5[group]
            columns=json.loads(node.attrs['columns_json'])
            raw=node['imu'][()][:,[columns.index(c) for c in PELVIS_COLS]]
            for row in rows.itertuples():
                window=raw[row.start_sample:row.end_sample].T.astype(np.float32,copy=False)
                features[row.Index]=extract_features_window(window)
    train=metadata.partition.eq('enrollment').to_numpy()
    destination=root/'outputs/traditional';destination.mkdir(parents=True,exist_ok=True)
    for name in ['lr','svm_rbf','rf']:
        classifier=make_classifier(name,62)
        classifier.fit(features[train],metadata.loc[train,'label'])
        results=[]
        for condition,part in metadata.loc[~train].groupby('condition',sort=True):
            y=part.label.to_numpy();pred=classifier.predict(features[part.index])
            ids=part.subject+'|'+part.condition+'|'+part.trial.astype(str)
            trial_accuracy,trial_f1=per_trial_metrics(y,pred,ids.tolist())
            results.append(dict(classifier=name,condition=condition,acc_window=accuracy_score(y,pred),
                f1_window=f1_score(y,pred,average='macro'),acc_trial=trial_accuracy,f1_trial=trial_f1,win_s=2.,seed=42))
        frame=pd.DataFrame(results)
        row=dict(classifier=name,condition='OVERALL_MEAN',win_s=2.,seed=42,
                 **{c:frame[c].mean() for c in ['acc_window','f1_window','acc_trial','f1_trial']})
        frame=pd.concat([frame,pd.DataFrame([row])],ignore_index=True)
        frame.to_csv(destination/f'results_tradml_{name}_CONSISTENT.csv',index=False,float_format='%.4f')
        print('Completed conventional baseline:',name,flush=True)
    provenance=dict(python=platform.python_version(),numpy=np.__version__,scikit_learn=sklearn.__version__,
        source_sha256=sha256(Path(__file__)),windows_sha256=sha256(root/'outputs/data/window_200/windows.csv'),
        source_h5_sha256=sha256(root/'data/altergait/gait_id_pelvis_62.h5'),
        output_sha256={p.name:sha256(p) for p in destination.glob('*.csv')},
        seed=SEED,feature_dimension=41,enrollment_windows=int(train.sum()),test_windows=int((~train).sum()))
    (destination/'training.json').write_text(json.dumps(provenance,indent=2))

if __name__=='__main__':main()


if __name__ == '__main__':
    main()
