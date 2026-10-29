"""Raw-timeline partitions for baseline enrollment and altered-gait evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold




PELVIS_COLS = ['AccelX_PELVIS (m/s^2)', 'AccelY_PELVIS (m/s^2)', 'AccelZ_PELVIS (m/s^2)', 'GyroX_PELVIS (deg/s)', 'GyroY_PELVIS (deg/s)', 'GyroZ_PELVIS (deg/s)']

def zscore(x_cxt):
    mu = x_cxt.mean(axis=1, keepdims=True)
    sd = x_cxt.std(axis=1, keepdims=True) + 1e-8
    return (x_cxt - mu) / sd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def raw_intervals(samples: int, config: dict) -> dict[str, tuple[int, int]]:
    boundary = int(samples * config['baseline_train_fraction'])
    validation_start = boundary + config['validation_gap_samples']
    width = config['window_samples']
    if boundary < width or samples - validation_start < width:
        raise ValueError(f'Baseline recording too short for isolated validation: {samples}')
    return {'selection_train': (0, boundary),
            'selection_validation': (validation_start, samples),
            'enrollment': (0, samples)}


def segment(raw: np.ndarray, start: int, stop: int, config: dict):
    width, hop = config['window_samples'], config['hop_samples']
    for offset in range(start, stop - width + 1, hop):
        block = raw[offset:offset + width].astype(np.float32, copy=False)
        movement = float(np.sqrt((block[:, :3] ** 2).sum(axis=1)).std())
        if movement >= config['movement_sd_min']:
            yield offset, zscore(block.T), movement


def check_partitions(windows: pd.DataFrame, gap: int) -> None:
    for group, rows in windows.groupby('recording'):
        train = rows.loc[rows.partition.eq('selection_train')]
        val = rows.loc[rows.partition.eq('selection_validation')]
        if not train.empty and not val.empty:
            if int(train.end_sample.max()) + gap > int(val.start_sample.min()):
                raise ValueError(f'Training and validation lack the raw-time gap: {group}')
    enrolled = set(windows.loc[windows.partition.eq('enrollment'), 'subject'])
    for partition in ['selection_train', 'selection_validation']:
        if set(windows.loc[windows.partition.eq(partition), 'subject']) != enrolled:
            raise ValueError(f'Not every enrolled identity has accepted {partition} windows')
    if not windows.loc[windows.partition.ne('test'), 'condition'].eq('baseline').all():
        raise ValueError('An altered condition entered model development or enrollment')


def prepare(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding='utf8'))
    source, out = Path(config['data_path']), Path(config['output_dir'])
    source_hash = sha256(source)
    config_hash = sha256(config_path)
    completed = out/'data_manifest.json'
    if completed.exists():
        previous = json.loads(completed.read_text())
        if previous['source_sha256'] != source_hash or previous['config_sha256'] != config_hash:
            raise ValueError('Existing data output belongs to another source/configuration')
        for name, digest in previous['output_sha256'].items():
            if sha256(out/name) != digest:
                raise ValueError(f'Cached artifact changed: {name}')
        return previous
    out.mkdir(parents=True, exist_ok=True)
    all_x, all_y, windows, recordings, intervals = [], [], [], [], []
    with h5py.File(source, 'r') as h5:
        decode = lambda a: [v.decode() if isinstance(v, bytes) else str(v) for v in a[()]]
        index = h5['index']
        records = list(zip(*(decode(index[k]) for k in ['subject', 'condition', 'trial_id', 'h5_group'])))
        identities = sorted({s for s,c,_,_ in records if c.lower() == 'baseline'})
        label = {subject:i for i,subject in enumerate(identities)}
        seen = set()
        for subject, condition, trial, group in records:
            if group in seen:
                raise ValueError(f'Duplicate HDF5 index entry: {group}')
            seen.add(group)
            if subject not in label:
                raise ValueError(f'Identity without baseline: {subject}')
            node = h5[group]
            fs = float(node.attrs['fs_hz'])
            if fs != config['sampling_hz']:
                raise ValueError(f'Unexpected sampling rate {fs} in {group}')
            columns = json.loads(node.attrs['columns_json'])
            # Match the original loader's C-contiguous float32 channel matrix.
            # Reduction order otherwise changes z-scores by floating-point roundoff.
            raw = np.ascontiguousarray(node['imu'][()][:, [columns.index(c) for c in PELVIS_COLS]],dtype=np.float32)
            if not np.isfinite(raw).all():
                raise ValueError(f'Nonfinite IMU recording: {group}')
            time = node['time_s'][()] if 'time_s' in node else None
            if time is not None and not np.all(np.diff(time) > 0):
                raise ValueError(f'Nonmonotonic recording time: {group}')
            base = dict(subject=subject,condition=condition,trial=trial,recording=group)
            recordings.append(dict(base,samples=len(raw),sampling_hz=fs,duration_seconds=len(raw)/fs,
                                   channel_order='|'.join(PELVIS_COLS)))
            roles = raw_intervals(len(raw),config) if condition.lower() == 'baseline' else {'test':(0,len(raw))}
            for role,(start,stop) in roles.items():
                intervals.append(dict(base,partition=role,start_sample=start,end_sample=stop))
                for offset, window, movement in segment(raw,start,stop,config):
                    windows.append(dict(base,partition=role,start_sample=offset,
                                        end_sample=offset+config['window_samples'],movement_sd=movement,
                                        window_index=len(all_x),label=label[subject]))
                    all_x.append(window)
                    all_y.append(label[subject])
    frame = pd.DataFrame(windows)
    check_partitions(frame,config['validation_gap_samples'])
    folds = []
    splitter = KFold(n_splits=config['folds'],shuffle=True,random_state=config['split_seed'])
    for fold,(enrolled,heldout) in enumerate(splitter.split(identities),1):
        for role,indices in [('enrolled',enrolled),('held_out',heldout)]:
            folds.extend(dict(fold=fold,subject=identities[i],role=role) for i in indices)
    arrays = np.stack(all_x).astype(np.float32)
    if not np.isfinite(arrays).all():
        raise ValueError('Normalization produced nonfinite values')
    np.savez_compressed(out/'windows.npz',x=arrays,y=np.asarray(all_y,dtype=np.int64))
    frame.to_csv(out/'windows.csv',index=False)
    pd.DataFrame(recordings).to_csv(out/'recordings.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'raw_intervals.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'subject_folds.csv',index=False)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    counts = frame.groupby(['partition','condition']).agg(windows=('window_index','size'),
               subjects=('subject','nunique'),recordings=('recording','nunique')).reset_index()
    counts.to_csv(out/'counts.csv',index=False)
    result = dict(source=str(source.resolve()),source_sha256=source_hash,config_sha256=config_hash,
                  participants=len(identities),recordings=len(recordings),identities=identities,
                  counts=counts.to_dict(orient='records'),
                  output_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='data_manifest.json'})
    completed.write_text(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args.config)
    print(json.dumps({k:result[k] for k in ['participants','recordings','counts']},indent=2))
