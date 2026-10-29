"""Ridge-model training, component analysis and representation controls."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits
from analysis.model import FeatureExtractor, load_hydra_source
from analysis.data import sha256, prepare
import h5py
from sklearn.preprocessing import Normalizer, StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    temporary.replace(path)


def select_alpha(train, y, validation, vy, alphas):
    """Exact centered dual ridge path, equal weighting of validation identities."""
    classes = np.unique(y)
    if len(classes) < 3:
        raise ValueError('This multiclass comparison requires at least three enrolled identities')
    mean = train.mean(axis=0, dtype=np.float64)
    z = train.astype(np.float64) - mean
    v = validation.astype(np.float64) - mean
    targets = 2 * (y[:, None] == classes).astype(np.float64) - 1
    target_mean = targets.mean(axis=0)
    values, vectors = eigh(z @ z.T, check_finite=False)
    # Remove only roundoff-negative eigenvalues of the positive semidefinite Gram matrix.
    values = np.maximum(values, 0)
    projected_targets = vectors.T @ (targets - target_mean)
    projected_validation = (v @ z.T) @ vectors
    rows = []
    for alpha in alphas:
        scores = (projected_validation / (values + alpha)) @ projected_targets + target_mean
        predicted = classes[scores.argmax(axis=1)]
        participant_accuracy = [float(np.mean(predicted[vy == c] == c)) for c in np.unique(vy)]
        rows.append(dict(alpha=float(alpha), participant_accuracy=float(np.mean(participant_accuracy)),
                         window_accuracy=float(np.mean(predicted == vy))))
    best = max(range(len(rows)), key=lambda i: rows[i]['participant_accuracy'])
    return rows[best]['alpha'], rows


def partition_indices(metadata, folds, fold, smoke=False):
    subjects = sorted(metadata.loc[metadata.partition.eq('enrollment'), 'subject'].unique())
    if smoke:
        subjects = subjects[:4]
    if fold:
        subjects = sorted(set(subjects) & set(folds.loc[(folds.fold == fold) & folds.role.eq('enrolled'), 'subject']))
    accepted = metadata.subject.isin(subjects)
    result = {role: metadata.index[metadata.partition.eq(role) & accepted].to_numpy()
              for role in ['selection_train', 'selection_validation', 'enrollment']}
    result['test'] = metadata.index[metadata.partition.eq('test')].to_numpy()
    if smoke:
        for role in result:
            subset = metadata.loc[result[role]]
            result[role] = subset.groupby('subject', sort=False).head(3).index.to_numpy()[:24] if role == 'test' else subset.groupby('subject', sort=False).head(8).index.to_numpy()
    if any(not len(result[role]) for role in result):
        raise ValueError('Empty data partition')
    fit_subjects = set(metadata.loc[result['enrollment'], 'subject'])
    if fold:
        held_out = set(folds.loc[(folds.fold == fold) & folds.role.eq('held_out'), 'subject'])
        for role in ['selection_train', 'selection_validation', 'enrollment']:
            if set(metadata.loc[result[role], 'subject']) & held_out:
                raise ValueError('A held-out identity entered fitting or selection')
    for role in ['selection_train', 'selection_validation', 'enrollment']:
        if not metadata.loc[result[role], 'condition'].eq('baseline').all():
            raise ValueError('Nonbaseline condition entered fitting')
        if set(metadata.loc[result[role], 'subject']) != fit_subjects:
            raise ValueError('Selection and enrollment classes differ')
    return result, subjects


def predict_batches(extractor, classifier, x, indices, batch_size):
    pieces = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        scores = classifier.decision_function(extractor.transform(x[batch]))
        if scores.shape != (len(batch), len(classifier.classes_)) or not np.isfinite(scores).all():
            raise ValueError('Invalid multiclass scores')
        pieces.append(scores.astype(np.float32))
    return np.concatenate(pieces)


def run_one(config, config_path, model, seed, fold, smoke=False):
    data_dir = Path(config['data_dir'])
    destination = Path(config['output_dir']) / ('smoke' if smoke else 'full') / model / f'seed_{seed}' / f'fold_{fold}'
    destination.mkdir(parents=True, exist_ok=True)
    finished = destination / 'completed.json'
    source_files = [Path(__file__), Path('analysis/model.py'), Path('analysis/data.py')]
    fingerprint = dict(config_sha256=sha256(config_path), data_manifest_sha256=sha256(data_dir/'data_manifest.json'),
                       source_sha256={str(p):sha256(p) for p in source_files})
    if finished.exists():
        previous = json.loads(finished.read_text())
        if previous['fingerprint'] != fingerprint:
            raise ValueError(f'Completed run provenance differs: {destination}')
        for name, digest in previous['output_sha256'].items():
            if sha256(destination/name) != digest:
                raise ValueError(f'Completed output changed: {destination/name}')
        print(f'RESUME verified {model} seed={seed} fold={fold}', flush=True)
        return
    manifest = json.loads((data_dir/'data_manifest.json').read_text())
    for name in ['windows.npz', 'windows.csv', 'subject_folds.csv']:
        if sha256(data_dir/name) != manifest['output_sha256'][name]:
            raise ValueError(f'Data artifact hash mismatch: {name}')
    metadata = pd.read_csv(data_dir/'windows.csv', dtype={'subject':str})
    folds = pd.read_csv(data_dir/'subject_folds.csv', dtype={'subject':str})
    indices, subjects = partition_indices(metadata, folds, fold, smoke)
    with np.load(data_dir/'windows.npz') as archive:
        x, y = archive['x'], archive['y']
    if not np.array_equal(y, metadata.label.to_numpy()) or not np.array_equal(metadata.window_index, np.arange(len(y))):
        raise ValueError('Data/metadata alignment failure')
    np.random.seed(seed)
    environment = dict(python=platform.python_version(), platform=platform.platform(), pid=os.getpid(),
                       packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','sktime','torch','numba']})
    write_json(destination/'run.json', dict(model=model, seed=seed, fold=fold, smoke=smoke,
              enrolled_subjects=subjects, counts={k:len(v) for k,v in indices.items()},
              environment=environment, fingerprint=fingerprint, config=config,
              classifier_adaptation='Ridge alpha selected by isolated baseline chronology instead of overlapping-window leave-one-out CV'))
    np.savez_compressed(destination/'indices.npz', **indices)
    started = time.perf_counter()
    print(f'SELECT {model} seed={seed} fold={fold}', flush=True)
    extractor = FeatureExtractor(model,seed,config['threads'],Path(config['hydra_source']))
    train = extractor.fit_transform(x[indices['selection_train']], y[indices['selection_train']])
    validation = extractor.transform(x[indices['selection_validation']])
    alphas = config['gaitprint_alphas'] if model == 'gaitprint' else config['alphas']
    alpha, path = select_alpha(train,y[indices['selection_train']],validation,y[indices['selection_validation']],alphas)
    selection_seconds = time.perf_counter()-started
    write_json(destination/'selection.json', dict(selected_alpha=alpha,candidates=path,
               selection_seconds=selection_seconds, feature_dimension=train.shape[1], test_outcomes_accessed=False))
    del extractor, train, validation
    gc.collect()
    print(f'ENROLL {model} seed={seed} fold={fold} alpha={alpha}', flush=True)
    started = time.perf_counter()
    extractor = FeatureExtractor(model,seed,config['threads'],Path(config['hydra_source']))
    enrollment = extractor.fit_transform(x[indices['enrollment']],y[indices['enrollment']])
    # Use float64 for the ridge solve, including small alphas on sparse features.
    classifier = RidgeClassifier(alpha=alpha, solver='cholesky').fit(enrollment.astype(np.float64),y[indices['enrollment']])
    enrollment_seconds = time.perf_counter()-started
    feature_dimension = enrollment.shape[1]
    del enrollment
    joblib.dump(dict(extractor=extractor,classifier=classifier),destination/'checkpoint.joblib',compress=3)
    check_indices = indices['enrollment'][:min(5,len(indices['enrollment']))]
    expected = predict_batches(extractor,classifier,x,check_indices,config['prediction_batch'])
    reloaded = joblib.load(destination/'checkpoint.joblib')
    observed = predict_batches(reloaded['extractor'],reloaded['classifier'],x,check_indices,config['prediction_batch'])
    np.testing.assert_allclose(observed,expected,rtol=1e-6,atol=1e-6)
    del reloaded
    print(f'SCORE {model} seed={seed} fold={fold} windows={len(indices["test"])}', flush=True)
    started = time.perf_counter()
    scores = predict_batches(extractor,classifier,x,indices['test'],config['prediction_batch'])
    prediction_seconds = time.perf_counter()-started
    np.savez_compressed(destination/'scores.npz', scores=scores, classes=classifier.classes_,
                        window_index=indices['test'],true_label=y[indices['test']])
    write_json(destination/'execution.json', dict(selection_seconds=selection_seconds,
               enrollment_seconds=enrollment_seconds, batch_prediction_seconds=prediction_seconds,
               feature_dimension=feature_dimension, checkpoint_reload_verified=True,
               timing_note='Execution log only. Dedicated synchronized latency benchmark follows all training.'))
    outputs = ['run.json','indices.npz','selection.json','checkpoint.joblib','scores.npz','execution.json']
    write_json(finished,dict(status='smoke_completed' if smoke else 'full_completed', fingerprint=fingerprint,
               output_sha256={name:sha256(destination/name) for name in outputs}))
    print(f'SAVED {destination}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/classical.json'))
    parser.add_argument('--model',nargs='+')
    parser.add_argument('--seed',type=int,nargs='+')
    parser.add_argument('--fold',type=int,nargs='+')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with threadpool_limits(limits=config['threads']):
        for model in args.model or config['models']:
            for seed in args.seed or config['seeds']:
                for fold in args.fold if args.fold is not None else config['folds']:
                    run_one(config,args.config,model,seed,fold,args.smoke)


class ComponentExtractor(FeatureExtractor):
    def __init__(self, variant, seed, threads, hydra_root):
        super().__init__('gaitprint',seed,threads,hydra_root)
        self.variant = variant

    def _raw(self,x):
        temporal = [np.asarray(t.transform(self._view(x,i)),dtype=np.float32) for i,t in enumerate(self.transforms)]
        blocks = list(temporal)
        if self.variant in ['temporal_multiscale','complete']:
            blocks.append(self.multiscale.transform(np.asarray(x,dtype=np.float32)))
        if self.variant in ['temporal_spectral','complete']:
            from analysis.model import featurize
            blocks.append(featurize(x))
        return [np.concatenate(blocks,axis=1)]


def run_design(config, config_path, width, variant, seed, smoke=False):
    root = Path(config['output_dir'])/('smoke' if smoke else 'full')/f'window_{width}'/variant/f'seed_{seed}'/ 'fold_0'
    data_path = Path(config['data_configs'][str(width)])
    data_config = json.loads(data_path.read_text())
    prepare(data_path)
    data_dir = Path(data_config['output_dir'])
    fingerprint = dict(config_sha256=sha256(config_path), data_sha256=sha256(data_dir/'data_manifest.json'),
        code_sha256={str(p):sha256(p) for p in [Path(__file__),Path('analysis/model.py'),Path('analysis/data.py')]})
    if (root/'completed.json').exists():
        previous=json.loads((root/'completed.json').read_text())
        if previous['fingerprint'] != fingerprint: raise ValueError('Design run provenance changed')
        for name,digest in previous['output_sha256'].items():
            if sha256(root/name)!=digest: raise ValueError('Design output changed')
        print(f'RESUME {root}',flush=True)
        return
    root.mkdir(parents=True,exist_ok=True)
    metadata=pd.read_csv(data_dir/'windows.csv',dtype={'subject':str})
    folds=pd.read_csv(data_dir/'subject_folds.csv',dtype={'subject':str})
    indices,subjects=partition_indices(metadata,folds,0,smoke)
    with np.load(data_dir/'windows.npz') as data: x,y=data['x'],data['y']
    run_config=dict(config,data_dir=str(data_dir))
    write_json(root/'run.json',dict(model=f'{variant}_{width}',seed=seed,fold=0,smoke=smoke,
        fingerprint=fingerprint,config=run_config,subjects=subjects,counts={k:len(v) for k,v in indices.items()},
        purpose='descriptive component/duration sensitivity, not test-driven design selection'))
    np.savez_compressed(root/'indices.npz',**indices)
    def extractor(): return ComponentExtractor(variant,seed,config['threads'],Path(config['hydra_source']))
    start=time.perf_counter()
    fitted=extractor()
    train=fitted.fit_transform(x[indices['selection_train']],y[indices['selection_train']])
    val=fitted.transform(x[indices['selection_validation']])
    alpha,path=select_alpha(train,y[indices['selection_train']],val,y[indices['selection_validation']],config['alphas'])
    write_json(root/'selection.json',dict(alpha=alpha,candidates=path,feature_dimension=train.shape[1],
        selection_seconds=time.perf_counter()-start,test_outcomes_accessed=False))
    del fitted,train,val
    gc.collect()
    fitted=extractor()
    features=fitted.fit_transform(x[indices['enrollment']],y[indices['enrollment']])
    classifier=RidgeClassifier(alpha=alpha,solver='cholesky').fit(features.astype(np.float64),y[indices['enrollment']])
    del features
    joblib.dump(dict(extractor=fitted,classifier=classifier),root/'checkpoint.joblib',compress=3)
    reloaded=joblib.load(root/'checkpoint.joblib')
    check=indices['enrollment'][:5]
    np.testing.assert_allclose(predict_batches(fitted,classifier,x,check,256),
        predict_batches(reloaded['extractor'],reloaded['classifier'],x,check,256),rtol=1e-6,atol=1e-6)
    del reloaded
    scores=predict_batches(fitted,classifier,x,indices['test'],config['prediction_batch'])
    np.savez_compressed(root/'scores.npz',scores=scores,classes=classifier.classes_,window_index=indices['test'],true_label=y[indices['test']])
    files=[p for p in root.iterdir() if p.is_file() and p.name!='completed.json']
    write_json(root/'completed.json',dict(status='smoke_completed' if smoke else 'full_completed',fingerprint=fingerprint,
        output_sha256={p.name:sha256(p) for p in files},checkpoint_reload_verified=True))
    print(f'SAVED {root}',flush=True)


def design_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/design.json'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); config=json.loads(args.config.read_text())
    jobs=config['jobs'][:1] if args.smoke else config['jobs']
    seeds=config['seeds'][:1] if args.smoke else config['seeds']
    with threadpool_limits(limits=config['threads']):
        for width,variant in jobs:
            for seed in seeds:
                run_design(config,args.config,width,variant,seed,args.smoke)


def invariant_channels(raw):
    """Six SO(3)-invariant channels, computed before axis standardization.

    Inputs are (batch, 6, time), in original physical units. The channels are
    acceleration/gyro norms, their cosine, first-difference norms and cosine.
    A single fixed rotation acts on acceleration and gyro together.
    """
    a, w = np.asarray(raw[:, :3], dtype=np.float64), np.asarray(raw[:, 3:], dtype=np.float64)
    da, dw = np.diff(a, axis=2, prepend=a[:, :, :1]), np.diff(w, axis=2, prepend=w[:, :, :1])
    norm = lambda z: np.sqrt(np.sum(z*z, axis=1))
    an, wn, dan, dwn = map(norm, [a, w, da, dw])
    cosine = lambda u, v, un, vn: np.divide(np.sum(u*v, axis=1), un*vn,
        out=np.zeros_like(un), where=(un*vn)>1e-12)
    return np.stack([an, wn, cosine(a, w, an, wn), dan, dwn,
                     cosine(da, dw, dan, dwn)], axis=1)


def standardize(x):
    return ((x-x.mean(axis=2, keepdims=True))/(x.std(axis=2, keepdims=True)+1e-8)).astype(np.float32)


class TemporalControlExtractor:
    def __init__(self, kernels, seed, threads):
        self.kernels, self.seed, self.threads = kernels, seed, threads

    def raw_features(self, x):
        return np.concatenate([np.asarray(t.transform(np.ascontiguousarray(
            np.diff(x, n=i, axis=2), dtype=np.float64)), dtype=np.float32)
            for i, t in enumerate(self.transforms)], axis=1)

    def fit_transform(self, x, y):
        self.transforms = [MiniRocketMultivariate(num_kernels=k, random_state=self.seed+i,
            n_jobs=self.threads).fit(np.ascontiguousarray(np.diff(x, n=i, axis=2), dtype=np.float64), y)
            for i, k in enumerate(self.kernels)]
        self.normalizer, self.scaler = Normalizer(), StandardScaler(with_mean=False)
        return self.scaler.fit_transform(self.normalizer.fit_transform(self.raw_features(x)))

    def transform(self, x):
        return self.scaler.transform(self.normalizer.transform(self.raw_features(x)))


def load_inputs(config, metadata):
    data_dir = Path(config['data_dir'])
    manifest = json.loads((data_dir/'data_manifest.json').read_text())
    for name in ['windows.npz', 'windows.csv', 'subject_folds.csv']:
        if sha256(data_dir/name) != manifest['output_sha256'][name]:
            raise ValueError(f'Cached input checksum mismatch: {name}')
    if sha256(Path(config['raw_h5'])) != manifest['source_sha256']:
        raise ValueError('Raw HDF5 differs from the existing experiment source')
    with np.load(data_dir/'windows.npz') as archive:
        x, y = archive['x'], archive['y']
    if not np.array_equal(y, metadata.label) or not np.array_equal(metadata.window_index, metadata.index):
        raise ValueError('Input alignment failure')
    invariant = np.empty_like(x)
    with h5py.File(config['raw_h5'], 'r') as h5:
        for recording, rows in metadata.groupby('recording', sort=False):
            raw = h5[recording]['imu'][()]
            batch = np.stack([raw[int(r.start_sample):int(r.end_sample)].T for r in rows.itertuples()])
            # Verify the reloaded raw windows reproduce the existing normalization.
            np.testing.assert_allclose(standardize(batch), x[rows.index], rtol=2e-4, atol=2e-4)
            invariant[rows.index] = standardize(invariant_channels(batch))
    return x, invariant, y


def run_control(config, fingerprint, metadata, folds, x, y, variant, seed):
    root = Path(config['output_dir'])/'controls'/variant/f'seed_{seed}'/'fold_0'
    if (root/'completed.json').exists():
        done = json.loads((root/'completed.json').read_text())
        if done['fingerprint'] != fingerprint:
            raise ValueError('Existing control provenance differs')
        for name, digest in done['output_sha256'].items():
            if sha256(root/name) != digest:
                raise ValueError(f'Changed control output {name}')
        print(f'RESUME {variant} seed={seed}', flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    indices, subjects = partition_indices(metadata, folds, 0)
    kernels = config['variants'][variant]['kernels']
    build = lambda: TemporalControlExtractor(kernels, seed, config['threads'])
    write_json(root/'run.json', dict(model=variant, seed=seed, fold=0, enrolled_subjects=subjects,
        config=config, fingerprint=fingerprint, counts={k:len(v) for k,v in indices.items()}))
    print(f'SELECT {variant} seed={seed}', flush=True)
    started = time.perf_counter()
    extractor = build()
    train = extractor.fit_transform(x[indices['selection_train']], y[indices['selection_train']])
    validation = extractor.transform(x[indices['selection_validation']])
    alpha, candidates = select_alpha(train, y[indices['selection_train']], validation,
        y[indices['selection_validation']], config['alphas'])
    dimension = train.shape[1]
    write_json(root/'selection.json', dict(alpha=alpha, candidates=candidates,
        feature_dimension=dimension, altered_outcomes_accessed=False))
    del extractor, train, validation
    gc.collect()
    extractor = build()
    train = extractor.fit_transform(x[indices['enrollment']], y[indices['enrollment']])
    classifier = RidgeClassifier(alpha=alpha, solver='cholesky').fit(train.astype(np.float64), y[indices['enrollment']])
    del train
    joblib.dump(dict(extractor=extractor, classifier=classifier), root/'checkpoint.joblib', compress=3)
    check = indices['enrollment'][:5]
    reloaded = joblib.load(root/'checkpoint.joblib')
    np.testing.assert_allclose(predict_batches(extractor, classifier, x, check, 5),
        predict_batches(reloaded['extractor'], reloaded['classifier'], x, check, 5), rtol=1e-6, atol=1e-6)
    del reloaded
    scores = predict_batches(extractor, classifier, x, indices['test'], config['prediction_batch'])
    np.savez_compressed(root/'scores.npz', scores=scores, classes=classifier.classes_,
        window_index=indices['test'], true_label=y[indices['test']])
    write_json(root/'completed.json', dict(status='full_completed', fingerprint=fingerprint,
        feature_dimension=dimension, elapsed_seconds=time.perf_counter()-started,
        checkpoint_reload_verified=True, output_sha256={p.name:sha256(p) for p in root.iterdir()
            if p.is_file() and p.name!='completed.json'}))
    print(f'SAVED {variant} seed={seed} dimension={dimension}', flush=True)


def controls_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    metadata = pd.read_csv(Path(config['data_dir'])/'windows.csv')
    folds = pd.read_csv(Path(config['data_dir'])/'subject_folds.csv')
    fingerprint = dict(config_sha256=sha256(args.config), data_manifest_sha256=sha256(Path(config['data_dir'])/'data_manifest.json'),
        source_sha256={str(p):sha256(p) for p in [Path(__file__),Path('analysis/model.py'),Path('analysis/data.py')]},
        packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','sktime','scikit-learn']})
    with threadpool_limits(limits=config['threads']):
        x, invariant, y = load_inputs(config, metadata)
        for variant in config['variants']:
            inputs = invariant if variant=='temporal_invariant' else x
            for seed in config['seeds']:
                run_control(config, fingerprint, metadata, folds, inputs, y, variant, seed)


if __name__ == '__main__':
    main()
