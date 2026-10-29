"""Inference and baseline-registration timing."""


import argparse
import gc
import json
import os
from pathlib import Path
import platform
import time
import h5py
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from analysis.data import sha256, PELVIS_COLS, zscore
from analysis.neural import write_json, seed_everything, build_torch, build_keras, resample_windows
from contextlib import ExitStack
from unittest.mock import patch


os.environ.setdefault('KERAS_BACKEND', 'torch')


def synchronize():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def load_predictor(folder):
    run = json.loads((folder/'run.json').read_text())
    complete = json.loads((folder/'completed.json').read_text())
    if complete['status']!='full_completed': raise ValueError('Timing requires a full completed run')
    for relative,digest in complete['output_sha256'].items():
        if sha256(folder/relative)!=digest: raise ValueError(f'Checkpoint artifact changed: {relative}')
    name, config = run['model'], run['config']
    classes = np.load(folder/'scores.npz')['classes']
    if (folder/'checkpoint.joblib').exists():
        import joblib
        from analysis.model import load_hydra_source
        if 'hydra' in name: load_hydra_source(Path(config['hydra_source']))
        saved = joblib.load(folder/'checkpoint.joblib')
        saved['extractor'].threads=4
        for transform in saved['extractor'].transforms:
            transform.n_jobs=4
        return run, lambda raw:zscore(np.ascontiguousarray(raw,dtype=np.float32).T)[None], lambda x: int(saved['classifier'].predict(saved['extractor'].transform(x))[0])
    settings = config['settings'][name]
    backend = settings['backend']
    models=[]
    for member in run['member_seeds']:
        root = folder/f'member_{member}'/'enrollment'
        if backend=='torch':
            builder = build_torch
            model = builder(name,len(classes)).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
            model.load_state_dict(torch.load(root/'model.pt',weights_only=True))
        else:
            model = build_keras(name,len(classes))
            model.load_weights(root/'model.weights.h5')
        models.append(model)
    def prepare(raw):
        x = resample_windows(zscore(np.ascontiguousarray(raw,dtype=np.float32).T)[None],settings['input_length'])
        return x.transpose(0,2,1) if name=='inceptiontime' else x
    def predict(x):
        probabilities=[]
        with torch.no_grad():
            for model in models:
                if backend=='torch':
                    value=model(torch.as_tensor(x,device='cuda' if torch.cuda.is_available() else 'cpu'))
                    if isinstance(value,tuple):value=value[0]
                    probabilities.append(torch.softmax(value,1).cpu().numpy())
                else:
                    value=model(x,training=False)
                    probabilities.append(value.detach().cpu().numpy())
        return int(classes[np.mean(probabilities,axis=0).argmax(1)[0]])
    return run,prepare,predict


def benchmark(folder, output, warmup=5, repeats=50):
    seed_everything(42)
    torch.set_num_threads(4)
    run,prepare,predict=load_predictor(folder)
    data_dir=Path(run['config']['data_dir'])
    data_config=json.loads((data_dir/'config.json').read_text())
    meta=pd.read_csv(data_dir/'windows.csv',dtype={'subject':str})
    enrollment=meta.loc[meta.partition.eq('enrollment')]
    sample=enrollment.sample(n=warmup+repeats,random_state=42,replace=False)
    raw=[]
    with h5py.File(data_config['data_path'],'r') as data:
        for row in sample.itertuples():
            group=data[row.recording]
            columns=json.loads(group.attrs['columns_json'])
            raw.append(np.ascontiguousarray(group['imu'][row.start_sample:row.end_sample][:,[columns.index(c) for c in PELVIS_COLS]],dtype=np.float32))
    rows=[]
    for i,window in enumerate(raw):
        synchronize(); begin=time.perf_counter_ns()
        transformed=prepare(window)
        synchronize(); after_preprocess=time.perf_counter_ns()
        identity=predict(transformed)
        synchronize(); end=time.perf_counter_ns()
        if i>=warmup:
            rows.append(dict(window_index=int(sample.index[i]),predicted=identity,
                        preprocessing_ms=(after_preprocess-begin)/1e6,
                        model_ms=(end-after_preprocess)/1e6,total_ms=(end-begin)/1e6))
    frame=pd.DataFrame(rows)
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'samples.csv',index=False)
    result=dict(model=run['model'],seed=run['seed'],fold=run['fold'],status='complete',
                completed_run_sha256=sha256(folder/'completed.json'),script_sha256=sha256(Path(__file__)),
                warmup=warmup,repeats=repeats,acquisition_seconds=data_config['window_samples']/data_config['sampling_hz'],batch_size=1,
                boundary='accepted raw window to identity. Gate, acquisition, file I/O excluded. Preprocessing included and separately measured.',
                hardware=dict(cpu=platform.processor(),gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,torch=torch.__version__),
                threads=4,
                summary={c:dict(median=float(frame[c].median()),q25=float(frame[c].quantile(.25)),q75=float(frame[c].quantile(.75)))
                         for c in ['preprocessing_ms','model_ms','total_ms']})
    write_json(output/'timing.json',result)
    print(json.dumps(result,indent=2))


class CheckpointClock:
    def __init__(self):
        self.seconds=0.
        self.depth=0

    def wrap(self, function):
        def call(*args, **kwargs):
            outer=self.depth==0
            self.depth+=1
            start=time.perf_counter()
            try:
                return function(*args,**kwargs)
            finally:
                self.depth-=1
                if outer: self.seconds+=time.perf_counter()-start
        return call


def registration_benchmark(folder, output):
    if output.exists(): raise ValueError('Use a new registration timing directory, never resume a timed fit')
    completed=json.loads((folder/'completed.json').read_text())
    if completed['status']!='full_completed': raise ValueError('Reference fit is not complete')
    for relative,digest in completed['output_sha256'].items():
        if sha256(folder/relative)!=digest: raise ValueError(f'Reference artifact changed: {relative}')
    run=json.loads((folder/'run.json').read_text())
    name,config,seed=run['model'],run['config'],run['seed']
    if run['fold']!=0: raise ValueError('Primary timing uses all enrolled identities')
    meta=pd.read_csv(Path(config['data_dir'])/'windows.csv')
    with np.load(folder/'indices.npz') as archive: indices=archive['enrollment']
    if not meta.loc[indices].condition.eq('baseline').all(): raise ValueError('Registration data are not baseline-only')
    data_dir=Path(config['data_dir'])
    with np.load(data_dir/'windows.npz') as archive: x,y=archive['x'][indices],archive['y'][indices]
    classes=np.unique(y)
    labels=np.searchsorted(classes,y)
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    io=CheckpointClock()
    epochs=[]
    synchronize(); start=time.perf_counter()
    if (folder/'checkpoint.joblib').exists():
        from sklearn.linear_model import RidgeClassifier
        from analysis.model import FeatureExtractor
        alpha=json.loads((folder/'selection.json').read_text())['selected_alpha']
        extractor=FeatureExtractor(name,seed,4,Path(config['hydra_source']))
        features=extractor.fit_transform(x,y)
        classifier=RidgeClassifier(alpha=alpha,solver='cholesky').fit(features.astype(np.float64),y)
        check=classifier.decision_function(features[:16])

    else:
        import analysis.neural as runner
        from analysis.neural import resample_windows
        settings=config['settings'][name]
        x=resample_windows(x,settings['input_length'])
        if name=='inceptiontime': x=x.transpose(0,2,1)
        probabilities=[]
        for member in run['member_seeds']:
            count=json.loads((folder/f'member_{member}'/'selection.json').read_text())['selected_epochs']
            epochs.append(count)
            scratch=output/f'member_{member}'; scratch.mkdir()
            with ExitStack() as stack:
                stack.enter_context(patch.object(torch,'save',io.wrap(torch.save)))
                stack.enter_context(patch.object(runner,'write_json',io.wrap(runner.write_json)))
                if settings['backend']=='keras':
                    import keras
                    stack.enter_context(patch.object(keras.Model,'save',io.wrap(keras.Model.save)))
                train=runner.torch_train if settings['backend']=='torch' else runner.keras_train
                model,_,_=train(name,len(classes),x,labels,None,None,settings,config,member,scratch,count,False)
            probabilities.append(runner.torch_scores(model,x[:16]) if settings['backend']=='torch' else model.predict(x[:16],verbose=0))
            del model; gc.collect(); torch.cuda.empty_cache()
        check=np.mean(probabilities,axis=0)
    synchronize(); seconds=time.perf_counter()-start
    if not np.isfinite(check).all(): raise ValueError('Timed registration produced nonfinite outputs')
    np.savez_compressed(output/'baseline_check.npz',scores=check,classes=classes,indices=indices[:16])
    result=dict(status='complete',model=name,seed=seed,fold=0,enrollment_windows=len(x),identities=len(classes),
        selected_epochs=epochs,wall_seconds=seconds,recovery_checkpoint_write_seconds=io.seconds,
        compute_seconds_excluding_checkpoint_writes=seconds-io.seconds,threads=4,
        reference_sha256=sha256(folder/'completed.json'),script_sha256=sha256(Path(__file__)),
        boundary='Fresh final-baseline fit using selected settings. Includes model construction, learned feature extraction/scaling/classifier fitting and a small baseline output check. Excludes raw acquisition, cached-window preparation and hyperparameter selection. Recovery checkpoint writes separately measured.',
        repeated_fits=1,limitation='One full registration repeat at seed 42, not a training-time confidence interval')
    write_json(output/'registration.json',result)
    # These are recovery files generated by this timing run, not reference fits.
    for scratch in output.iterdir():
        if scratch.is_dir():
            for filename in ['last.pt','last.keras','last.rng.pt']:
                path=scratch/filename
                if path.exists(): path.unlink()
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registration',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        (registration_benchmark if args.registration else benchmark)(args.run,args.output)


if __name__=='__main__':
    main()
