"""Pinned neural architectures, baseline selection and final fitting."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
import argparse
import gc
import importlib.metadata
import json
import os
import platform
import random
import time
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from analysis.data import sha256


AUTHOR_COMMITS = {'Gait_Recognition_LSTM': 'c837c85a2b545cfa5387bea5cb9e560504ff3ad5', 'InceptionTime': '952d115e83fb3a66d75c5858c702f8dd5eeb18b7'}


def check_source(repository):
    root=Path('external')/repository
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    changes=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if head!=AUTHOR_COMMITS[repository] or changes:
        raise ValueError(f'Author source not clean at pinned commit: {repository}')
    return root


def resample_windows(x,length):
    """Linear temporal resampling, retaining the same two-second interval."""
    if x.shape[-1]==length:return np.ascontiguousarray(x,dtype=np.float32)
    old=np.arange(x.shape[-1])
    new=np.linspace(0,x.shape[-1]-1,length)
    return np.stack([np.stack([np.interp(new,old,c) for c in w]) for w in x]).astype(np.float32)


def build_torch(name, classes):
    if name != 'multi_model_lstm':
        raise ValueError('This audited adapter is only for Multi-Model LSTM')
    root = check_source('Gait_Recognition_LSTM')
    if 'gaitprint_author_lstm' not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            'gaitprint_author_lstm', root/'LSTM_Pthon/authen_netw.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules['gaitprint_author_lstm'] = module
        spec.loader.exec_module(module)
    return sys.modules['gaitprint_author_lstm'].LSTM_6Chan_OU(
        10, 40, 2, classes, segment_length=200)


def build_keras(name,classes):
    import keras
    if name=='inceptiontime':
        root=check_source('InceptionTime')
        source=(root/'classifiers/inception.py').read_text()
        tree=ast.parse(source)
        author_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Classifier_INCEPTION')
        methods=[n for n in author_class.body if isinstance(n,ast.FunctionDef) and n.name in ['_inception_module','_shortcut_layer']]
        # Import the two original architecture methods without their legacy training/I/O.
        namespace={'keras':keras}
        for method in methods:
            text=ast.unparse(method).replace('keras.layers.normalization.BatchNormalization','keras.layers.BatchNormalization')
            exec(compile(text,str(root/'classifiers/inception.py'),'exec'),namespace)
        class Architecture:
            use_bottleneck=True
            bottleneck_size=32
            kernel_size=40
            nb_filters=32
        Architecture._inception_module=namespace['_inception_module']
        Architecture._shortcut_layer=namespace['_shortcut_layer']
        architecture=Architecture()
        inputs=keras.layers.Input((200,6))
        current=residual=inputs
        for depth in range(6):
            current=architecture._inception_module(current)
            if depth%3==2:
                current=architecture._shortcut_layer(residual,current)
                residual=current
        outputs=keras.layers.Dense(classes,activation='softmax')(keras.layers.GlobalAveragePooling1D()(current))
        model=keras.Model(inputs,outputs)
        model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        return model
    raise ValueError(name)


os.environ.setdefault('KERAS_BACKEND','torch')


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf8')
    temp.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def partitions(meta,folds,fold,smoke):
    enrolled=set(meta.loc[meta.partition.eq('enrollment'),'subject'])
    if fold:enrolled &= set(folds.loc[(folds.fold==fold)&folds.role.eq('enrolled'),'subject'])
    if smoke:enrolled=set(sorted(enrolled)[:4])
    result={p:meta.index[meta.partition.eq(p)&meta.subject.isin(enrolled)].to_numpy()
            for p in ['selection_train','selection_validation','enrollment']}
    result['test']=meta.index[meta.partition.eq('test')].to_numpy()
    if smoke:
        result={p:meta.loc[i].groupby('subject',sort=False).head(4).index.to_numpy()[:24] if p=='test'
                else meta.loc[i].groupby('subject',sort=False).head(8).index.to_numpy() for p,i in result.items()}
    for p in ['selection_train','selection_validation','enrollment']:
        if set(meta.loc[result[p],'subject'])!=enrolled or not meta.loc[result[p],'condition'].eq('baseline').all():
            raise ValueError('Invalid baseline-only identity partitions')
    return result


def torch_scores(model,x,batch=128):
    model.eval()
    output=[]
    device=next(model.parameters()).device
    with torch.no_grad():
        for start in range(0,len(x),batch):
            value=model(torch.from_numpy(x[start:start+batch]).to(device))
            if isinstance(value,tuple):value=value[0]
            output.append(torch.softmax(value,dim=1).cpu().numpy())
    return np.concatenate(output)


def validation_metrics(probabilities,y):
    if not np.isfinite(probabilities).all():raise ValueError('Nonfinite validation probabilities')
    pred=probabilities.argmax(axis=1)
    loss=float(-np.log(np.maximum(probabilities[np.arange(len(y)),y],1e-12)).mean())
    accuracy=float(np.mean([np.mean(pred[y==c]==c) for c in np.unique(y)]))
    return accuracy,loss


def torch_train(name,classes,x,y,v,vy,settings,config,seed,folder,epochs,selection):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed)
    model=build_torch(name,classes).to(device)
    optimizer=(torch.optim.SGD(model.parameters(),lr=settings['lr'],momentum=settings['momentum'])
               if settings['optimizer']=='SGD' else torch.optim.Adam(model.parameters(),lr=settings['lr']))
    history=[]
    best=-1.
    best_epoch=0
    state_path=folder/'last.pt'
    if state_path.exists():
        saved=torch.load(state_path,map_location=device,weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch']
        random.setstate(saved['random_state']);np.random.set_state(saved['numpy_state'])
        torch.set_rng_state(saved['torch_state'].cpu())
        if torch.cuda.is_available():torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_state']])
    inputs=torch.from_numpy(x).to(device)
    targets=torch.from_numpy(y).long().to(device)
    started=time.perf_counter()
    for epoch in range(len(history),epochs):
        if selection and history and epoch-best_epoch>=config['patience']:break
        model.train()
        order=torch.randperm(len(x),device=device)
        loss_sum=0.
        for indices in order.split(settings['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            logits=model(inputs[indices])
            if isinstance(logits,tuple):logits=logits[0]
            loss=torch.nn.functional.cross_entropy(logits,targets[indices])
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(indices)
        row=dict(epoch=epoch+1,train_loss=loss_sum/len(x))
        if selection:
            accuracy,val_loss=validation_metrics(torch_scores(model,v,config['prediction_batch']),vy)
            row.update(validation_participant_accuracy=accuracy,validation_loss=val_loss)
            if accuracy>best:best,best_epoch=accuracy,epoch+1
        history.append(row)
        temporary=folder/'last.tmp.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=best,
                   best_epoch=best_epoch,random_state=random.getstate(),numpy_state=np.random.get_state(),
                   torch_state=torch.get_rng_state(),cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),temporary)
        temporary.replace(state_path)
        write_json(folder/'history.json',history)
        print(f'{name} {folder.parent.name}/{folder.name} epoch={epoch+1} loss={row["train_loss"]:.5f}'+
              (f' baseline_val={best:.5f}' if selection else ''),flush=True)
    return model,best_epoch,dict(execution_seconds=time.perf_counter()-started,epochs=len(history))


def keras_train(name,classes,x,y,v,vy,settings,config,seed,folder,epochs,selection):
    import keras
    keras.backend.clear_session()
    seed_everything(seed);keras.utils.set_random_seed(seed)
    state_path=folder/'last.keras'
    progress_path=folder/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else dict(history=[],best=-1.,best_epoch=0)
    model=keras.models.load_model(state_path,safe_mode=False) if state_path.exists() else build_keras(name,classes)
    if bool(state_path.exists())!=bool(progress_path.exists()):raise ValueError('Incomplete Keras checkpoint transaction')
    rng_path=folder/'last.rng.pt'
    if state_path.exists() and rng_path.exists():
        saved_rng=torch.load(rng_path,weights_only=False)
        random.setstate(saved_rng['python']);np.random.set_state(saved_rng['numpy'])
        torch.set_rng_state(saved_rng['torch'])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(saved_rng['cuda'])
        for variable in model.variables:
            if variable.path in saved_rng['keras_seed_variables']:
                variable.assign(saved_rng['keras_seed_variables'][variable.path])
    scheduler=keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=.5,patience=50,min_lr=.0001) if name=='inceptiontime' else None
    class SchedulerState(keras.callbacks.Callback):
        def on_train_begin(self,logs=None):
            if scheduler:
                for key,value in progress.get('scheduler',{}).items():setattr(scheduler,key,value)
    class Checkpoint(keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            row=dict(epoch=epoch+1,train_loss=float(logs['loss']))
            if selection:
                probabilities=self.model.predict(v,batch_size=config['prediction_batch'],verbose=0)
                accuracy,val_loss=validation_metrics(probabilities,vy)
                row.update(validation_participant_accuracy=accuracy,validation_loss=val_loss)
                if accuracy>progress['best']:progress.update(best=accuracy,best_epoch=epoch+1)
                if epoch+1-progress['best_epoch']>=config['patience']:self.model.stop_training=True
            progress['history'].append(row)
            temporary=folder/'last.tmp.keras'
            self.model.save(temporary)
            temporary.replace(state_path)
            if scheduler:progress['scheduler']={key:float(getattr(scheduler,key)) if key=='best' else int(getattr(scheduler,key)) for key in ['best','wait','cooldown_counter']}
            torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                       cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                       keras_seed_variables={variable.path:keras.ops.convert_to_numpy(variable) for variable in self.model.variables if 'seed_generator' in variable.path}),rng_path)
            write_json(progress_path,progress)
            write_json(folder/'history.json',progress['history'])
            print(f'{name} {folder.parent.name}/{folder.name} epoch={epoch+1} loss={row["train_loss"]:.5f}'+
                  (f' baseline_val={progress["best"]:.5f}' if selection else ''),flush=True)
    start=len(progress['history'])
    started=time.perf_counter()
    if start<epochs and not (selection and start and start-progress['best_epoch']>=config['patience']):
        model.fit(x,np.eye(classes,dtype=np.float32)[y],batch_size=settings['batch_size'],
                  initial_epoch=start,epochs=epochs,shuffle=True,verbose=0,
                  callbacks=([scheduler,SchedulerState()] if scheduler else [])+[Checkpoint()])
    return model,progress['best_epoch'],dict(execution_seconds=time.perf_counter()-started,epochs=len(progress['history']))


def run(config,config_path,name,seed,fold,smoke):
    data_dir=Path(config['data_dir'])
    out=Path(config['output_dir'])/('smoke' if smoke else 'full')/name/f'seed_{seed}'/f'fold_{fold}'
    out.mkdir(parents=True,exist_ok=True)
    fingerprint=dict(config_sha256=sha256(config_path),data_manifest_sha256=sha256(data_dir/'data_manifest.json'),
                     source_sha256={str(p):sha256(p) for p in [Path(__file__),Path('analysis/data.py')]},author_commits=AUTHOR_COMMITS)
    complete=out/'completed.json'
    if (out/'run.json').exists():
        if json.loads((out/'run.json').read_text())['fingerprint']!=fingerprint:raise ValueError(f'Run provenance changed: {out}')
    if complete.exists():
        for filename,digest in json.loads(complete.read_text())['output_sha256'].items():
            if sha256(out/filename)!=digest:raise ValueError('Completed output changed')
        print(f'RESUME verified {out}',flush=True);return
    data_manifest=json.loads((data_dir/'data_manifest.json').read_text())
    for filename in ['windows.npz','windows.csv','subject_folds.csv']:
        if sha256(data_dir/filename)!=data_manifest['output_sha256'][filename]:raise ValueError('Data changed')
    meta=pd.read_csv(data_dir/'windows.csv',dtype={'subject':str})
    folds=pd.read_csv(data_dir/'subject_folds.csv',dtype={'subject':str})
    indices=partitions(meta,folds,fold,smoke)
    classes=np.sort(meta.loc[indices['enrollment'],'label'].unique())
    class_map={int(c):i for i,c in enumerate(classes)}
    with np.load(data_dir/'windows.npz') as archive:x,y=archive['x'],archive['y']
    if not np.array_equal(y,meta.label.to_numpy()):raise ValueError('Metadata alignment failure')
    settings=config['settings'][name]
    x=resample_windows(x,settings['input_length'])
    if name=='inceptiontime':x=np.ascontiguousarray(x.transpose(0,2,1))
    member_count=settings.get('ensemble_members',1)
    member_seeds=[100*seed+i for i in range(member_count)] if member_count>1 else [seed]
    environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                     hardware=torch.cuda.get_device_name() if torch.cuda.is_available() else platform.processor(),
                     pid=os.getpid())
    write_json(out/'run.json',dict(model=name,seed=seed,fold=fold,smoke=smoke,fingerprint=fingerprint,
               counts={r:len(i) for r,i in indices.items()},classes=classes.tolist(),member_seeds=member_seeds,
               environment=environment,config=config,score_definition='Identification probabilities; trial verification uses their mean per claimed identity.'))
    np.savez_compressed(out/'indices.npz',**indices)
    collected=[]
    train_fn=torch_train if settings['backend']=='torch' else keras_train
    epochs=2 if smoke else config['max_epochs']
    local_y={r:np.asarray([class_map[int(z)] for z in y[indices[r]]],dtype=np.int64)
             for r in ['selection_train','selection_validation','enrollment']}
    for member_seed in member_seeds:
        member_dir=out/f'member_{member_seed}'
        selection_dir=member_dir/'selection';selection_dir.mkdir(parents=True,exist_ok=True)
        final_dir=member_dir/'enrollment';final_dir.mkdir(parents=True,exist_ok=True)
        selection_file=member_dir/'selection.json'
        if selection_file.exists():chosen=json.loads(selection_file.read_text())['selected_epochs']
        else:
            model,chosen,execution=train_fn(name,len(classes),x[indices['selection_train']],local_y['selection_train'],
                       x[indices['selection_validation']],local_y['selection_validation'],settings,config,member_seed,selection_dir,epochs,True)
            write_json(selection_file,dict(selected_epochs=chosen,test_outcomes_accessed=False,execution=execution))
            del model;gc.collect();torch.cuda.empty_cache()
        if chosen<1:raise ValueError('Epoch selection failed')
        model,_,execution=train_fn(name,len(classes),x[indices['enrollment']],local_y['enrollment'],None,None,
                     settings,config,member_seed,final_dir,chosen,False)
        write_json(final_dir/'execution.json',execution)
        pred_fn=(lambda m,z:torch_scores(m,z,config['prediction_batch'])) if settings['backend']=='torch' else (
                lambda m,z:m.predict(z,batch_size=config['prediction_batch'],verbose=0))
        check=x[indices['enrollment'][:5]]
        before=pred_fn(model,check)
        if settings['backend']=='torch':
            saved=torch.load(final_dir/'last.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'])
        else:
            import keras
            del model;gc.collect();torch.cuda.empty_cache()
            model=keras.models.load_model(final_dir/'last.keras',safe_mode=False)
        np.testing.assert_allclose(pred_fn(model,check),before,rtol=1e-5,atol=1e-6)
        probabilities=pred_fn(model,x[indices['test']])
        if probabilities.shape!=(len(indices['test']),len(classes)) or not np.isfinite(probabilities).all():raise ValueError('Invalid model scores')
        np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-5)
        # Retain inference checkpoints. Optimizer-bearing recovery checkpoints are
        # temporary and are removed only after the complete run manifest is saved.
        if settings['backend']=='torch':
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},final_dir/'model.pt')
        else:
            model.save_weights(final_dir/'model.weights.h5')
        np.savez_compressed(member_dir/'scores.npz',scores=probabilities,classes=classes,window_index=indices['test'],true_label=y[indices['test']])
        collected.append(probabilities)
        del model;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(out/'scores.npz',scores=np.mean(collected,axis=0),classes=classes,window_index=indices['test'],true_label=y[indices['test']])
    transient={'last.pt','last.keras','last.rng.pt'}
    files=[p for p in out.rglob('*') if p.is_file() and p.name not in transient|{'completed.json'}]
    write_json(complete,dict(status='smoke_completed' if smoke else 'full_completed',fingerprint=fingerprint,
               checkpoint_reload_verified=True,output_sha256={str(p.relative_to(out)):sha256(p) for p in files}))
    for temporary in out.rglob('last.*'):
        if temporary.name in transient and temporary.resolve().is_relative_to(out.resolve()):
            temporary.unlink()
    print(f'SAVED {out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/neural.json'))
    parser.add_argument('--model',nargs='+');parser.add_argument('--seed',nargs='+',type=int)
    parser.add_argument('--fold',nargs='+',type=int);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    torch.set_num_threads(config['threads'])
    with threadpool_limits(limits=config['threads']):
        for model in args.model or config['models']:
            for seed in args.seed or config['seeds']:
                for fold in args.fold if args.fold is not None else config['folds']:
                    run(config,args.config,model,seed,fold,args.smoke)


if __name__ == '__main__':
    main()
