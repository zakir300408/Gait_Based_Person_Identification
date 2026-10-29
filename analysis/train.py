"""Run the complete fixed training protocol."""


from pathlib import Path
import subprocess
import json
import os
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]


SOURCES = {
    'hydra':('https://github.com/angus924/hydra','144bb7aa3186654042f00f36b078370c620acb06'),
    'InceptionTime':('https://github.com/hfawaz/InceptionTime','952d115e83fb3a66d75c5858c702f8dd5eeb18b7'),
    'Gait_Recognition_LSTM':('https://github.com/halam189/Gait_Recognition_LSTM','c837c85a2b545cfa5387bea5cb9e560504ff3ad5'),
}


def prepare_sources():
    for name,(url,revision) in SOURCES.items():
        path = ROOT/'external'/name
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            subprocess.run(['git','clone',url,str(path)],check=True)
            subprocess.run(['git','-C',str(path),'checkout','--detach',revision],check=True)
        actual = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        dirty = subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip()
        if actual!=revision or dirty:
            raise ValueError(f'{path}: expected clean author source at {revision}; existing files were preserved')
    print('All three author sources match their pinned revisions.',flush=True)


def conda_executable():
    executable=os.environ.get('CONDA_EXE') or shutil.which('conda')
    if not executable:
        raise RuntimeError('Activate gaitprint with Conda before running analysis.train.')
    return executable


def command(module, *arguments, neural=False):
    prefix = [conda_executable(),'run','--no-capture-output','-n','gaitprint-neural','python'] if neural else [sys.executable]
    subprocess.run(prefix+['-m',module,*map(str,arguments)],cwd=ROOT,check=True)


def run_pipeline():
    from analysis.data import sha256
    if Path.cwd().resolve()!=ROOT:
        raise ValueError('Run python -m analysis.train from the repository root')
    manifest = json.loads((ROOT/'results/manifest.json').read_text())
    raw = ROOT/'data/altergait/gait_id_pelvis_62.h5'
    if sha256(raw)!=manifest['source_h5_sha256']:
        raise ValueError('The study HDF5 input differs from the recorded source')
    # Verify the second environment before starting any expensive fits.
    subprocess.run([conda_executable(),'run','-n','gaitprint-neural','python','-c',
                    'import os; os.environ["KERAS_BACKEND"]="torch"; import torch,keras,numpy,pandas,h5py,sklearn; print(torch.__version__,keras.__version__)'],
                   cwd=ROOT,check=True)
    prepare_sources()
    for width in [100,200,300,400,500]:
        command('analysis.data','--config',f'configs/data_{width}.json')
    command('analysis.train','--stage','classical','--config','configs/classical.json')
    command('analysis.train','--stage','neural','--config','configs/neural.json',neural=True)
    command('analysis.train','--stage','design','--config','configs/design.json')
    command('analysis.train','--stage','controls','--config','configs/controls.json')
    command('analysis.traditional')
    command('analysis.train','--stage','collect')
    # Timings are new measurements on this computer, never copied paper values.
    for model in manifest['models']:
        folder = ROOT/'outputs/fits'/('neural' if model in ['inceptiontime','multi_model_lstm'] else 'classical')/'full'/model/'seed_42/fold_0'
        for module,directory in [('analysis.timing','registration'),('analysis.timing','inference')]:
            destination=ROOT/'outputs/retrained/source/timing_final'/model/directory
            marker=destination/('registration.json' if directory=='registration' else 'timing.json')
            if marker.exists():
                validate_timing(marker,folder)
                continue
            command(module,'--run',folder,'--output',destination,*(['--registration'] if directory=='registration' else []),neural=model in ['inceptiontime','multi_model_lstm'])
    from analysis.reproduce import run
    run(ROOT/'outputs/retrained/source',ROOT/'outputs/retrained/paper')


def validate_timing(marker,folder):
    from analysis.data import sha256
    result=json.loads(marker.read_text())
    key='reference_sha256' if marker.name=='registration.json' else 'completed_run_sha256'
    module='timing'
    if (result['status']!='complete' or result[key]!=sha256(folder/'completed.json')
            or result['script_sha256']!=sha256(ROOT/f'analysis/{module}.py')):
        raise ValueError(f'Timing provenance changed: {marker}; preserve this run and use a new output directory')


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--stage':
        stage=sys.argv[2]
        del sys.argv[1:3]
        if stage in {'classical','design','controls'}:
            from analysis import classical
            {'classical':classical.main,'design':classical.design_main,'controls':classical.controls_main}[stage]()
        elif stage=='neural':
            from analysis.neural import main as fit_neural
            fit_neural()
        elif stage=='collect':
            from analysis.reproduce import collect
            collect()
        else:
            raise ValueError(f'Unknown training stage: {stage}')
    else:
        run_pipeline()


if __name__=='__main__':
    main()
