"""Reproduce tables and figures from recorded or newly fitted results."""


import json
from pathlib import Path
import shutil
import pandas as pd
from analysis.data import sha256
from analysis.metrics import derive_run
import zipfile
import numpy as np
from analysis.statistics import analyze


ROOT = Path(__file__).resolve().parents[1]


def collect():
    source = ROOT/'outputs/retrained/source'
    (source/'analysis').mkdir(parents=True,exist_ok=True)
    evidence = {name:[] for name in ['identification','participant_accuracy','trial_decisions','trial_claims','verification']}
    provenance = []
    for name in ['classical','neural']:
        config = json.loads((ROOT/f'configs/{name}.json').read_text())
        metadata = pd.read_csv(ROOT/config['data_dir']/'windows.csv')
        for model in config['models']:
            for seed in config['seeds']:
                for fold in config['folds']:
                    folder = ROOT/config['output_dir']/'full'/model/f'seed_{seed}'/f'fold_{fold}'
                    frames = derive_run(folder,metadata)
                    for key,frame in zip(evidence,frames):
                        evidence[key].append(frame)
                    provenance.append(dict(path=folder.relative_to(ROOT).as_posix(),sha256=sha256(folder/'completed.json')))
    for key,frames in evidence.items():
        pd.concat(frames,ignore_index=True).to_csv(source/'analysis'/f'{key}.csv',index=False)
    data = ROOT/'outputs/data/window_200'
    (source/'data').mkdir(exist_ok=True)
    for filename in ['windows.csv','raw_intervals.csv','subject_folds.csv','recordings.csv','counts.csv','data_manifest.json']:
        shutil.copy2(data/filename,source/'data'/filename)
    config = json.loads((ROOT/'configs/design.json').read_text())
    people,pooled = [],[]
    for width,variant in config['jobs']+[[200,'complete']]:
        metadata = pd.read_csv(ROOT/f'outputs/data/window_{width}/windows.csv')
        for seed in config['seeds']:
            folder = (ROOT/f'outputs/fits/classical/full/gaitprint/seed_{seed}/fold_0' if width==200 and variant=='complete'
                      else ROOT/f'outputs/fits/design/full/window_{width}/{variant}/seed_{seed}/fold_0')
            identification,participant,*_ = derive_run(folder,metadata)
            for frame,collection in [(identification,pooled),(participant,people)]:
                frame = frame.loc[frame.group.isin(['trunk_sway_small','trunk_sway_large'])].copy()
                frame['window_samples'] = width
                frame['variant'] = variant
                collection.append(frame)
    (source/'design').mkdir(exist_ok=True)
    pd.concat(people,ignore_index=True).to_csv(source/'design/participant_by_seed.csv',index=False)
    pd.concat(pooled,ignore_index=True).to_csv(source/'design/pooled_by_seed.csv',index=False)
    people,pooled = [],[]
    metadata = pd.read_csv(data/'windows.csv')
    roots = {'gaitprint':ROOT/'outputs/fits/classical/full/gaitprint',
             'temporal':ROOT/'outputs/fits/design/full/window_200/temporal',
             'temporal_budget':ROOT/'outputs/fits/controls/temporal_budget',
             'temporal_invariant':ROOT/'outputs/fits/controls/temporal_invariant'}
    for name,folder in roots.items():
        for seed in [42,43,44]:
            identification,participant,*_ = derive_run(folder/f'seed_{seed}/fold_0',metadata)
            identification['model'] = name
            participant['model'] = name
            people.append(participant)
            pooled.append(identification)
    (source/'controls').mkdir(exist_ok=True)
    pd.concat(people,ignore_index=True).to_csv(source/'controls/control_participant_accuracy.csv',index=False)
    pd.concat(pooled,ignore_index=True).to_csv(source/'controls/control_seed_metrics.csv',index=False)
    (source/'traditional').mkdir(exist_ok=True)
    for path in (ROOT/'outputs/traditional').iterdir():
        if path.is_file() and (path.suffix=='.csv' or path.name=='training.json'):
            shutil.copy2(path,source/'traditional'/path.name)
    (source/'training.json').write_text(json.dumps(dict(status='complete',runs=provenance),indent=2))
    print('Exported 126 primary model/seed/fold runs and all component/control runs.',flush=True)


def extract_evidence():
    manifest = json.loads((ROOT/'results/manifest.json').read_text())
    archive = ROOT/'results/evidence.zip'
    if sha256(archive)!=manifest['archive_sha256']:
        raise ValueError('The supplied evidence archive failed its SHA256 check')
    destination = ROOT/'outputs/reference'
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        if set(zipped.namelist())!=set(manifest['files']):
            raise ValueError('Evidence archive inventory differs from its manifest')
        for name in zipped.namelist():
            path = (destination/name).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise ValueError('Unsafe evidence archive path')
            path.parent.mkdir(parents=True,exist_ok=True)
            with zipped.open(name) as source,path.open('wb') as target:
                shutil.copyfileobj(source,target)
            if sha256(path)!=manifest['files'][name]:
                raise ValueError(f'Evidence hash mismatch: {name}')
    return destination


def compare_frame(observed, expected, keys):
    """Require equal cells, with tolerance only for floating-point roundoff."""
    if observed.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f'Duplicate comparison keys: {keys}')
    a = observed.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError(f'Reference row coverage differs: {keys}')
    for column in b.columns:
        if column not in a:
            raise ValueError(f'Reference column missing: {column}')
        if pd.api.types.is_numeric_dtype(b[column]):
            np.testing.assert_allclose(a[column],b[column],rtol=2e-9,atol=2e-11,equal_nan=True,
                                       err_msg=f'Reproduction mismatch: {column}')
        elif not a[column].fillna('').equals(b[column].fillna('')):
            raise ValueError(f'Reference values differ: {column}')


def verify_reference(source, output):
    checks = [
        ('statistics/identification_summary.csv','expected_statistics/identification_summary.csv',['group','metric','model']),
        ('statistics/paired_comparisons.csv','expected_statistics/paired_comparisons.csv',['group','metric','comparator']),
        ('statistics/frozen_thresholds.csv','expected_statistics/frozen_thresholds.csv',['model','metric']),
        ('design/design_summary.csv','expected_design/design_summary.csv',['window_samples','variant','group']),
        ('design/ablation_comparisons.csv','expected_design/ablation_comparisons.csv',['group','comparator']),
        ('controls/control_summary.csv','expected_controls/control_summary.csv',['model','group']),
        ('controls/control_paired_statistics.csv','expected_controls/control_paired_statistics.csv',['left','right','condition']),
        ('unknown/summary.csv','expected_unknown/summary.csv',['model','condition']),
        ('unknown/fold_rates.csv','expected_unknown/fold_rates.csv',['model','seed','fold','swap','condition']),
    ]
    for actual,expected,keys in checks:
        compare_frame(pd.read_csv(output/actual),pd.read_csv(source/expected),keys)
    actual = pd.read_csv(output/'statistics/verification.csv')
    expected = pd.read_csv(source/'expected_statistics/verification.csv')
    expected = expected.loc[expected.level.eq('trial_mean_score')]
    compare_frame(actual,expected,['model','protocol','group','level'])
    return len(checks)+1


def tables(source, output):
    directory = output/'tables'
    directory.mkdir(parents=True,exist_ok=True)
    # Numerical tables remain machine-readable; no values are transcribed.
    for src,dest in [('statistics/identification_summary.csv','identification.csv'),
                     ('statistics/verification.csv','verification_eer.csv'),
                     ('statistics/frozen_thresholds.csv','enrolled_threshold_transfer.csv'),
                     ('unknown/summary.csv','unknown_wearer.csv'),
                     ('controls/control_paired_statistics.csv','representation_controls.csv'),
                     ('design/design_summary.csv','duration_and_components.csv')]:
        shutil.copy2(output/src,directory/dest)
    for path in (source/'traditional').glob('*.csv'):
        shutil.copy2(path,directory/path.name)


def run(source, output, check_reference=False):
    output.mkdir(parents=True,exist_ok=True)
    analyze(source,output,ROOT)
    count = verify_reference(source,output) if check_reference else 0
    tables(source,output)
    from analysis.figures import render_all
    render_all(source,output)
    report = dict(status='complete',reference_tables_verified=count,
        source=str(source.relative_to(ROOT)),
        output_sha256={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*')
                       if p.is_file() and p.name!='reproduction.json'},
        note=('Saved-score analysis verifies the reported arithmetic; it does not retrain models or remeasure timing.'
              if check_reference else 'Analyses of fresh fits and current-computer timings; not a reference-value equality check.'))
    (output/'reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(f'Complete: {output.relative_to(ROOT)}; {count} reference tables verified.',flush=True)


def main():
    source = extract_evidence()
    run(source,ROOT/'outputs/paper',check_reference=True)


if __name__ == '__main__':
    main()
