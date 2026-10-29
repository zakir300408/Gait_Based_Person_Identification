"""Participant-cluster statistics and figure source tables."""


import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import json
from pathlib import Path
from analysis.metrics import eer_from_claims, frozen_thresholds, groups, cluster_intervals, evaluate_case, split_unknowns


SEEDS = {42, 43, 44}


SWAY = {'trunk_sway_small', 'trunk_sway_large'}


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    corrected = np.empty(len(p))
    corrected[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p)-np.arange(len(p)))))
    return corrected


def average_seeds(frame):
    """Retain only complete 3-seed participant cells, never count seeds as people."""
    keys = ['model', 'group', 'subject']
    counts = frame.groupby(keys).seed.agg(lambda values: set(values))
    if any(values != SEEDS for values in counts):
        raise ValueError('Participant summaries require all three predefined seeds')
    sizes = frame.groupby(keys)[['windows', 'trials']].nunique()
    if (sizes.to_numpy() != 1).any():
        raise ValueError('Evaluation units differ between seeds')
    return frame.groupby(keys, as_index=False).mean(numeric_only=True)


def family(group, metric):
    if metric.startswith('rank'):
        return 'sway_rank3_rank5'
    if group in SWAY:
        return f'{metric}_{group}'
    if group not in {'overall', 'trunk_sway_only'}:
        return f'{metric}_seven_conditions_and_joint'
    return f'{metric}_pooled_descriptive'


def identification_statistics(frame, replicates=10000, seed=20261026):
    summaries, comparisons = [], []
    rng = np.random.default_rng(seed)
    for group, rows in frame.groupby('group', sort=True):
        participants = sorted(rows.subject.unique())
        bootstrap = rng.integers(0, len(participants), (replicates, len(participants)))
        for metric in ['window_accuracy', 'trial_accuracy', 'rank3', 'rank5']:
            if metric.startswith('rank') and group not in SWAY:
                continue
            weights_name = 'trials' if metric == 'trial_accuracy' else 'windows'
            values = rows.pivot(index='subject', columns='model', values=metric).reindex(participants)
            weights = rows.pivot(index='subject', columns='model', values=weights_name).reindex(participants)
            if values.isna().any().any() or weights.isna().any().any():
                raise ValueError('Unmatched participants between models')
            if not np.all(weights.to_numpy() == weights.iloc[:, [0]].to_numpy()):
                raise ValueError('Unmatched evaluation units between models')
            weight = weights.iloc[:, 0].to_numpy()
            sampled_weight = weight[bootstrap]
            boot_values = {}
            for model in values:
                accuracy = values[model].to_numpy()
                sampled = (accuracy[bootstrap] * sampled_weight).sum(1) / sampled_weight.sum(1)
                boot_values[model] = sampled
                lo, hi = np.quantile(sampled, [.025, .975])
                summaries.append(dict(group=group, metric=metric, model=model, participants=len(participants),
                     units=int(weight.sum()), estimate=float(np.average(accuracy, weights=weight)),
                     ci_low=float(lo), ci_high=float(hi), participant_mean=float(accuracy.mean()),
                     participant_sd=float(accuracy.std(ddof=1))))
            if 'gaitprint' not in values:
                continue
            for other in values:
                if other == 'gaitprint':
                    continue
                difference = np.round(values.gaitprint.to_numpy()-values[other].to_numpy(), 12)
                result = wilcoxon(difference, zero_method='pratt', alternative='two-sided', method='approx') if np.any(difference) else None
                lo, hi = np.quantile(boot_values['gaitprint']-boot_values[other], [.025, .975])
                comparisons.append(dict(group=group, metric=metric, comparator=other, participants=len(difference),
                     delta=float(np.average(difference, weights=weight)), participant_mean_delta=float(difference.mean()),
                     delta_ci_low=float(lo), delta_ci_high=float(hi), statistic=float(result.statistic) if result else 0.,
                     p_raw=float(result.pvalue) if result else 1., family=family(group, metric)))
    comparisons = pd.DataFrame(comparisons)
    if len(comparisons):
        for _, indices in comparisons.groupby('family').groups.items():
            comparisons.loc[indices, 'p_holm'] = holm(comparisons.loc[indices, 'p_raw'])
            comparisons.loc[indices, 'family_size'] = len(indices)
    return pd.DataFrame(summaries), comparisons


def verification_summary(frame):
    output = []
    keys = ['model', 'protocol', 'group', 'level']
    for values, rows in frame.groupby(keys):
        seed_means = rows.groupby('seed').eer.mean()
        if set(seed_means.index) != SEEDS:
            continue
        expected_folds = {0} if values[1] == 'enrolled' else {1,2,3,4,5}
        if any(set(part.fold) != expected_folds for _, part in rows.groupby('seed')):
            continue
        # Fold variability remains descriptive because enrollment sets overlap.
        fold_means = rows.groupby('fold').eer.mean()
        output.append(dict(zip(keys, values), mean_eer=float(seed_means.mean()),
                           seed_sd_eer=float(seed_means.std(ddof=1)),
                           descriptive_fold_sd=float(fold_means.std(ddof=1)) if len(fold_means)>1 else 0.,
                           seeds=len(seed_means), folds=len(expected_folds)))
    return pd.DataFrame(output)


def frozen_summary(frame):
    """Report macro-fold rates and claim-weighted rates without pooling seeds."""
    output = []
    for model, rows in frame.groupby('model'):
        if set(rows.seed) != SEEDS or any(set(part.fold)!={1,2,3,4,5} for _,part in rows.groupby('seed')):
            continue
        if rows.duplicated(['seed','fold']).any():
            raise ValueError('Duplicate frozen-threshold seed/fold')
        for metric, count in [('far','n_impostors'), ('frr','n_genuine')]:
            sizes = rows.groupby('fold')[count].nunique()
            if (sizes != 1).any():
                raise ValueError('Frozen-threshold claim counts differ between seeds')
            macro = rows.groupby('seed')[metric].mean()
            weighted = rows.groupby('seed').apply(lambda part: np.average(part[metric],weights=part[count]), include_groups=False)
            output.append(dict(model=model, metric=metric, macro_fold_mean=float(macro.mean()),
                macro_seed_sd=float(macro.std(ddof=1)), claim_weighted_mean=float(weighted.mean()),
                claim_weighted_seed_sd=float(weighted.std(ddof=1)),
                claims_per_seed=int(rows.loc[rows.seed.eq(42),count].sum()), seeds=3, folds=5))
    return pd.DataFrame(output)


def write_csv(frame, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory/name, index=False)


def identification(source, output, ledger_path):
    people = pd.read_csv(source/'analysis/participant_accuracy.csv')
    people = people.loc[people.fold.eq(0)]
    averaged = average_seeds(people)
    summary, tests = identification_statistics(averaged)
    ledger = pd.read_csv(ledger_path)
    # The paper retained its original comparison families after excluding four
    # adapters. Their p-values are retained solely to reproduce multiplicity.
    fresh = (source/'training.json').exists()
    if fresh:
        # New fits cannot reuse historical p-values from excluded adapters.
        # Keep original family sizes conservatively with unused comparisons at p=1.
        ledger['p_raw'] = 1.
    family_values = pd.concat([tests[['family','p_raw']], ledger[['family','p_raw']]], ignore_index=True)
    for family, rows in family_values.groupby('family', sort=True):
        adjusted = holm(rows.p_raw)
        retained = rows.index[rows.index < len(tests)]
        positions = rows.index.get_indexer(retained)
        tests.loc[retained, 'p_holm'] = adjusted[positions]
        tests.loc[retained, 'family_size'] = len(rows)
    (output/'statistics').mkdir(parents=True,exist_ok=True)
    (output/'statistics/multiplicity.json').write_text(json.dumps(dict(
        mode='new fits: excluded comparisons assigned p=1' if fresh else 'reference fits: original p-value ledger',
        description='Original comparison family sizes retained; excluded adapters are not benchmark results in this release.'),indent=2))
    for frame, name in [(averaged,'participant_seed_means.csv'),
                        (summary,'identification_summary.csv'),(tests,'paired_comparisons.csv')]:
        write_csv(frame,output/'statistics',name)
    return summary


def verification(source, output):
    claims = pd.read_csv(source/'analysis/trial_claims.csv')
    if claims.score.isna().any() or not np.isfinite(claims.score).all():
        raise ValueError('Trial claims contain nonfinite scores')
    if claims.duplicated(['model','seed','fold','recording','claim']).any():
        raise ValueError('Duplicate trial claims')
    rows = []
    for (model,seed,fold), part in claims.groupby(['model','seed','fold'],sort=True):
        for group, subset in groups(part):
            rows.append(dict(model=model,seed=seed,fold=fold,group=group,
                protocol=subset.protocol.iloc[0],level='trial_mean_score',
                **eer_from_claims(subset.genuine,subset.score)))
    derived = pd.DataFrame(rows)
    write_csv(derived,output/'analysis','verification.csv')
    write_csv(verification_summary(derived),output/'statistics','verification.csv')
    folds = pd.read_csv(source/'data/subject_folds.csv')
    transfer = [frozen_thresholds(part,folds) for _,part in
        claims.loc[claims.fold.eq(0)].groupby(['model','seed'],sort=True)]
    frozen = pd.concat(transfer,ignore_index=True)
    write_csv(frozen,output/'analysis','frozen_thresholds.csv')
    write_csv(frozen_summary(frozen),output/'statistics','frozen_thresholds.csv')
    return claims


def design(source, output):
    people = pd.read_csv(source/'design/participant_by_seed.csv')
    keys = ['window_samples','variant','group','subject']
    if not people.groupby(keys).seed.agg(lambda values:set(values)=={42,43,44}).all():
        raise ValueError('Design summaries require all predefined seeds')
    averaged = people.groupby(keys,as_index=False).mean(numeric_only=True)
    summaries = []
    rng = np.random.default_rng(20261026)
    for values, rows in averaged.groupby(['window_samples','variant','group']):
        accuracy = rows.window_accuracy.to_numpy()
        samples = accuracy[rng.integers(0,len(rows),(10000,len(rows)))].mean(1)
        ci = np.quantile(samples,[.025,.975])
        summaries.append(dict(zip(['window_samples','variant','group'],values),
            participants=len(rows),participant_mean_accuracy=float(accuracy.mean()),
            ci_low=float(ci[0]),ci_high=float(ci[1]),
            pooled_window_accuracy=float(np.average(accuracy,weights=rows.windows)),
            trial_accuracy=float(np.average(rows.trial_accuracy,weights=rows.trials)),
            windows=int(rows.windows.sum()),trials=int(rows.trials.sum())))
    comparisons = []
    for group, rows in averaged.loc[averaged.window_samples.eq(200)].groupby('group'):
        values = rows.pivot(index='subject',columns='variant',values='window_accuracy')
        if values.isna().any().any():
            raise ValueError('Unmatched component participants')
        for variant in ['temporal','temporal_multiscale','temporal_spectral']:
            delta = np.round(values.complete.to_numpy()-values[variant].to_numpy(),12)
            test = wilcoxon(delta,zero_method='pratt',method='approx',alternative='two-sided') if np.any(delta) else None
            comparisons.append(dict(group=group,comparator=variant,participants=len(values),
                mean_delta=float(delta.mean()),p_raw=float(test.pvalue) if test else 1.))
    tests = pd.DataFrame(comparisons)
    tests['p_holm'] = holm(tests.p_raw)
    if len(tests)!=6:
        raise ValueError('Expected six component comparisons')
    write_csv(pd.DataFrame(summaries),output/'design','design_summary.csv')
    write_csv(tests,output/'design','ablation_comparisons.csv')


def controls(source, output, statistics_config):
    # Preserve the original binary values: perturbing tied participant
    # accuracies by one ULP changes Pratt zero handling in Wilcoxon tests.
    people = pd.read_csv(source/'controls/control_participant_accuracy.csv',float_precision='round_trip')
    seed_metrics = pd.read_csv(source/'controls/control_seed_metrics.csv')
    metrics = ['window_accuracy','participant_mean_accuracy','trial_accuracy']
    summary = seed_metrics.groupby(['model','group'])[metrics].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    write_csv(summary.reset_index(),output/'controls','control_summary.csv')
    averages = people.groupby(['model','group','subject'],as_index=False).agg(
        accuracy=('window_accuracy','mean'),correct=('correct','mean'),windows=('windows','mean'))
    rows = []
    for pair in statistics_config['comparisons']:
        for condition in statistics_config['conditions']:
            a = averages.loc[averages.model.eq(pair['left'])&averages.group.eq(condition)]
            b = averages.loc[averages.model.eq(pair['right'])&averages.group.eq(condition)]
            joined = a.merge(b,on='subject',suffixes=('_left','_right'),validate='one_to_one')
            if len(joined)!=len(a) or len(joined)!=len(b):
                raise ValueError('Unmatched control participants')
            delta = (joined.accuracy_left-joined.accuracy_right).to_numpy()
            rng = np.random.default_rng(statistics_config['bootstrap_seed'])
            indices = rng.integers(len(delta),size=(statistics_config['bootstrap_iterations'],len(delta)))
            means = delta[indices].mean(axis=1)
            pooled = joined.correct_left.sum()/joined.windows_left.sum()-joined.correct_right.sum()/joined.windows_right.sum()
            rows.append(dict(left=pair['left'],right=pair['right'],condition=condition,n=len(delta),
                participant_mean_difference=float(delta.mean()),pooled_difference=float(pooled),
                ci_low=float(np.quantile(means,.025)),ci_high=float(np.quantile(means,.975)),
                improved=int((delta>1e-9).sum()),worsened=int((delta < -1e-9).sum()),
                tied=int((abs(delta)<=1e-9).sum()),
                p=float(wilcoxon(delta,zero_method='pratt',alternative='two-sided',method='approx').pvalue)))
    tests = pd.DataFrame(rows)
    if len(tests)!=statistics_config['holm_family']:
        raise ValueError('Control multiplicity family changed')
    tests['p_holm'] = holm(tests.p)
    write_csv(tests,output/'controls','control_paired_statistics.csv')
    # Figure 5a uses seed-averaged per-participant differences, not test windows.
    gains = []
    for condition in statistics_config['conditions']:
        part = averages.loc[averages.group.eq(condition)].pivot(index='subject',columns='model',values='accuracy')
        for subject,value in ((part.gaitprint-part.temporal_budget)*100).items():
            gains.append(dict(subject=subject,condition=condition,accuracy_gain_pp=value))
    write_csv(pd.DataFrame(gains),output/'figure_sources','participant_feature_budget_gains.csv')


def unknown(claims, source, output, config):
    folds = pd.read_csv(source/'data/subject_folds.csv')
    metadata = pd.read_csv(source/'data/windows.csv')
    label_map = metadata[['label','subject']].drop_duplicates().set_index('label').subject.to_dict()
    assignments = split_unknowns(folds,config['split_seed'])
    write_csv(assignments,output/'unknown','identity_roles.csv')
    claims = claims.loc[claims.model.isin(config['models'])&claims.fold.gt(0)]
    evaluations, counts = [], []
    coverage = set()
    for (model,seed,fold),part in claims.groupby(['model','seed','fold'],sort=True):
        enrolled = set(folds.loc[folds.fold.eq(fold)&folds.role.eq('enrolled'),'subject'])
        held = set(folds.loc[folds.fold.eq(fold)&folds.role.eq('held_out'),'subject'])
        # Unknown probes have no genuine claim and claim only enrolled people.
        if not set(part.subject)<=enrolled|held or enrolled&held:
            raise ValueError('Unknown evaluation roster mismatch')
        if not set(part.claim.map(label_map))<=enrolled:
            raise ValueError('A claim targets a person excluded from the fitted classifier')
        if part.loc[part.subject.isin(held),'genuine'].any():
            raise ValueError('Unknown wearer has a genuine enrolled claim')
        coverage.add((model,seed,fold))
        for swap in [0,1]:
            roles = assignments.loc[assignments.fold.eq(fold)&assignments.swap.eq(swap)]
            calibration, probes, threshold = evaluate_case(part,roles,enrolled,config['target_far'])
            probes['swap'] = swap
            probes['claim_subject'] = probes.claim.map(label_map)
            evaluations.append(probes)
            for condition in ['sway_combined','trunk_sway_small','trunk_sway_large']:
                rows = probes if condition=='sway_combined' else probes.loc[probes.condition.eq(condition)]
                neg,pos = rows.loc[rows.genuine.eq(0)],rows.loc[rows.genuine.eq(1)]
                counts.append(dict(model=model,seed=int(seed),fold=int(fold),swap=swap,condition=condition,
                    threshold=threshold,target_far=config['target_far'],n_development_claims=len(calibration),
                    n_development_people=calibration.subject.nunique(),
                    development_far=float(calibration.score.ge(threshold).mean()),
                    n_evaluation_impostor_people=neg.subject.nunique(),n_impostors=len(neg),n_genuine=len(pos),
                    false_accepts=int(neg.accepted.sum()),false_rejects=int((~pos.accepted).sum()),
                    far=float(neg.accepted.mean()),frr=float((~pos.accepted).mean())))
    expected={(m,s,f) for m in config['models'] for s in config['seeds'] for f in range(1,6)}
    if coverage!=expected:
        raise ValueError('Incomplete unknown-wearer model/seed/fold coverage')
    evaluated = pd.concat(evaluations,ignore_index=True)
    write_csv(evaluated,output/'unknown','evaluation_claims.csv')
    write_csv(pd.DataFrame(counts),output/'unknown','fold_rates.csv')
    summaries = []
    for model,part in evaluated.groupby('model'):
        for condition in ['sway_combined','trunk_sway_small','trunk_sway_large']:
            rows = part if condition=='sway_combined' else part.loc[part.condition.eq(condition)]
            neg,pos = rows.loc[rows.genuine.eq(0)],rows.loc[rows.genuine.eq(1)]
            seed_rates = rows.groupby(['seed','genuine']).accepted.mean().unstack()
            summaries.append(dict(model=model,condition=condition,far=float(neg.accepted.mean()),
                frr=float((~pos.accepted).mean()),seed_sd_far=float(seed_rates[0].std(ddof=1)),
                seed_sd_frr=float((1-seed_rates[1]).std(ddof=1)),
                **cluster_intervals(rows,config['bootstrap_iterations'],config['bootstrap_seed'])))
    write_csv(pd.DataFrame(summaries),output/'unknown','summary.csv')
    return evaluated


def figure_sources(evaluated, output):
    """Derive the diagnostic margins and wearer heatmap from frozen claims."""
    rows = evaluated.loc[evaluated.model.eq('gaitprint')].copy()
    rows['margin'] = rows.score-rows.threshold
    genuine = rows.loc[rows.genuine.eq(1)].copy()
    genuine['probe_type'] = 'Genuine claim'
    impostor = rows.loc[rows.genuine.eq(0)].copy()
    keys = ['subject','condition','recording','seed','fold','swap']
    best = impostor.loc[impostor.groupby(keys).score.idxmax()].copy()
    best['probe_type'] = 'Unknown: best claim'
    write_csv(pd.concat([genuine,best],ignore_index=True),output/'figure_sources','verification_margin_source.csv')
    rates = evaluated.loc[evaluated.genuine.eq(1)].copy()
    rates['rejected'] = 1-rates.accepted.astype(int)
    heat = rates.groupby(['subject','model','condition']).rejected.mean().unstack(['model','condition'])
    columns = [(m,c) for m in ['gaitprint','multirocket_hydra'] for c in ['trunk_sway_small','trunk_sway_large']]
    heat = heat.reindex(columns=pd.MultiIndex.from_tuples(columns))
    # Match the paper's participant ordering: mean GaitPrint FRR, then subject.
    heat['sort_key'] = heat['gaitprint'].mean(axis=1)
    heat = heat.sort_values('sort_key',ascending=False,kind='stable').drop(columns='sort_key')
    heat.columns = ['gaitprint__trunk_sway_small','gaitprint__trunk_sway_large',
                    'multirocket_hydra__trunk_sway_small','multirocket_hydra__trunk_sway_large']
    heat.insert(0,'rank',np.arange(1,len(heat)+1))
    heat.to_csv(output/'figure_sources/wearer_rejection_heatmap.csv')


def analyze(source, output, root):
    print('Recomputing participant statistics and original multiplicity corrections',flush=True)
    identification(source,output,root/'results/multiplicity_ledger.csv')
    print('Recomputing verification and enrolled-impostor threshold transfer',flush=True)
    claims = verification(source,output)
    print('Recomputing component, duration and representation-control analyses',flush=True)
    design(source,output)
    controls(source,output,json.loads((root/'configs/control_statistics.json').read_text()))
    print('Calibrating independent unknown-wearer thresholds and resampling uncertainty',flush=True)
    evaluated = unknown(claims,source,output,json.loads((root/'configs/unknown.json').read_text()))
    figure_sources(evaluated,output)
