"""Identification, verification and independent unknown-wearer evaluation."""


import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,auc
from analysis.data import sha256


def eer_from_claims(genuine,scores):
    genuine=np.asarray(genuine,dtype=int)
    scores=np.asarray(scores,dtype=float)
    if set(np.unique(genuine))!={0,1}:raise ValueError('Verification requires both genuine and impostor claims')
    far,tar,thresholds=roc_curve(genuine,scores)
    frr=1-tar
    i=int(np.argmin(np.abs(far-frr)))
    return dict(eer=float((far[i]+frr[i])/2),threshold_at_eer=float(thresholds[i]),
                far_at_eer=float(far[i]),frr_at_eer=float(frr[i]),auc=float(auc(far,tar)),
                n_genuine=int(genuine.sum()),n_impostor=int((genuine==0).sum()))


def majority(predicted):
    values,counts=np.unique(predicted,return_counts=True)
    return int(values[counts.argmax()])


def groups(frame):
    sway=frame.condition.str.startswith('trunk_sway_')
    yield 'overall',frame
    yield 'trunk_sway_only',frame.loc[sway]
    yield 'non_sway',frame.loc[~sway]
    for condition,rows in frame.groupby('condition',sort=True):yield condition,rows


def derive_run(folder,metadata,score_file='scores.npz'):
    complete=json.loads((folder/'completed.json').read_text())
    if complete['status']!='full_completed':raise ValueError('Smoke run excluded from analysis')
    for filename in [score_file,'run.json']:
        if sha256(folder/filename)!=complete['output_sha256'][filename]:raise ValueError(f'Artifact checksum failure: {folder/filename}')
    run=json.loads((folder/'run.json').read_text())
    info={k:run[k] for k in ['model','seed','fold']}
    with np.load(folder/score_file) as archive:
        scores,classes,indices,truth=[archive[k] for k in ['scores','classes','window_index','true_label']]
    frame=metadata.loc[indices].copy().reset_index(drop=True)
    if not np.array_equal(truth,frame.label.to_numpy()):raise ValueError('Scores and truth not aligned')
    if len(set(indices))!=len(indices) or not frame.partition.eq('test').all():raise ValueError('Unexpected scored windows')
    if not np.isfinite(scores).all():raise ValueError('Nonfinite scores')
    frame['predicted']=classes[scores.argmax(axis=1)]
    frame['correct']=frame.predicted.to_numpy()==truth
    column={int(c):i for i,c in enumerate(classes)}
    enrolled=np.isin(truth,classes)
    frame['enrolled']=enrolled
    frame['rank']=np.nan
    frame['score_tie_at_truth']=False
    if enrolled.any():
        positions=np.flatnonzero(enrolled)
        true_cols=np.asarray([column[int(truth[i])] for i in positions])
        true_scores=scores[positions,true_cols]
        # Pessimistic rank ties match the existing rank-analysis convention.
        frame.loc[positions,'rank']=(scores[positions]>=true_scores[:,None]).sum(axis=1)
        frame.loc[positions,'score_tie_at_truth']=(scores[positions]==true_scores[:,None]).sum(axis=1)>1
    trial_rows=[];claims=[]
    for recording,rows in frame.groupby('recording',sort=False):
        positions=rows.index.to_numpy()
        values=scores[positions].mean(axis=0)
        first=rows.iloc[0]
        true=int(first.label)
        known=true in column
        base=dict(info,subject=first.subject,condition=first.condition,trial=first.trial,recording=recording)
        prediction=majority(rows.predicted.to_numpy())
        trial_rows.append(dict(base,label=true,predicted=prediction,correct=prediction==true,
                     enrolled=known,windows=len(rows),rank_from_mean_score=int(np.sum(values>=values[column[true]])) if known else None))
        for j,claim in enumerate(classes):
            # Open-set genuine probes contribute only their true claim. Held-out
            # probes contribute every enrolled claim, never a genuine claim.
            if run['fold'] and known and int(claim)!=true:continue
            claims.append(dict(base,claim=int(claim),genuine=int(int(claim)==true),score=float(values[j]),
                               protocol='enrolled' if run['fold']==0 else 'subject_held_out'))
    trials=pd.DataFrame(trial_rows)
    claims=pd.DataFrame(claims)
    subject_rows=[];identification=[];verification=[]
    for group,rows in groups(frame.loc[frame.enrolled]):
        if rows.empty:continue
        group_trials=trials.loc[trials.enrolled & trials.recording.isin(rows.recording)]
        subject=rows.groupby('subject').agg(correct=('correct','sum'),windows=('correct','size'))
        trial_subject=group_trials.groupby('subject').agg(trial_correct=('correct','sum'),trials=('correct','size'))
        subject=subject.join(trial_subject)
        subject['window_accuracy']=subject.correct/subject.windows
        subject['trial_accuracy']=subject.trial_correct/subject.trials
        for rank in range(1,11):
            subject[f'rank{rank}']=rows.assign(reached=rows['rank']<=rank).groupby('subject').reached.mean()
            subject[f'trial_rank{rank}']=group_trials.assign(reached=group_trials.rank_from_mean_score<=rank).groupby('subject').reached.mean()
        subject_rows.extend(dict(info,group=group,subject=s,**r.to_dict()) for s,r in subject.iterrows())
        identification.append(dict(info,group=group,windows=len(rows),trials=len(group_trials),participants=len(subject),
             window_accuracy=float(rows.correct.mean()),participant_mean_accuracy=float(subject.window_accuracy.mean()),
             participant_sd_accuracy=float(subject.window_accuracy.std(ddof=1)),trial_accuracy=float(group_trials.correct.mean()),
             **{f'rank{k}':float((rows['rank']<=k).mean()) for k in range(1,11)},
             **{f'trial_rank{k}':float((group_trials.rank_from_mean_score<=k).mean()) for k in range(1,11)},
             score_tie_windows=int(rows.score_tie_at_truth.sum())))
    for group,rows in groups(claims):
        if rows.empty:continue
        verification.append(dict(info,group=group,protocol=rows.protocol.iloc[0],level='trial_mean_score',
                                  **eer_from_claims(rows.genuine,rows.score)))
    # Window claims are derived from saved score matrices without exporting a
    # redundant multi-million-row CSV. They are descriptive, not independent n.
    for group, rows in groups(frame):
        if rows.empty:continue
        positions=rows.index.to_numpy()
        genuine=truth[positions,None]==classes[None,:]
        keep=np.ones(genuine.shape,dtype=bool) if run['fold']==0 else (~enrolled[positions,None]|genuine)
        verification.append(dict(info,group=group,protocol='enrolled' if run['fold']==0 else 'subject_held_out',
                                 level='window',**eer_from_claims(genuine[keep],scores[positions][keep])))
    return pd.DataFrame(identification),pd.DataFrame(subject_rows),trials,claims,pd.DataFrame(verification)


def frozen_thresholds(claims,folds,target_far=.001):
    if not claims.fold.eq(0).all():raise ValueError('Frozen threshold transfer uses the all-enrolled classifier')
    rows=[]
    for fold,assignment in folds.groupby('fold'):
        development=set(assignment.loc[assignment.role.eq('enrolled'),'subject'])
        evaluation=set(assignment.loc[assignment.role.eq('held_out'),'subject'])
        if development&evaluation:raise ValueError('Threshold participant overlap')
        train=claims.loc[claims.subject.isin(development)&~claims.condition.str.startswith('trunk_sway_')]
        test=claims.loc[claims.subject.isin(evaluation)&claims.condition.str.startswith('trunk_sway_')]
        impostors=train.loc[train.genuine.eq(0),'score'].to_numpy(dtype=float)
        threshold=float(np.nextafter(np.quantile(impostors,1-target_far,method='higher'),np.inf))
        negative=test.loc[test.genuine.eq(0),'score'].to_numpy()
        positive=test.loc[test.genuine.eq(1),'score'].to_numpy()
        rows.append(dict(model=claims.model.iloc[0],seed=int(claims.seed.iloc[0]),fold=int(fold),threshold=threshold,
                         target_far=target_far,development_far=float(np.mean(impostors>=threshold)),
                         far=float(np.mean(negative>=threshold)),frr=float(np.mean(positive<threshold)),
                         n_development_impostors=len(impostors),n_impostors=len(negative),n_genuine=len(positive)))
    return pd.DataFrame(rows)


def split_unknowns(folds, split_seed):
    rows = []
    for fold, group in folds.groupby('fold'):
        unknown = np.array(sorted(group.loc[group.role.eq('held_out'), 'subject']))
        halves = np.array_split(np.random.default_rng(split_seed+int(fold)).permutation(unknown), 2)
        for swap in [0, 1]:
            for role, people in [('development_impostor', halves[swap]),
                                 ('evaluation_impostor', halves[1-swap])]:
                rows.extend(dict(fold=int(fold), swap=swap, role=role, subject=s) for s in people)
    return pd.DataFrame(rows)


def calibrate_threshold(scores, target_far):
    scores = np.asarray(scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Calibration requires finite development impostor scores')
    return float(np.nextafter(np.quantile(scores, 1-target_far, method='higher'), np.inf))


def evaluate_case(claims, roles, enrolled, target_far):
    dev = set(roles.loc[roles.role.eq('development_impostor'), 'subject'])
    test = set(roles.loc[roles.role.eq('evaluation_impostor'), 'subject'])
    if dev & test or (dev | test) & enrolled:
        raise ValueError('Unknown development, evaluation and enrollment identities must be disjoint')
    sway = claims.condition.str.startswith('trunk_sway_')
    calibration = claims.loc[claims.subject.isin(dev) & ~sway & claims.genuine.eq(0)]
    probes = claims.loc[sway & ((claims.subject.isin(test) & claims.genuine.eq(0)) |
                               (claims.subject.isin(enrolled) & claims.genuine.eq(1)))].copy()
    if set(calibration.subject) != dev or not set(probes.loc[probes.genuine.eq(0), 'subject']) <= test:
        raise ValueError('Calibration coverage or evaluation identity violation')
    threshold = calibrate_threshold(calibration.score, target_far)
    probes['threshold'] = threshold
    probes['accepted'] = probes.score.ge(threshold)
    return calibration, probes, threshold


def cluster_intervals(probes, iterations, seed):
    """Fixed-fit/threshold uncertainty; source and claimed identity clusters.

    Seed/swap/fold repeats are first averaged for each physical trial claim.
    FAR uses a two-way (source wearer, claimed enrolled identity) bootstrap;
    FRR uses a source-wearer bootstrap. Models and thresholds are not refitted.
    """
    claims = probes.groupby(['subject','claim_subject','recording','genuine'], as_index=False).accepted.mean()
    people = sorted(set(claims.subject) | set(claims.claim_subject))
    label = {s:i for i,s in enumerate(people)}
    source = claims.subject.map(label).to_numpy()
    target = claims.claim_subject.map(label).to_numpy()
    genuine = claims.genuine.to_numpy().astype(bool)
    accepted = claims.accepted.to_numpy()
    rng = np.random.default_rng(seed)
    unknown_trials = probes.loc[probes.genuine.eq(0)].groupby(
        ['subject','recording','seed','fold','swap'], as_index=False).accepted.max()
    unknown_trials = unknown_trials.groupby(['subject','recording'], as_index=False).accepted.mean()
    unknown_source = unknown_trials.subject.map(label).to_numpy()
    far, frr, unknown_far = [], [], []
    n = len(people)
    for _ in range(iterations):
        sw = rng.multinomial(n, np.full(n, 1/n))
        tw = rng.multinomial(n, np.full(n, 1/n))
        negative_weights = sw[source[~genuine]]*tw[target[~genuine]]
        positive_weights = sw[source[genuine]]
        if negative_weights.sum() and positive_weights.sum():
            far.append(np.average(accepted[~genuine], weights=negative_weights))
            frr.append(np.average(1-accepted[genuine], weights=positive_weights))
            unknown_far.append(np.average(unknown_trials.accepted, weights=sw[unknown_source]))
    return dict(far_low=float(np.quantile(far,.025)), far_high=float(np.quantile(far,.975)),
        frr_low=float(np.quantile(frr,.025)), frr_high=float(np.quantile(frr,.975)),
        unknown_trial_far=float(unknown_trials.accepted.mean()),
        unknown_trial_far_low=float(np.quantile(unknown_far,.025)),
        unknown_trial_far_high=float(np.quantile(unknown_far,.975)),
        unique_unknown_trials=len(unknown_trials),
        unique_genuine_trials=int(genuine.sum()), unique_impostor_trial_claims=int((~genuine).sum()),
        interval_scope='conditional on fixed models and thresholds; source/claim clusters; not seed independence')
