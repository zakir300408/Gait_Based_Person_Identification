"""Checks for leakage boundaries, threshold independence and invariance."""
import unittest

import numpy as np
import pandas as pd

from analysis.classical import invariant_channels
from analysis.data import check_partitions, raw_intervals
from analysis.metrics import majority
from analysis.metrics import evaluate_case


class ProtocolTests(unittest.TestCase):
    def test_raw_time_gap_is_required(self):
        config=dict(baseline_train_fraction=.8,validation_gap_samples=200,window_samples=200)
        intervals=raw_intervals(2000,config)
        self.assertEqual(intervals['selection_train'],(0,1600))
        self.assertEqual(intervals['selection_validation'],(1800,2000))
        frame=pd.DataFrame([
            dict(recording='baseline',subject='P1',condition='baseline',partition='selection_train',start_sample=1400,end_sample=1600),
            dict(recording='baseline',subject='P1',condition='baseline',partition='selection_validation',start_sample=1800,end_sample=2000),
            dict(recording='baseline',subject='P1',condition='baseline',partition='enrollment',start_sample=0,end_sample=200),
        ])
        check_partitions(frame,200)
        frame.loc[1,'start_sample']=1700
        with self.assertRaises(ValueError):
            check_partitions(frame,200)

    def test_rotation_control_is_invariant(self):
        rng=np.random.default_rng(7)
        raw=rng.normal(size=(4,6,200))
        rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(rotation)<0:
            rotation[:,0]*=-1
        rotated=np.concatenate([np.einsum('ij,bjt->bit',rotation,raw[:,:3]),
                                np.einsum('ij,bjt->bit',rotation,raw[:,3:])],axis=1)
        np.testing.assert_allclose(invariant_channels(raw),invariant_channels(rotated),atol=1e-12)

    def test_evaluation_scores_cannot_change_threshold(self):
        roles=pd.DataFrame([dict(role='development_impostor',subject='D'),
                            dict(role='evaluation_impostor',subject='U')])
        claims=pd.DataFrame([
            dict(subject='D',condition='dual_task',genuine=0,score=.2),
            dict(subject='D',condition='fpa_small',genuine=0,score=.4),
            dict(subject='U',condition='trunk_sway_small',genuine=0,score=.1),
            dict(subject='E',condition='trunk_sway_small',genuine=1,score=.8),
        ])
        _,first,threshold=evaluate_case(claims,roles,{'E'},.001)
        claims.loc[claims.subject.eq('U'),'score']=100.
        _,second,changed=evaluate_case(claims,roles,{'E'},.001)
        self.assertEqual(threshold,changed)
        self.assertFalse(first.loc[first.subject.eq('U'),'accepted'].iloc[0])
        self.assertTrue(second.loc[second.subject.eq('U'),'accepted'].iloc[0])
        roles.loc[1,'subject']='D'
        with self.assertRaises(ValueError):
            evaluate_case(claims,roles,{'E'},.001)

    def test_primary_trial_ties_use_smallest_label(self):
        self.assertEqual(majority(np.array([9,3,9,3])),3)


if __name__=='__main__':
    unittest.main()
