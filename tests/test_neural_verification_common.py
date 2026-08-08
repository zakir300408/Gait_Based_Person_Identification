"""Structural tests for neural verification score export."""

from __future__ import annotations

import numpy as np

from ablation.neural_verification_common import subject_folds, trial_rows


def test_subject_folds_are_disjoint_and_cover_every_subject_once() -> None:
    data = {"subjects": [f"P{index:03d}" for index in range(1, 63)]}
    folds = subject_folds(data)
    held_out = []
    for _, enrolled, impostors in folds:
        assert set(enrolled).isdisjoint(impostors)
        assert len(enrolled) + len(impostors) == 62
        held_out.extend(impostors)
    assert sorted(held_out) == sorted(data["subjects"])


def test_trial_rows_exports_true_genuine_and_all_impostor_claims() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.6, 0.4]], dtype=float)
    metadata = [
        {"subject": "P001", "condition": "trunk_sway_small", "trial": "Trial"},
        {"subject": "P001", "condition": "trunk_sway_small", "trial": "Trial"},
    ]
    labels = np.asarray([0, 1])
    genuine = trial_rows(
        "Model", 1, "trunk_sway_small", probabilities, metadata,
        labels, np.asarray([0, 0]),
    )
    impostor = trial_rows(
        "Model", 1, "trunk_sway_small", probabilities, metadata,
        labels, None,
    )
    assert len(genuine) == 1
    assert genuine[0]["genuine"] == 1
    assert np.isclose(genuine[0]["score"], 0.7)
    assert len(impostor) == 2
    assert [row["genuine"] for row in impostor] == [0, 0]
    assert np.allclose([row["score"] for row in impostor], [0.7, 0.3])
