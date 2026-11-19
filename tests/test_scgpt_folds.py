# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Fold-index logic for the scGPT backend (pure sklearn, no scGPT/GPU needed)."""

from __future__ import annotations

import numpy as np
import pytest

from scfm_ft.backends.scgpt._train import _fold_indices, _stratified_split

SPLITS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def _labels(n_per_class=(60, 30, 10), seed=0):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.full(n, i) for i, n in enumerate(n_per_class)])
    return rng.permutation(y)


def test_single_split_matches_legacy_stratified_split():
    """num_crossval_splits=1 must reproduce the earlier single-split results exactly."""
    y = _labels()
    folds = _fold_indices(y, 1, SPLITS, seed=42)
    train, _valid, test = _stratified_split(y, SPLITS, seed=42)
    assert len(folds) == 1
    np.testing.assert_array_equal(folds[0][0], train)
    np.testing.assert_array_equal(folds[0][1], test)


def test_five_fold_holds_out_every_cell_exactly_once():
    y = _labels()
    folds = _fold_indices(y, 5, SPLITS, seed=42)
    assert len(folds) == 5
    held_out = np.concatenate([te for _tr, te in folds])
    assert sorted(held_out.tolist()) == list(range(len(y)))
    for tr, te in folds:
        assert set(tr).isdisjoint(te)
        assert len(tr) + len(te) == len(y)


def test_five_fold_is_stratified():
    y = _labels()
    for _tr, te in _fold_indices(y, 5, SPLITS, seed=42):
        counts = np.bincount(y[te], minlength=3)
        # 60/30/10 per class over 5 folds -> 12/6/2 in every fold.
        np.testing.assert_array_equal(counts, [12, 6, 2])


def test_zero_splits_trains_on_everything_with_no_test():
    y = _labels()
    folds = _fold_indices(y, 0, SPLITS, seed=42)
    assert len(folds) == 1
    assert len(folds[0][0]) == len(y)
    assert len(folds[0][1]) == 0


@pytest.mark.parametrize("seed", [0, 42])
def test_five_fold_is_deterministic_for_a_seed(seed):
    y = _labels()
    a = _fold_indices(y, 5, SPLITS, seed=seed)
    b = _fold_indices(y, 5, SPLITS, seed=seed)
    for (tra, tea), (trb, teb) in zip(a, b):
        np.testing.assert_array_equal(tea, teb)
        np.testing.assert_array_equal(tra, trb)


# --- donor-aware folds (T013) ------------------------------------------------------------------
def _donors(y, n_donors=12, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([f"d{i}" for i in rng.integers(0, n_donors, size=len(y))])


@pytest.mark.parametrize("n_splits", [1, 5])
def test_grouped_folds_never_put_a_donor_on_both_sides(n_splits):
    y = _labels()
    groups = _donors(y)
    folds = _fold_indices(y, n_splits, SPLITS, seed=42, groups=groups)
    assert len(folds) == n_splits
    for train, test in folds:
        assert len(test) > 0
        assert not set(groups[train].tolist()) & set(groups[test].tolist())
        assert len(set(train.tolist()) | set(test.tolist())) == len(y)


def test_grouped_zero_splits_still_trains_on_everything():
    y = _labels()
    ((train, test),) = _fold_indices(y, 0, SPLITS, seed=42, groups=_donors(y))
    assert len(train) == len(y) and len(test) == 0


def test_without_groups_the_cell_level_split_is_unchanged():
    y = _labels()
    a = _fold_indices(y, 5, SPLITS, seed=42)
    b = _fold_indices(y, 5, SPLITS, seed=42, groups=None)
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)
