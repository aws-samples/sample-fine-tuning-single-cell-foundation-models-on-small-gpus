# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Donor-aware splits: no group may appear on both sides of a split (T013)."""

from __future__ import annotations

import numpy as np
import pytest

from scfm_ft.splits import group_folds, group_split, indices_for, split_record

SIZES = {"train": 0.8, "valid": 0.1, "test": 0.1}


def _cells(n_donors=10, per_donor=(5, 40), seed=0):
    """Per-cell donor ids and labels: donors of different sizes, each with several labels."""
    rng = np.random.default_rng(seed)
    groups, labels = [], []
    for d in range(n_donors):
        n = int(rng.integers(*per_donor))
        groups += [f"donor{d}"] * n
        labels += list(rng.choice(["T", "B", "NK", "Mono"], size=n))
    return np.array(groups), np.array(labels)


def test_every_group_lands_in_exactly_one_split():
    groups, _ = _cells()
    s = group_split(groups, SIZES, seed=1)
    assigned = s.train + s.valid + s.test
    assert sorted(assigned) == sorted(set(groups.tolist()))
    assert len(assigned) == len(set(assigned))
    assert s.train and s.valid and s.test


def test_cells_follow_their_group():
    groups, _ = _cells()
    s = group_split(groups, SIZES, seed=1)
    tr, va, te = (set(indices_for(groups, m).tolist()) for m in (s.train, s.valid, s.test))
    assert not (tr & va or tr & te or va & te)
    assert len(tr | va | te) == len(groups)
    for name, idx in (("train", tr), ("valid", va), ("test", te)):
        assert set(groups[list(idx)].tolist()) == set(s.of(name))


def test_held_out_splits_reach_their_cell_fraction():
    groups, _ = _cells(n_donors=40, per_donor=(10, 12))
    s = group_split(groups, SIZES, seed=3)
    n = len(groups)
    assert len(indices_for(groups, s.test)) >= 0.1 * n
    assert len(indices_for(groups, s.valid)) >= 0.1 * n
    assert len(indices_for(groups, s.train)) >= 0.7 * n


def test_split_is_deterministic_for_a_seed_and_changes_with_it():
    groups, _ = _cells(n_donors=30)
    assert group_split(groups, SIZES, 7) == group_split(groups, SIZES, 7)
    assert any(group_split(groups, SIZES, 7) != group_split(groups, SIZES, s) for s in range(8, 12))


def test_too_few_groups_is_an_error_not_a_cell_level_fallback():
    groups = np.array(["only_donor"] * 50)  # for example PBMC3k: one donor
    with pytest.raises(ValueError, match="at least 3 groups"):
        group_split(groups, SIZES, 0)
    with pytest.raises(ValueError, match="at least 2 groups"):
        group_split(groups, {"train": 0.9, "valid": 0.0, "test": 0.1}, 0)
    two = np.array(["a"] * 5 + ["b"] * 5)
    s = group_split(two, {"train": 0.9, "valid": 0.0, "test": 0.1}, 0)
    assert len(s.train) == 1 and len(s.test) == 1 and s.valid == []


def test_minimum_groups_still_leave_train_one():
    groups = np.array(["a"] * 2 + ["b"] * 2 + ["c"] * 96)
    s = group_split(groups, {"train": 0.2, "valid": 0.4, "test": 0.4}, 0)
    assert len(s.train) == len(s.valid) == len(s.test) == 1


def test_group_values_keep_their_type():
    # Geneformer filters the dataset with `example[key] in ids`, so ids must match exactly.
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    s = group_split(groups, SIZES, 0)
    assert all(type(g) is int for g in s.train + s.valid + s.test)


def test_group_folds_never_split_a_group_and_cover_every_cell():
    groups, labels = _cells(n_donors=12)
    folds = group_folds(groups, labels, 5, seed=0)
    assert len(folds) == 5
    held = np.concatenate([te for _, te in folds])
    assert sorted(held.tolist()) == list(range(len(groups)))
    for tr, te in folds:
        assert not set(groups[tr].tolist()) & set(groups[te].tolist())
    with pytest.raises(ValueError, match="at least 5 groups"):
        group_folds(groups[: (groups == "donor0").sum()], labels[: (groups == "donor0").sum()], 5, 0)


def test_split_record_reports_labels_the_model_never_trained_on():
    groups = np.array(["a", "a", "b", "b", "c"])
    labels = np.array(["T", "B", "T", "B", "NK"])
    rec = split_record(groups, labels, {"train": ["a", "b"], "test": ["c"]}, "donor_id")
    assert rec["group_column"] == "donor_id"
    assert rec["splits"]["train"]["n_cells"] == 4 and rec["splits"]["test"]["n_groups"] == 1
    assert rec["labels_missing_from_train"] == {"test": ["NK"]}


def test_scgpt_loader_reads_the_group_column(tmp_path):
    loompy = pytest.importorskip("loompy")
    from scfm_ft.backends.scgpt._io import load_counts_matrix

    path = tmp_path / "d.loom"
    loompy.create(
        str(path),
        np.ones((3, 4), dtype=np.float32),
        {"gene_symbols": np.array(["A", "B", "C"])},
        {"cell_type": np.array(["T", "B", "T", "B"]), "donor_id": np.array(["x", "x", "y", "y"])},
    )
    _genes, counts, _labels, groups = load_counts_matrix(path, "cell_type", "donor_id")
    assert counts.shape == (4, 3) and groups.tolist() == ["x", "x", "y", "y"]
    assert load_counts_matrix(path, "cell_type")[3] is None
    with pytest.raises(KeyError, match="Group column 'patient'"):
        load_counts_matrix(path, "cell_type", "patient")


def test_sm_entry_accepts_a_group_column(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("sm_entry", root / "sagemaker" / "sm_entry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["sm_entry.py", "--group-column", "donor_id"])
    assert mod._parse_args().group_column == "donor_id"
    monkeypatch.setattr(sys, "argv", ["sm_entry.py"])
    assert mod._parse_args().group_column == ""
    src = (root / "sagemaker" / "sm_entry.py").read_text()
    assert '"group_column": args.group_column.strip(\'"\') or None' in src
