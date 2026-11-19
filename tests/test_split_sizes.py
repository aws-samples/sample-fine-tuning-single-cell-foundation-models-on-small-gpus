# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""--split-sizes wiring and Geneformer cross-validation coverage."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from scfm_ft.finetune import crossval_coverage

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "sagemaker" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_split_covers_only_half_the_cells_under_5_fold():
    """The vendored validate() skips the test hold-out and evaluates 1/9 per fold."""
    cov = crossval_coverage({"train": 0.8, "valid": 0.1, "test": 0.1}, 5)
    assert cov == pytest.approx(0.5)


def test_full_cv_split_covers_every_cell():
    assert crossval_coverage({"train": 0.8, "valid": 0.2, "test": 0.0}, 5) == pytest.approx(1.0)


def test_single_split_coverage_is_the_eval_fraction():
    assert crossval_coverage({"train": 0.8, "valid": 0.1, "test": 0.1}, 1) == pytest.approx(0.1)


def test_sm_entry_parses_split_sizes(monkeypatch):
    sm_entry = _load("sm_entry")
    assert sm_entry._parse_split_sizes("0.8,0.2,0") == {"train": 0.8, "valid": 0.2, "test": 0.0}
    # SageMaker passes hyperparameters JSON-quoted.
    assert sm_entry._parse_split_sizes('"0.8,0.1,0.1"')["test"] == pytest.approx(0.1)
    for bad in ("0.8,0.2", "0.5,0.5,0.5", "0.9,-0.1,0.2"):
        with pytest.raises(ValueError, match="split-sizes"):
            sm_entry._parse_split_sizes(bad)
    monkeypatch.setattr(sys, "argv", ["sm_entry.py", "--split-sizes", "0.8,0.2,0"])
    assert sm_entry._parse_args().split_sizes == "0.8,0.2,0"


def test_launcher_accepts_and_forwards_split_sizes(monkeypatch):
    launch = _load("launch")
    monkeypatch.setattr(
        sys, "argv", ["launch.py", "--image-uri", "x", "--role", "r", "--split-sizes", "0.8,0.2,0"]
    )
    assert launch._parse().split_sizes == "0.8,0.2,0"
    assert '"split-sizes": args.split_sizes' in (ROOT / "sagemaker" / "launch.py").read_text()
