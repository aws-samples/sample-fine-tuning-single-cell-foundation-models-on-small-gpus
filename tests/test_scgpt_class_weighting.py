# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""scGPT class weighting: weight math, config, entry-point and launcher wiring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from scfm_ft.backends.scgpt._train import _class_weights
from scfm_ft.config import Config

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "sagemaker" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- weight math ---------------------------------------------------------------------
Y = np.array([0] * 60 + [1] * 30 + [2] * 10)  # 60/30/10 imbalance


def test_none_returns_no_weights():
    assert _class_weights(Y, 3, "none") is None


def test_balanced_matches_sklearn():
    from sklearn.utils.class_weight import compute_class_weight

    expected = compute_class_weight("balanced", classes=np.arange(3), y=Y)
    np.testing.assert_allclose(_class_weights(Y, 3, "balanced"), expected)


@pytest.mark.parametrize("mode", ["balanced", "sqrt"])
def test_weights_keep_the_loss_scale(mode):
    """sum_c weight_c * count_c == n, so the weighted loss stays on the same scale."""
    w = _class_weights(Y, 3, mode)
    assert (w * np.bincount(Y)).sum() == pytest.approx(len(Y))


def test_rare_classes_get_more_weight_and_sqrt_is_gentler():
    bal = _class_weights(Y, 3, "balanced")
    sq = _class_weights(Y, 3, "sqrt")
    assert bal[2] > bal[1] > bal[0]
    assert sq[2] > sq[1] > sq[0]
    assert sq[2] / sq[0] == pytest.approx(np.sqrt(bal[2] / bal[0]))


def test_class_missing_from_training_gets_zero_weight():
    w = _class_weights(np.array([0, 0, 1]), 3, "balanced")
    assert w[2] == 0.0
    assert (w[:2] > 0).all()


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown class_weighting"):
        _class_weights(Y, 3, "focal")


# --- config ----------------------------------------------------------------------------
def _cfg(tmp_path: Path, extra: str = "") -> Config:
    p = tmp_path / "c.yaml"
    p.write_text(
        "input: { h5ad_path: data/x.h5ad }\n"
        "model: { backend: scgpt, preset: scgpt-human }\n" + extra
    )
    return Config.from_yaml(p)


def test_config_default_is_none(tmp_path):
    assert _cfg(tmp_path).scgpt.class_weighting == "none"


def test_config_accepts_sqrt_and_rejects_unknown(tmp_path):
    assert _cfg(tmp_path, "scgpt: { class_weighting: sqrt }\n").scgpt.class_weighting == "sqrt"
    with pytest.raises(ValueError):
        _cfg(tmp_path, "scgpt: { class_weighting: focal }\n")


# --- SageMaker entry point and launcher -----------------------------------------------
def test_sm_entry_parses_class_weighting(monkeypatch):
    sm_entry = _load("sm_entry")
    monkeypatch.setattr(sys, "argv", ["sm_entry.py", "--backend", "scgpt"])
    assert sm_entry._parse_args().scgpt_class_weighting == "none"
    monkeypatch.setattr(
        sys, "argv", ["sm_entry.py", "--backend", "scgpt", "--scgpt-class-weighting", "balanced"]
    )
    assert sm_entry._parse_args().scgpt_class_weighting == "balanced"


def test_launcher_parses_class_weighting(monkeypatch):
    launch = _load("launch")
    base = ["launch.py", "--image-uri", "x", "--role", "r", "--backend", "scgpt"]
    monkeypatch.setattr(sys, "argv", base)
    assert launch._parse().scgpt_class_weighting == "none"
    monkeypatch.setattr(sys, "argv", [*base, "--scgpt-class-weighting", "sqrt"])
    assert launch._parse().scgpt_class_weighting == "sqrt"
    monkeypatch.setattr(sys, "argv", [*base, "--scgpt-class-weighting", "focal"])
    with pytest.raises(SystemExit):
        launch._parse()


def test_launcher_forwards_class_weighting_hyperparameter():
    """The hyperparameter dict main() submits must carry the flag to sm_entry."""
    src = (ROOT / "sagemaker" / "launch.py").read_text()
    assert '"scgpt-class-weighting": args.scgpt_class_weighting' in src


def test_seed_flag_reaches_entry_point_and_launcher(monkeypatch):
    sm_entry = _load("sm_entry")
    monkeypatch.setattr(sys, "argv", ["sm_entry.py", "--backend", "scgpt"])
    assert sm_entry._parse_args().seed == 42
    monkeypatch.setattr(sys, "argv", ["sm_entry.py", "--backend", "scgpt", "--seed", "3"])
    assert sm_entry._parse_args().seed == 3

    launch = _load("launch")
    monkeypatch.setattr(
        sys, "argv", ["launch.py", "--image-uri", "x", "--role", "r", "--seed", "3"]
    )
    assert launch._parse().seed == 3
    src = (ROOT / "sagemaker" / "launch.py").read_text()
    assert '"seed": args.seed' in src
    assert '"seed": args.seed' in (ROOT / "sagemaker" / "sm_entry.py").read_text()
