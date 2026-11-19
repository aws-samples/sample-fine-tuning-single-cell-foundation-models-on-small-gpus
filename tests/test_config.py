# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Config schema tests. No GPU, no geneformer install required."""

from __future__ import annotations

import textwrap

import pytest

from scfm_ft.config import Config


def _write(tmp_path, body: str):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_minimal_config_defaults(tmp_path):
    cfg = Config.from_yaml(
        _write(tmp_path, """
        name: t
        input:
          h5ad_path: data/x.h5ad
        """)
    )
    assert cfg.model.preset == "V2-104M"
    assert cfg.model.input_size == 4096
    assert cfg.model.version == "V2"
    assert cfg.classifier.kind == "cell"
    assert cfg.training.num_crossval_splits == 1


def test_v1_preset_input_size(tmp_path):
    cfg = Config.from_yaml(
        _write(tmp_path, """
        input: { h5ad_path: data/x.h5ad }
        model: { preset: V1-10M }
        """)
    )
    assert cfg.model.input_size == 2048
    assert cfg.model.version == "V1"


def test_model_path_resolves(tmp_path):
    cfg = Config.from_yaml(
        _write(tmp_path, """
        input: { h5ad_path: data/x.h5ad }
        model: { preset: V2-316M, geneformer_repo: /opt/Geneformer }
        """)
    )
    assert str(cfg.model.model_path) == "/opt/Geneformer/Geneformer-V2-316M"


def test_split_sizes_must_sum_to_one(tmp_path):
    with pytest.raises(ValueError, match="sum to 1.0"):
        Config.from_yaml(
            _write(tmp_path, """
            input: { h5ad_path: data/x.h5ad }
            training: { num_crossval_splits: 1, split_sizes: { train: 0.5, valid: 0.1, test: 0.1 } }
            """)
        )


def test_train_all_data_skips_split_check(tmp_path):
    # num_crossval_splits == 0 means no eval split, so split sizes aren't validated.
    cfg = Config.from_yaml(
        _write(tmp_path, """
        input: { h5ad_path: data/x.h5ad }
        training: { num_crossval_splits: 0, split_sizes: { train: 0.5 } }
        """)
    )
    assert cfg.training.num_crossval_splits == 0


def test_invalid_preset_rejected(tmp_path):
    with pytest.raises(ValueError):
        Config.from_yaml(
            _write(tmp_path, """
            input: { h5ad_path: data/x.h5ad }
            model: { preset: V3-999M }
            """)
        )


def test_derived_paths(tmp_path):
    cfg = Config.from_yaml(
        _write(tmp_path, """
        name: run1
        output_dir: /tmp/out
        input: { h5ad_path: data/x.h5ad }
        """)
    )
    assert cfg.tokenized_dir.name == "tokenized"
    assert cfg.model_out_dir.name == "model"
    assert cfg.eval_dir.name == "eval"
