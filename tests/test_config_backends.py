# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Config tests for the multi-backend schema. No GPU, no ML stack required."""

from __future__ import annotations

import textwrap

import pytest

from scfm_ft.config import Config


def _write(tmp_path, body: str):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_backend_defaults_to_geneformer(tmp_path):
    cfg = Config.from_yaml(
        _write(tmp_path, """
        input: { h5ad_path: data/x.h5ad }
        """)
    )
    assert cfg.model.backend == "geneformer"
    assert cfg.model.preset == "V2-104M"


def test_scgpt_config_validates(tmp_path):
    cfg = Config.from_yaml(
        _write(tmp_path, """
        input: { h5ad_path: data/x.h5ad }
        model: { backend: scgpt, preset: scgpt-human }
        scgpt: { n_input_bins: 51, use_flash_attn: true, model_root: /scgpt }
        """)
    )
    assert cfg.model.backend == "scgpt"
    assert cfg.model.preset == "scgpt-human"
    assert cfg.scgpt.n_input_bins == 51
    assert str(cfg.scgpt.model_root) == "/scgpt"


def test_scgpt_flash_attn_and_grad_checkpointing_defaults(tmp_path):
    # Defaults: flash-attn on, gradient checkpointing off.
    cfg = Config.from_yaml(
        _write(tmp_path, """
        input: { h5ad_path: data/x.h5ad }
        model: { backend: scgpt, preset: scgpt-human }
        """)
    )
    assert cfg.scgpt.use_flash_attn is True
    assert cfg.scgpt.grad_checkpointing is False
    assert cfg.scgpt.max_seq_len == 1200


def test_scgpt_flash_attn_off_and_grad_checkpointing_on(tmp_path):
    cfg = Config.from_yaml(
        _write(tmp_path, """
        input: { h5ad_path: data/x.h5ad }
        model: { backend: scgpt, preset: scgpt-human }
        scgpt: { use_flash_attn: false, grad_checkpointing: true, max_seq_len: 800 }
        """)
    )
    assert cfg.scgpt.use_flash_attn is False
    assert cfg.scgpt.grad_checkpointing is True
    assert cfg.scgpt.max_seq_len == 800


def test_scgpt_preset_rejected_on_geneformer_backend(tmp_path):
    # An scGPT preset under the geneformer backend is rejected by the cross-check.
    with pytest.raises(ValueError, match="Unknown geneformer preset"):
        Config.from_yaml(
            _write(tmp_path, """
            input: { h5ad_path: data/x.h5ad }
            model: { backend: geneformer, preset: scgpt-human }
            """)
        )


def test_geneformer_preset_rejected_on_scgpt_backend(tmp_path):
    with pytest.raises(ValueError, match="Unknown scgpt preset"):
        Config.from_yaml(
            _write(tmp_path, """
            input: { h5ad_path: data/x.h5ad }
            model: { backend: scgpt, preset: V2-104M }
            """)
        )


def test_unknown_backend_rejected(tmp_path):
    with pytest.raises(ValueError):
        Config.from_yaml(
            _write(tmp_path, """
            input: { h5ad_path: data/x.h5ad }
            model: { backend: scbert, preset: whatever }
            """)
        )


def test_geneformer_properties_raise_for_scgpt_preset(tmp_path):
    # version/input_size/model_path are Geneformer-only; accessing them for an
    # scGPT preset raises a clear error rather than a bare KeyError.
    cfg = Config.from_yaml(
        _write(tmp_path, """
        input: { h5ad_path: data/x.h5ad }
        model: { backend: scgpt, preset: scgpt-human }
        """)
    )
    with pytest.raises(ValueError, match="not a Geneformer preset"):
        _ = cfg.model.input_size


def test_checkpoint_dir_override_expands(tmp_path):
    cfg = Config.from_yaml(
        _write(tmp_path, """
        input: { h5ad_path: data/x.h5ad }
        model: { backend: scgpt, preset: scgpt-human, checkpoint_dir: ~/ckpt }
        """)
    )
    assert cfg.model.checkpoint_dir is not None
    assert not str(cfg.model.checkpoint_dir).startswith("~")


def test_existing_geneformer_config_still_loads(tmp_path):
    # Back-compat: a pre-backend config (no model.backend, quantize in training)
    # loads unchanged.
    cfg = Config.from_yaml(
        _write(tmp_path, """
        name: legacy
        input: { h5ad_path: data/pbmc3k.loom }
        model: { preset: V1-10M, geneformer_repo: vendor/Geneformer }
        training: { epochs: 10, quantize: false, num_crossval_splits: 1 }
        """)
    )
    assert cfg.model.backend == "geneformer"
    assert cfg.model.version == "V1"
    assert cfg.training.quantize is False
