# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Pretrained-weight loading for the scGPT backend (real torch modules, no scGPT/GPU).

The published scGPT checkpoints store each encoder layer's attention input projection
as ``self_attn.Wqkv.*`` (FlashAttention naming). Standard PyTorch attention calls the
same tensor ``self_attn.in_proj_*``, so the loader must rename it on that path.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from scfm_ft.backends.scgpt._train import _load_pretrained_weights

D, HEADS, LAYERS, CLASSES = 16, 2, 2, 3


class _StandardAttentionModel(nn.Module):
    """Mirrors scGPT's layout on the standard path: transformer_encoder + fresh head."""

    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(D, HEADS, dim_feedforward=32, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.cls_decoder = nn.Linear(D, CLASSES)


class _FlashLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.Wqkv = nn.Linear(D, 3 * D)
        self.self_attn.out_proj = nn.Linear(D, D)


class _FlashAttentionModel(nn.Module):
    """Mirrors scGPT's layout on the FlashAttention path (Wqkv naming)."""

    def __init__(self):
        super().__init__()
        self.transformer_encoder = nn.Module()
        self.transformer_encoder.layers = nn.ModuleList(_FlashLayer() for _ in range(LAYERS))
        self.cls_decoder = nn.Linear(D, CLASSES)


def _flash_style_checkpoint(model: nn.Module) -> dict:
    """Turn a standard-path state dict into how the published checkpoint names it."""
    state = {}
    for k, v in model.state_dict().items():
        if k.startswith("cls_decoder"):
            continue  # pretrained checkpoints have no task head
        state[k.replace("in_proj_", "Wqkv.")] = torch.randn_like(v)
    return state


def test_standard_path_renames_wqkv_and_loads_attention():
    model = _StandardAttentionModel()
    ckpt = _flash_style_checkpoint(_StandardAttentionModel())
    assert any("Wqkv.weight" in k for k in ckpt)  # the checkpoint really uses Wqkv names

    summary = _load_pretrained_weights(model, ckpt, use_fast=False)

    loaded = model.state_dict()
    for i in range(LAYERS):
        w = f"transformer_encoder.layers.{i}.self_attn.in_proj_weight"
        b = f"transformer_encoder.layers.{i}.self_attn.in_proj_bias"
        assert torch.equal(loaded[w], ckpt[w.replace("in_proj_", "Wqkv.")])
        assert torch.equal(loaded[b], ckpt[b.replace("in_proj_", "Wqkv.")])
    # Only the fresh head stays at init.
    assert all(k.startswith("cls_decoder") for k in summary["missing"])
    assert summary["unused"] == []


def test_flash_path_loads_wqkv_names_as_is():
    model = _FlashAttentionModel()
    ckpt = {
        k: torch.randn_like(v)
        for k, v in _FlashAttentionModel().state_dict().items()
        if not k.startswith("cls_decoder")
    }
    summary = _load_pretrained_weights(model, ckpt, use_fast=True)
    w = "transformer_encoder.layers.0.self_attn.Wqkv.weight"
    assert torch.equal(model.state_dict()[w], ckpt[w])
    assert all(k.startswith("cls_decoder") for k in summary["missing"])


def test_missing_encoder_weights_raise_instead_of_training_from_random():
    """The original bug: Wqkv names loaded into a standard-attention model, unrenamed."""
    model = _StandardAttentionModel()
    ckpt = _flash_style_checkpoint(_StandardAttentionModel())
    with pytest.raises(RuntimeError, match="transformer_encoder parameters did not load"):
        # use_fast=True skips the rename, reproducing the old silent mismatch.
        _load_pretrained_weights(model, ckpt, use_fast=True)


def test_shape_mismatch_is_not_loaded_and_is_reported():
    model = _StandardAttentionModel()
    ckpt = _flash_style_checkpoint(_StandardAttentionModel())
    ckpt["transformer_encoder.layers.0.linear1.weight"] = torch.randn(7, 7)
    with pytest.raises(RuntimeError, match="linear1.weight"):
        _load_pretrained_weights(model, ckpt, use_fast=False)
