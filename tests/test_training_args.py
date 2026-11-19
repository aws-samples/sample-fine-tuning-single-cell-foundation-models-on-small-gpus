# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Training arguments handed to Geneformer's Classifier."""

from __future__ import annotations

from scfm_ft.config import Config
from scfm_ft.finetune import _build_training_args


def _cfg(**training) -> Config:
    return Config.model_validate(
        {"name": "t", "input": {"h5ad_path": "data/x.h5ad"}, "training": training}
    )


def test_final_epoch_model_is_kept():
    # Geneformer defaults load_best_model_at_end=True (selected by eval loss), and
    # merges our training_args over its defaults, so this override wins.
    assert _build_training_args(_cfg())["load_best_model_at_end"] is False


def test_final_epoch_kept_on_the_quantized_path_too():
    args = _build_training_args(_cfg(quantize=True))
    assert args["load_best_model_at_end"] is False
    assert args["bf16"] is True


def test_no_best_metric_is_configured():
    # With selection off, a metric_for_best_model would be dead config; keep it absent.
    assert "metric_for_best_model" not in _build_training_args(_cfg())
