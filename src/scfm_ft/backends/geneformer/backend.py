# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Geneformer backend.

This wraps today's Geneformer pipeline behind the :class:`ModelBackend` interface
with **zero behavior change** — ``tokenize``/``finetune``/``evaluate`` delegate to
the existing ``scfm_ft.{tokenize,finetune,evaluate}`` functions, which still
hold the real Geneformer-specific logic (rank-value encoding, the QLoRA guard, the
datestamped-dir confusion-matrix plotting).

Keeping the implementation in those modules means the 17 existing tests keep passing
unchanged; this file is only the adapter that lets the shared layer dispatch to them
through the registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from scfm_ft.backends.base import (
    METRIC_ACCURACY,
    METRIC_MACRO_F1,
    FinetuneResult,
    TokenizeResult,
)

if TYPE_CHECKING:
    from scfm_ft.config import Config

# Instance-type hints keyed by preset (see docs/SAGEMAKER.md). Full-precision
# fine-tune: V2-104M fits ml.g5.xlarge (A10G); V2-316M needs ml.p4d.24xlarge.
_INSTANCE_BY_PRESET = {
    "V1-10M": "ml.g5.xlarge",
    "V2-104M": "ml.g5.xlarge",
    "V2-316M": "ml.p4d.24xlarge",
}


class GeneformerBackend:
    """``ModelBackend`` for Geneformer (rank-value encoding + ``geneformer.Classifier``)."""

    name = "geneformer"

    # --- discovery / validation -------------------------------------------------
    def supported_presets(self) -> list[str]:
        from scfm_ft.config import MODEL_PRESETS

        return list(MODEL_PRESETS)

    def resolve_checkpoint(self, cfg: Config) -> Path:
        return cfg.model.model_path

    def validate_config(self, cfg: Config) -> None:
        if cfg.model.preset not in self.supported_presets():
            raise ValueError(
                f"Unknown geneformer preset {cfg.model.preset!r}; "
                f"choose from {self.supported_presets()}."
            )
        # QLoRA (4-bit) is experimental: it was validated on transformers 4.46 with a
        # workaround that transformers 5 no longer needs, but has not been re-run on a
        # GPU since. The detailed guard lives in sm_entry (with its
        # GENEFORMER_FT_ALLOW_QUANTIZE escape hatch); this is the early check that
        # mirrors it.
        if cfg.training.quantize and not cfg.geneformer.allow_quantize:
            raise ValueError(
                "quantize (QLoRA) is experimental and has not been re-validated on "
                "transformers 5. Drop quantize, or set geneformer.allow_quantize=true "
                "to run it anyway."
            )

    # --- pipeline stages --------------------------------------------------------
    def tokenize(self, cfg: Config, input_path: Path) -> TokenizeResult:
        from scfm_ft.tokenize import tokenize as _tokenize

        # The existing tokenize() reads cfg.input.h5ad_path directly; input_path is
        # that same path (the shared layer passes cfg.input.h5ad_path). Kept in the
        # signature for interface symmetry with backends that take an explicit path.
        return TokenizeResult(dataset_path=_tokenize(cfg))

    def finetune(self, cfg: Config, dataset_path: Path) -> FinetuneResult:
        from scfm_ft.finetune import finetune as _finetune

        res = _finetune(cfg, dataset_path=dataset_path)
        artifacts: dict[str, Path] = {
            "labeled_dataset": res.labeled_dataset,
            "id_class_dict": res.id_class_dict,
        }
        if res.test_dataset is not None:
            artifacts["test_dataset"] = res.test_dataset
        return FinetuneResult(
            model_dir=res.model_dir,
            metrics=res.metrics,
            label_map=res.id_class_dict,
            artifacts=artifacts,
        )

    def evaluate(self, cfg: Config) -> dict:
        from scfm_ft.evaluate import evaluate as _evaluate

        metrics = _evaluate(cfg)
        # Geneformer's validate() already emits accuracy + macro_f1 under these keys;
        # surface them under the standardized names so cross-backend code is uniform.
        return {
            **metrics,
            METRIC_ACCURACY: metrics.get(METRIC_ACCURACY),
            METRIC_MACRO_F1: metrics.get(METRIC_MACRO_F1),
        }

    # --- infra hints ------------------------------------------------------------
    def default_instance_type(self, preset: str) -> str:
        return _INSTANCE_BY_PRESET.get(preset, "ml.g5.xlarge")

    def docker_layers(self) -> list[str]:
        return [
            "clone huggingface.co/ctheodoris/Geneformer + git-lfs pull weights to /Geneformer",
            "patch perturber_utils.py (QLoRA device_map)",
            "patch Geneformer for transformers 5 (geneformer_transformers5.py)",
            "pip install --no-deps /Geneformer  (deps come from uv.lock)",
        ]
