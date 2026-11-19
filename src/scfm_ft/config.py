# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Typed configuration schema for Geneformer fine-tuning runs.

A single YAML file drives all stages (tokenize / finetune / evaluate). The schema
is validated with pydantic so misconfigurations fail fast with a clear message
rather than deep inside the training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from scfm_ft.input_checks import (
    DEFAULT_MAX_CELLS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_GENES,
    GIB,
    InputLimits,
)

# Pretrained base model per Geneformer version. The directory name is the
# subdirectory within the cloned Geneformer repo that holds the weights.
# input_size is fixed by the pretraining of each model and must match the tokenizer.
MODEL_PRESETS: dict[str, dict[str, object]] = {
    "V1-10M": {"version": "V1", "model_dir": "Geneformer-V1-10M", "input_size": 2048},
    "V2-104M": {"version": "V2", "model_dir": "Geneformer-V2-104M", "input_size": 4096},
    "V2-316M": {"version": "V2", "model_dir": "Geneformer-V2-316M", "input_size": 4096},
}

# scGPT preset names, kept here (not imported from the backend) so config validation
# has no import-time dependency on the ML stack. The backend owns the full preset
# metadata (SCGPT_PRESETS in backends/scgpt/backend.py); these names must match it.
SCGPT_PRESET_NAMES: tuple[str, ...] = ("scgpt-human", "scgpt-blood", "scgpt-cp")


class InputConfig(BaseModel):
    """Where the labeled single-cell data comes from."""

    h5ad_path: Path = Field(
        ..., description="AnnData .h5ad with ensembl_id, n_counts, and label column(s)."
    )
    # Safety caps checked from the file header before any reader loads the matrix (see
    # input_checks.py). Generous defaults; raise them for a larger dataset.
    max_file_gib: float = Field(
        default=DEFAULT_MAX_FILE_BYTES / GIB, gt=0, description="Largest input file, in GiB."
    )
    max_cells: int = Field(default=DEFAULT_MAX_CELLS, gt=0, description="Most cells accepted.")
    max_genes: int = Field(default=DEFAULT_MAX_GENES, gt=0, description="Most genes accepted.")

    @field_validator("h5ad_path")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @property
    def limits(self) -> InputLimits:
        return InputLimits(
            max_file_bytes=int(self.max_file_gib * GIB),
            max_cells=self.max_cells,
            max_genes=self.max_genes,
        )


class ModelConfig(BaseModel):
    """Which pretrained model (and which backend) to fine-tune.

    ``backend`` selects the :class:`~scfm_ft.backends.base.ModelBackend`;
    ``preset`` is validated against that backend's ``supported_presets()`` (see the
    top-level :class:`Config` validator) so an scGPT typo cannot masquerade as a
    Geneformer preset and vice versa.
    """

    backend: Literal["geneformer", "scgpt"] = "geneformer"
    preset: str = "V2-104M"
    # Optional explicit checkpoint location, overriding a backend's auto-resolution.
    checkpoint_dir: Path | None = None
    # Root of the cloned Geneformer repo (contains the per-model subdirectories).
    # Only meaningful for the geneformer backend.
    geneformer_repo: Path = Field(default=Path("vendor/Geneformer"))

    @field_validator("geneformer_repo")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("checkpoint_dir")
    @classmethod
    def _expand_ckpt(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v is not None else None

    # --- Geneformer-only derived properties -------------------------------------
    # These are valid only for the geneformer backend (its presets live in
    # MODEL_PRESETS). Accessing them for another backend's preset raises a clear
    # error rather than a bare KeyError.
    def _require_geneformer_preset(self) -> dict:
        try:
            return MODEL_PRESETS[self.preset]
        except KeyError:
            raise ValueError(
                f"{self.preset!r} is not a Geneformer preset; "
                f"version/input_size/model_path apply to the geneformer backend only."
            )

    @property
    def version(self) -> str:
        return str(self._require_geneformer_preset()["version"])

    @property
    def input_size(self) -> int:
        return int(self._require_geneformer_preset()["input_size"])

    @property
    def model_path(self) -> Path:
        return self.geneformer_repo / str(self._require_geneformer_preset()["model_dir"])


class GeneformerParams(BaseModel):
    """Geneformer-specific knobs (kept out of the shared config)."""

    # Escape hatch to run the experimental QLoRA (4-bit) path, which has not been
    # re-validated on a GPU since the move to transformers 5 (see sm_entry's guard).
    # Off by default.
    allow_quantize: bool = False


class ScgptParams(BaseModel):
    """scGPT-specific knobs.

    scGPT uses value binning (expression → ``n_input_bins`` bins) over a pretrained
    gene vocabulary, and its own transformer loop. ``model_root`` is where the
    per-preset checkpoint subdirectories (``scGPT_human`` …) live when
    ``model.checkpoint_dir`` is not set.
    """

    model_root: Path = Field(default=Path("/scgpt"))
    n_input_bins: int = 51
    # FlashAttention: use scGPT's fast (flash-attn) transformer kernels instead of
    # the vanilla PyTorch attention. Big speed + memory win on CUDA, but requires the
    # flash-attn package (CUDA-only, baked into the scgpt-ft image). When requested
    # but unavailable (no CUDA, flash-attn not importable), the training loop falls
    # back to vanilla attention and logs why, rather than crashing.
    use_flash_attn: bool = True
    # Activation (gradient) checkpointing: recompute transformer-block activations in
    # the backward pass instead of storing them, trading ~20-30% extra compute for a
    # large activation-memory saving. Lets a bigger batch / larger model fit on the
    # same GPU. Off by default (fastest); turn on when you are memory-bound.
    grad_checkpointing: bool = False
    # Maximum gene-token sequence length fed to the transformer. scGPT keeps every
    # in-vocab gene as a token; with no cap, a dataset retaining ~2000 genes makes the
    # vanilla-attention score matrix (O(seq^2)) explode — on a 16GB T4 that is a
    # multi-hundred-GiB allocation and an instant CUDA OOM. Capping to the top
    # ``max_seq_len`` highest-variance genes (the standard scGPT cell-type HVG count)
    # bounds the forward so it fits. 0 disables the cap (keep all genes).
    max_seq_len: int = 1200
    # Class weighting for the classification loss, to help rare cell types.
    #   "none"     — plain cross-entropy (default; reproduces earlier results).
    #   "balanced" — weight each class by n / (k * count), as scikit-learn does.
    #   "sqrt"     — square root of the balanced weights: a gentler correction.
    # Weights come from each fold's TRAINING cells only, never the held-out cells.
    class_weighting: Literal["none", "balanced", "sqrt"] = "none"
    freeze_backbone: bool = False
    # Optional masked-value / cell-embedding auxiliary objective during fine-tune.
    do_mvc: bool = False

    @field_validator("model_root")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()


class ClassifierConfig(BaseModel):
    """Classification task definition."""

    kind: Literal["cell", "gene"] = "cell"
    # Column in adata.obs holding the class label (cell classifier).
    label_column: str = "cell_type"
    # Optional column in adata.obs naming the group each cell comes from, usually the
    # donor. When set, train/valid/test (and CV folds) are split by group, so no donor
    # has cells in both training and held-out data. See scfm_ft.splits.
    group_column: str | None = None
    # Optional: restrict to a subset of label values; None/empty means use all.
    states: list[str] | None = None
    # Drop classes rarer than this fraction of cells (0 disables).
    rare_threshold: float = 0.0
    # Cap total cells / cells per class for faster experimentation (None = all).
    max_ncells: int | None = None
    max_ncells_per_class: int | None = None


class TrainingConfig(BaseModel):
    """Fine-tuning hyperparameters (passed to HF TrainingArguments)."""

    epochs: int = 10
    learning_rate: float = 5e-5
    batch_size: int = 12
    freeze_layers: int = 0
    # Membership-inference audit after training (scfm_ft.privacy): compare the model's
    # loss on training cells with held-out cells. Needs held-out cells
    # (num_crossval_splits >= 1); the Geneformer backend also needs classifier.group_column.
    privacy_audit: bool = False
    # Cells scored on each side (members, non-members); 0 = all.
    privacy_audit_max_cells: int = 20000
    # QLoRA: load the base model in 4-bit (bitsandbytes nf4) and fine-tune LoRA
    # adapters instead of the full weights. Big memory saving; lets larger models
    # or batches fit on a smaller GPU. False = full-precision full fine-tune.
    quantize: bool = False
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 500
    weight_decay: float = 0.001
    # 0 = train on all data; 1 = single train/eval split; 5 = 5-fold CV.
    num_crossval_splits: Literal[0, 1, 5] = 1
    split_sizes: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "valid": 0.1, "test": 0.1}
    )
    seed: int = 42

    @model_validator(mode="after")
    def _check_splits(self) -> TrainingConfig:
        if self.num_crossval_splits != 0:
            total = sum(self.split_sizes.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"split_sizes must sum to 1.0, got {total}")
        return self


class RuntimeConfig(BaseModel):
    """Hardware / process settings."""

    nproc: int = 4
    ngpu: int = 1
    forward_batch_size: int = 100


class Config(BaseModel):
    """Top-level run configuration."""

    name: str = "scfm-ft-run"
    output_dir: Path = Field(default=Path("outputs"))
    input: InputConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    # Per-backend sub-config blocks. Each backend reads only its own block; a field
    # in the wrong block is simply ignored by the other backend, and preset/backend
    # compatibility is enforced by _check_backend_preset below.
    geneformer: GeneformerParams = Field(default_factory=GeneformerParams)
    scgpt: ScgptParams = Field(default_factory=ScgptParams)

    @field_validator("output_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @model_validator(mode="after")
    def _check_backend_preset(self) -> Config:
        """Reject a preset that does not belong to the selected backend.

        Kept inline (rather than importing the backend registry) so config has no
        import-time dependency on the ML backends. The preset→backend namespacing
        makes cross-backend collisions impossible.
        """
        if self.model.backend == "geneformer":
            if self.model.preset not in MODEL_PRESETS:
                raise ValueError(
                    f"Unknown geneformer preset {self.model.preset!r}; "
                    f"choose from {sorted(MODEL_PRESETS)}."
                )
        elif self.model.backend == "scgpt":
            if self.model.preset not in SCGPT_PRESET_NAMES:
                raise ValueError(
                    f"Unknown scgpt preset {self.model.preset!r}; "
                    f"choose from {sorted(SCGPT_PRESET_NAMES)}."
                )
        return self

    @property
    def tokenized_dir(self) -> Path:
        return self.output_dir / "tokenized"

    @property
    def model_out_dir(self) -> Path:
        return self.output_dir / "model"

    @property
    def eval_dir(self) -> Path:
        return self.output_dir / "eval"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
