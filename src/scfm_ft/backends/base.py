# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The model-backend seam.

Every foundation model this repo can fine-tune (Geneformer today, scGPT next,
scBERT/UCE later) implements the :class:`ModelBackend` protocol. The shared layers
— the CLI, the SageMaker entry point, the config — talk ONLY to this interface and
never import ``geneformer`` or ``scgpt`` directly. Adding a model therefore means
writing one backend module and registering it (see ``backends/__init__.py``); no
shared-layer edits.

The pipeline is three stages — **tokenize → finetune → evaluate** — and the
interface mirrors that, plus two infra hints the SageMaker layer consumes instead
of hard-coding them.

``runtime_checkable`` lets the conformance test assert each registered backend
satisfies the protocol offline, in milliseconds, so a backend that drops a method
fails in CI rather than on a GPU three minutes into a job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from scfm_ft.config import Config

# ---------------------------------------------------------------------------
# Standardized held-out metric keys.
#
# Both backends MUST emit these keys so the evaluate stage and any cross-model
# A/B comparison stay backend-agnostic. Defining them once here is the single
# source of truth; a backend that invents its own key name for accuracy breaks
# the comparison silently, so new backends map their metrics onto these.
# ---------------------------------------------------------------------------
METRIC_ACCURACY = "accuracy"
METRIC_MACRO_F1 = "macro_f1"
METRIC_CONF_MATRIX = "conf_matrix"

#: Scalar metrics every backend reports (``conf_matrix`` is held separately
#: because it is a non-JSON-serializable structure, not a scalar).
REQUIRED_METRIC_KEYS: tuple[str, ...] = (METRIC_ACCURACY, METRIC_MACRO_F1)


@dataclass
class TokenizeResult:
    """Outcome of the tokenize stage.

    ``dataset_path`` is a model-ready tokenized dataset on disk — a Geneformer
    ``.dataset`` directory, an scGPT binned-token dataset, etc. The shared layer
    only ever passes this path back into ``finetune``; it never inspects the format.
    """

    dataset_path: Path


@dataclass
class FinetuneResult:
    """Outcome of the fine-tune stage — model-neutral.

    Supersedes the Geneformer-specific ``finetune.FinetuneResult``. Shared code reads
    only ``model_dir`` and ``metrics``; ``artifacts`` carries backend-specific extra
    files (confusion-matrix JSON, label map, scGPT vocab) by name so the evaluate
    stage can find them without knowing the backend.
    """

    model_dir: Path
    metrics: dict | None = None
    label_map: Path | None = None
    artifacts: dict[str, Path] | None = None


@runtime_checkable
class ModelBackend(Protocol):
    """Interface every fine-tuning backend implements.

    Each method takes the typed :class:`~scfm_ft.config.Config` so a backend
    reads only the fields it owns (its own sub-config block plus the shared
    training/classifier/runtime/input sections). No hidden globals; every stage
    returns a path or a dataclass.
    """

    #: Registry key / human name, e.g. ``"geneformer"`` or ``"scgpt"``.
    name: str

    # --- discovery / validation -------------------------------------------------
    def supported_presets(self) -> list[str]:
        """Preset names this backend accepts (e.g. Geneformer ``V2-104M``)."""
        ...

    def resolve_checkpoint(self, cfg: Config) -> Path:
        """Filesystem location of the pretrained weights for ``cfg``'s preset."""
        ...

    def validate_config(self, cfg: Config) -> None:
        """Fail fast on an unsupported combination (e.g. scGPT + ``quantize``).

        Raises a clear error; returns ``None`` when the config is runnable.
        """
        ...

    # --- pipeline stages --------------------------------------------------------
    def tokenize(self, cfg: Config, input_path: Path) -> TokenizeResult:
        """Convert the raw ``AnnData``/``loom`` at ``input_path`` into a tokenized
        dataset on disk, returning its path."""
        ...

    def finetune(self, cfg: Config, dataset_path: Path) -> FinetuneResult:
        """Fine-tune the pretrained model on the tokenized dataset."""
        ...

    def evaluate(self, cfg: Config) -> dict:
        """Render the held-out evaluation report from the fine-tune outputs,
        returning the standardized metrics dict (see ``REQUIRED_METRIC_KEYS``)."""
        ...

    # --- infra hints (consumed by the SageMaker layer) --------------------------
    def default_instance_type(self, preset: str) -> str:
        """Recommended SageMaker instance type for ``preset`` (a hint, overridable)."""
        ...

    def docker_layers(self) -> list[str]:
        """Human-readable description of how the image installs this backend —
        documents the per-backend Docker layer so the recipe is derived from the
        backend, not duplicated in the Dockerfile."""
        ...
