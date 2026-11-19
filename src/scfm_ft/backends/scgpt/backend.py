# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""scGPT backend.

scGPT (`bowang-lab/scGPT <https://github.com/bowang-lab/scGPT>`_) is a transformer
foundation model for single-cell data. It differs from Geneformer in every place the
:class:`ModelBackend` interface isolates:

* **Tokenization** — a **gene vocabulary + value binning** (expression binned into
  ``n_input_bins`` bins), not Geneformer's rank-value encoding. ``tokenize`` wraps
  scGPT's ``GeneVocab`` + ``tokenize_and_pad_batch`` over the SAME ``AnnData`` the
  shared loaders in ``datasets.py`` already produce.
* **Fine-tune** — scGPT ships its own ``TransformerModel`` with a cell-type
  classification head (and optional GEPC/MVC auxiliary objectives). ``finetune``
  drives that loop, honoring the shared ``TrainingConfig`` (epochs/lr/batch/splits/
  seed) and its own train/valid/test split from the SAME ``split_sizes`` so
  cross-model numbers are comparable.
* **Checkpoints** — a pretrained ``.pt`` + ``args.json`` + ``vocab.json`` (e.g.
  *scGPT_human*). ``resolve_checkpoint`` points at the baked-in weights;
  ``supported_presets`` exposes ``scgpt-human`` / ``scgpt-blood`` / ``scgpt-cp``.
* **Quantization** — not wired here; ``validate_config`` raises a clear error if
  ``quantize`` is set, mirroring Geneformer's guard but owned by this backend.

All scGPT / torch imports are **lazy** (inside the stage methods) so that the
lightweight CLI subcommands and the config/conformance tests import this backend
without the CUDA + ``flash-attn`` stack installed. The methods that need a GPU
(``tokenize``/``finetune``/``evaluate``) are exercised by the on-SageMaker smoke
run, as Geneformer's real path is today; the offline tests cover discovery,
validation and dispatch wiring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from scfm_ft.backends.base import (
    METRIC_ACCURACY,
    METRIC_CONF_MATRIX,
    METRIC_MACRO_F1,
    FinetuneResult,
    TokenizeResult,
)

if TYPE_CHECKING:
    from scfm_ft.config import Config

# Pretrained scGPT checkpoints. The directory name is the subdirectory within the
# scGPT model root (baked into the image / mounted channel) that holds the
# {model.pt, args.json, vocab.json} triple.
SCGPT_PRESETS: dict[str, dict[str, object]] = {
    "scgpt-human": {"model_dir": "scGPT_human", "embsize": 512},
    "scgpt-blood": {"model_dir": "scGPT_bc", "embsize": 512},
    "scgpt-cp": {"model_dir": "scGPT_CP", "embsize": 512},
}

# scGPT fine-tune is single-GPU-friendly for these cell-type tasks; a larger
# instance only helps very large atlases.
_INSTANCE_BY_PRESET = {
    "scgpt-human": "ml.g5.2xlarge",
    "scgpt-blood": "ml.g5.2xlarge",
    "scgpt-cp": "ml.g5.2xlarge",
}

# Special tokens scGPT's vocabulary expects.
_PAD_TOKEN = "<pad>"  # nosec B105 -- tokenizer padding token, not a credential
_SPECIAL_TOKENS = [_PAD_TOKEN, "<cls>", "<eoc>"]


class ScgptBackend:
    """``ModelBackend`` for scGPT (gene-vocab + value-binning, own transformer loop)."""

    name = "scgpt"

    # --- discovery / validation -------------------------------------------------
    def supported_presets(self) -> list[str]:
        return list(SCGPT_PRESETS)

    def resolve_checkpoint(self, cfg: Config) -> Path:
        if cfg.model.checkpoint_dir is not None:
            return cfg.model.checkpoint_dir
        model_dir = str(SCGPT_PRESETS[cfg.model.preset]["model_dir"])
        return cfg.scgpt.model_root / model_dir

    def validate_config(self, cfg: Config) -> None:
        if cfg.model.preset not in self.supported_presets():
            raise ValueError(
                f"Unknown scgpt preset {cfg.model.preset!r}; "
                f"choose from {self.supported_presets()}."
            )
        if cfg.training.quantize:
            raise ValueError(
                "quantize (QLoRA) is not supported for the scgpt backend. "
                "scGPT is not wired for 4-bit fine-tuning here; drop quantize."
            )
        if cfg.classifier.kind != "cell":
            raise ValueError(
                f"scgpt backend supports cell-type classification only "
                f"(classifier.kind='cell'), got {cfg.classifier.kind!r}."
            )

    # --- pipeline stages --------------------------------------------------------
    def tokenize(self, cfg: Config, input_path: Path) -> TokenizeResult:
        """Value-bin + gene-vocab tokenize the input AnnData/loom into a dataset on disk.

        Reads the same raw-count matrix the shared loaders produce, maps genes through
        scGPT's ``GeneVocab`` (dropping genes absent from the pretrained vocab), bins
        each cell's nonzero expression into ``cfg.scgpt.n_input_bins`` bins, and saves
        the padded token/value tensors as an ``.npz`` the fine-tune stage reloads.
        """
        import numpy as np

        from scfm_ft.backends.scgpt._io import load_counts_matrix
        from scfm_ft.input_checks import check_input_file

        # Header-only checks (format, size, cell and gene caps) before the matrix is loaded.
        check_input_file(input_path, cfg.input.limits)

        ckpt = self.resolve_checkpoint(cfg)
        vocab = _load_gene_vocab(ckpt)

        cl = cfg.classifier
        genes, counts, labels, groups = load_counts_matrix(input_path, cl.label_column, cl.group_column)

        # Keep only genes present in the pretrained vocabulary (scGPT drops OOV genes).
        in_vocab = np.array([g in vocab for g in genes])
        if not in_vocab.any():
            raise RuntimeError(
                f"None of the {len(genes)} input genes are in the scGPT vocab at {ckpt}. "
                "Check that gene identifiers match scGPT's vocabulary (gene symbols)."
            )
        genes = genes[in_vocab]
        counts = counts[:, in_vocab]
        gene_ids = np.array([vocab[g] for g in genes], dtype=np.int64)

        from scgpt.preprocess import binning  # lazy: CUDA/scgpt only present in image

        binned = np.vstack(
            [binning(row, n_bins=cfg.scgpt.n_input_bins) for row in counts]
        ).astype(np.int64)

        out_dir = cfg.tokenized_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = out_dir / f"{cfg.name}_scgpt.npz"
        arrays = {
            "gene_ids": gene_ids,
            "values": binned,
            # Unicode, not object, dtype so the file loads with allow_pickle=False.
            "labels": np.asarray(labels).astype(str),
            "pad_token_id": np.int64(vocab[_PAD_TOKEN]),
        }
        if groups is not None:
            arrays["groups"] = np.asarray(groups).astype(str)
        np.savez(dataset_path, **arrays)
        return TokenizeResult(dataset_path=dataset_path)

    def finetune(self, cfg: Config, dataset_path: Path) -> FinetuneResult:
        """Fine-tune scGPT's cell-type classifier, honoring the shared split + hp config.

        Drives scGPT's ``TransformerModel`` training loop. Splits the tokenized dataset
        into train/valid/test using ``cfg.training.split_sizes`` (same discipline as
        Geneformer), trains for ``cfg.training.epochs``, evaluates on the held-out test
        split, and writes the standardized metrics + confusion matrix so the evaluate
        stage renders a backend-agnostic report.
        """
        import numpy as np

        from scfm_ft.backends.scgpt._train import train_scgpt_classifier

        ckpt = self.resolve_checkpoint(cfg)
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"scGPT checkpoint not found at {ckpt}. Bake the pretrained weights "
                f"(model.pt/args.json/vocab.json) into the image or mount them, or set "
                f"model.checkpoint_dir."
            )

        # allow_pickle=False (the default): the .npz holds only numeric and unicode
        # arrays, so loading it can never unpickle objects.
        data = np.load(dataset_path, allow_pickle=False)
        out_dir = cfg.model_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics, conf_matrix, label_map = train_scgpt_classifier(
            checkpoint_dir=Path(ckpt),
            gene_ids=data["gene_ids"],
            values=data["values"],
            labels=data["labels"],
            pad_token_id=int(data["pad_token_id"]),
            cfg=cfg,
            output_dir=out_dir,
            groups=_groups_for(cfg, data),
        )

        prefix = cfg.name
        # Standardized metrics JSON (scalars only; conf matrix persisted separately).
        serializable = {k: v for k, v in metrics.items() if k != METRIC_CONF_MATRIX}
        (out_dir / f"{prefix}_metrics.json").write_text(
            json.dumps(serializable, indent=2, default=str)
        )
        label_map_path = out_dir / f"{prefix}_label_map.json"
        label_map_path.write_text(json.dumps(label_map, indent=2))

        artifacts: dict[str, Path] = {"label_map": label_map_path}
        if conf_matrix is not None:
            from scfm_ft.conf_matrix import conf_matrix_path, save_conf_matrix

            artifacts["conf_matrix"] = save_conf_matrix(
                conf_matrix, conf_matrix_path(out_dir, prefix)
            )

        return FinetuneResult(
            model_dir=out_dir,
            metrics=serializable,
            label_map=label_map_path,
            artifacts=artifacts,
        )

    def evaluate(self, cfg: Config) -> dict:
        """Render the held-out evaluation report from the fine-tune outputs.

        Like Geneformer's evaluate stage, this does NOT re-run inference (the
        fine-tune already evaluated on the held-out test split); it loads the
        persisted metrics + confusion matrix and writes the standardized report.
        """
        from scfm_ft.conf_matrix import conf_matrix_path, load_conf_matrix

        model_out = cfg.model_out_dir
        prefix = cfg.name
        metrics_file = model_out / f"{prefix}_metrics.json"
        conf_mat_file = conf_matrix_path(model_out, prefix)

        if not metrics_file.exists():
            raise FileNotFoundError(
                f"No fine-tune metrics at {metrics_file}. Run `finetune` first "
                f"(num_crossval_splits >= 1 for a held-out eval)."
            )

        eval_dir = cfg.eval_dir
        eval_dir.mkdir(parents=True, exist_ok=True)
        metrics = json.loads(metrics_file.read_text())
        (eval_dir / f"{prefix}_eval_metrics.json").write_text(json.dumps(metrics, indent=2))

        if conf_mat_file.exists():
            _plot_conf_mat(load_conf_matrix(conf_mat_file), eval_dir, prefix)

        return {
            **metrics,
            METRIC_ACCURACY: metrics.get(METRIC_ACCURACY),
            METRIC_MACRO_F1: metrics.get(METRIC_MACRO_F1),
        }

    # --- infra hints ------------------------------------------------------------
    def default_instance_type(self, preset: str) -> str:
        return _INSTANCE_BY_PRESET.get(preset, "ml.g5.2xlarge")

    def docker_layers(self) -> list[str]:
        return [
            "CUDA base + torch matching flash-attn",
            "pip install scgpt flash-attn (build-heavy, CUDA-only)",
            "bake scGPT pretrained weights (model.pt/args.json/vocab.json) into image",
            "pip install the scfm_ft package (shared pipeline)",
        ]


# ---------------------------------------------------------------------------
# Small helpers kept module-local so the heavy scGPT internals stay lazy.
# ---------------------------------------------------------------------------
def _groups_for(cfg: Config, data):
    """Per-cell group ids from the tokenized ``.npz``, or ``None`` without a group column."""
    if not cfg.classifier.group_column:
        return None
    if "groups" not in data.files:
        raise KeyError(
            f"classifier.group_column is {cfg.classifier.group_column!r} but the tokenized "
            "dataset has no groups; re-tokenize with the group column set."
        )
    return data["groups"]


def _load_gene_vocab(checkpoint_dir: Path):
    """Load scGPT's ``GeneVocab`` from ``vocab.json`` in the checkpoint dir."""
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    vocab_file = Path(checkpoint_dir) / "vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"scGPT vocab.json not found at {vocab_file}.")
    vocab = GeneVocab.from_file(vocab_file)
    for tok in _SPECIAL_TOKENS:
        if tok not in vocab:
            vocab.append_token(tok)
    return vocab


def _plot_conf_mat(conf_matrix, eval_dir: Path, prefix: str) -> None:
    """Render a confusion-matrix heatmap matching the Geneformer report figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="g", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"scGPT — {prefix}")
    fig.tight_layout()
    fig.savefig(eval_dir / f"{prefix}_conf_matrix.png", dpi=150)
    plt.close(fig)
