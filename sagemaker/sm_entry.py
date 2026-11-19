# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""SageMaker training entry point for scfm-ft.

SageMaker invokes this script inside the training container. It maps SageMaker's
environment (channels, hyperparameters, model dir) onto a scfm-ft ``Config``
and runs the fine-tune (optionally the tokenize stage first). The fine-tuned model
lands under ``SM_MODEL_DIR`` and SageMaker uploads it to S3 automatically.

Hyperparameters are passed by the launcher as ``--key value`` CLI args (SageMaker's
convention) and/or via the ``SM_HP_*`` environment. Data channels arrive as local
directories named by ``SM_CHANNEL_<NAME>``:

  train  : the input dataset — either a raw ``.loom``/``.h5ad`` (we tokenize it) or
           an already-tokenized ``.dataset`` directory (we skip tokenization).

Environment contract (set by SageMaker):
  SM_MODEL_DIR          where to write the final model (persisted to S3)
  SM_CHANNEL_TRAIN      local path to the 'train' input channel
  SM_NUM_GPUS           GPU count on the instance
  SM_HP_<NAME>          hyperparameter values
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from scfm_ft.backends import get_backend
from scfm_ft.config import Config
from scfm_ft.input_checks import check_dataset_dir
from scfm_ft.integrity import IntegrityError, parse_manifest_pins, verify_channel


def _parse_split_sizes(value: str) -> dict[str, float]:
    """'0.8,0.2,0' -> {'train': 0.8, 'valid': 0.2, 'test': 0.0}; must sum to 1."""
    parts = [float(x) for x in str(value).strip().strip('"').split(",")]
    if len(parts) != 3 or any(x < 0 for x in parts) or abs(sum(parts) - 1.0) > 1e-6:
        raise ValueError(f"--split-sizes must be three non-negative numbers summing to 1, got {value!r}")
    return dict(zip(("train", "valid", "test"), parts))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Core task
    p.add_argument("--name", default="sm_run")
    p.add_argument("--preset", default="V2-104M")
    p.add_argument("--label-column", default="cell_type")
    p.add_argument("--group-column", default="", help="obs column naming each cell's donor; empty = split by cell")
    p.add_argument("--classifier-kind", default="cell", choices=["cell", "gene"])
    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--freeze-layers", type=int, default=0)
    p.add_argument("--num-crossval-splits", type=int, default=1, choices=[0, 1, 5])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-sizes", default="0.8,0.1,0.1")
    # QLoRA (4-bit base + LoRA adapters). SageMaker passes hyperparameters as
    # strings, so accept a string and coerce to bool.
    p.add_argument("--quantize", default="false")
    # Model backend: which foundation model to fine-tune.
    p.add_argument("--backend", default="geneformer", choices=["geneformer", "scgpt"])
    # Data: name of the file within the train channel, or "" to use a tokenized dir.
    p.add_argument("--train-file", default="data.loom")
    # Where training data comes from: "channel" uses the SageMaker train channel;
    # "pbmc3k" self-fetches the public PBMC3k dataset in-container (no S3 upload
    # needed). Self-fetch is handy when staging the data to S3 is inconvenient.
    p.add_argument("--data-source", default="channel", choices=["channel", "pbmc3k"])
    # SageMaker-provided locations (fall back to env for local testing).
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument(
        "--train-channel",
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    p.add_argument("--geneformer-repo", default=os.environ.get("GENEFORMER_REPO", "/Geneformer"))
    p.add_argument("--scgpt-model-root", default=os.environ.get("SCGPT_MODEL_ROOT", "/scgpt"))
    # scGPT training knobs (string bools — SageMaker passes hyperparameters as strings).
    p.add_argument("--scgpt-flash-attn", default="true")
    p.add_argument("--scgpt-grad-checkpointing", default="false")
    p.add_argument("--scgpt-max-seq-len", type=int, default=1200)
    p.add_argument(
        "--scgpt-class-weighting", default="none", choices=["none", "balanced", "sqrt"]
    )
    # Input safety (see scfm_ft/integrity.py and scfm_ft/input_checks.py).
    # String bool: SageMaker passes hyperparameters as strings.
    p.add_argument("--require-input-hashes", default="false")
    p.add_argument("--privacy-audit", default="false", help="Membership-inference audit after training")
    # Expected SHA-256 of each channel's SHA256SUMS: "train=<hex>[,weights=<hex>]".
    p.add_argument("--manifest-sha256", default="")
    p.add_argument("--max-input-gib", type=float, default=4.0)
    p.add_argument("--max-cells", type=int, default=1_000_000)
    return p.parse_known_args()[0]


def _truthy(value: object) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def _verify_input_channels(args: argparse.Namespace) -> None:
    """Check each S3-backed input channel against its ``SHA256SUMS`` before it is read.

    The PBMC3k self-fetch verifies its own pinned hashes, so it has no channel to check.
    With ``--require-input-hashes`` a channel without a manifest, or with files the
    manifest does not list, stops the job here instead of training on unverified bytes.
    """
    require = _truthy(args.require_input_hashes)
    channels: list[tuple[str, Path]] = []
    if args.data_source == "channel":
        channels.append(("train", Path(args.train_channel)))
    weights = os.environ.get("SM_CHANNEL_WEIGHTS")
    if args.backend == "scgpt" and weights and Path(weights).is_dir():
        channels.append(("weights", Path(weights)))
    pins = parse_manifest_pins(getattr(args, "manifest_sha256", "") or "")
    unused = sorted(set(pins) - {name for name, _ in channels})
    if unused:
        raise IntegrityError(
            f"A manifest pin was given for the {', '.join(unused)} channel, which this job does not "
            "verify, so the pin would be ignored."
        )
    for name, directory in channels:
        print(f"[integrity] verifying the {name} channel ({directory})")
        verify_channel(directory, require=require, manifest_sha256=pins.get(name))


def _resolve_input(train_channel: Path, train_file: str) -> tuple[Path | None, Path | None]:
    """Return (raw_input, tokenized_dataset).

    If the channel holds a pre-tokenized ``.dataset`` directory, use it directly and
    skip tokenization. Otherwise locate the raw ``.loom``/``.h5ad`` to tokenize.
    """
    # Pre-tokenized dataset shipped directly?
    datasets = sorted(train_channel.glob("*.dataset"))
    if datasets:
        return None, datasets[0]

    # Named raw file, else first loom/h5ad in the channel.
    candidate = train_channel / train_file
    if candidate.exists():
        return candidate, None
    for pat in ("*.loom", "*.h5ad"):
        hits = sorted(train_channel.glob(pat))
        if hits:
            return hits[0], None
    raise FileNotFoundError(
        f"No .dataset, .loom, or .h5ad found in train channel {train_channel}."
    )


def _resolve_input_raw(train_channel: Path, train_file: str) -> Path:
    """Locate the raw ``.loom``/``.h5ad`` in the channel (scGPT has no pre-tokenized
    dataset form, so it always tokenizes from the raw matrix in-container)."""
    candidate = train_channel / train_file
    if candidate.exists():
        return candidate
    for pat in ("*.loom", "*.h5ad"):
        hits = sorted(train_channel.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No .loom or .h5ad found in train channel {train_channel} for the scgpt backend."
    )


def main() -> None:
    args = _parse_args()

    quantize = str(args.quantize).lower() in ("1", "true", "yes", "on")
    # The same env escape hatch that lifts the fast-fail guard below must also set
    # cfg.geneformer.allow_quantize, or the backend's own validate_config (which
    # checks that config field, not this env var) rejects the run before training.
    allow_quantize = os.environ.get("GENEFORMER_FT_ALLOW_QUANTIZE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if args.backend == "geneformer" and quantize and not allow_quantize:
        # QLoRA (4-bit base + LoRA) is EXPERIMENTAL for the classifier fine-tune path.
        # On transformers 4.46, adding a fresh classification head to a 4-bit base ran
        # _init_weights over every module and crashed on the bitsandbytes uint8/"Byte"
        # weights; finetune.py carried a workaround for that, and the path was validated
        # on V2-104M and V2-316M. transformers 5 fixed the cause upstream (it initialises
        # only keys missing from the checkpoint and skips every loaded tensor), so the
        # workaround is gone, but the 4-bit path has not been re-run on a GPU since the
        # upgrade. Fail fast with an explanation instead of spending a GPU job on it by
        # default.
        raise SystemExit(
            "ERROR: --quantize (QLoRA) is experimental in this Geneformer build.\n"
            "  It was validated on transformers 4.46 and has not been re-run on a GPU\n"
            "  since the move to transformers 5.\n"
            "  Use full-precision fine-tuning instead (drop --quantize): it fits\n"
            "  every supported preset -- V2-104M at batch 12 on ml.g5.xlarge,\n"
            "  V2-316M on ml.p4d.24xlarge.\n"
            "  To run it anyway, set GENEFORMER_FT_ALLOW_QUANTIZE=1."
        )

    train_channel = Path(args.train_channel)
    ngpu = int(os.environ.get("SM_NUM_GPUS", "1"))

    # Verify S3-backed inputs against their SHA256SUMS before anything reads them.
    _verify_input_channels(args)

    # A pre-tokenized Geneformer .dataset can be shipped directly; scGPT always
    # tokenizes from the raw matrix in-container (its tokenized form is an .npz).
    tokenized: Path | None = None
    if args.data_source == "pbmc3k":
        # Self-fetch the public PBMC3k dataset in-container (scanpy downloads it),
        # so no data needs to be uploaded to S3. Writes a Geneformer-ready loom.
        from scfm_ft.datasets import load_pbmc3k

        print("Self-fetching PBMC3k dataset (scanpy)...")
        # Private per-run directory (mode 0700, unpredictable name) rather than a fixed
        # /tmp path, so nothing can pre-create or symlink the file (Bandit B108).
        fetch_dir = Path(tempfile.mkdtemp(prefix="scfm-pbmc3k-"))
        raw_input = load_pbmc3k(out=fetch_dir / "pbmc3k.loom", label_column=args.label_column)
    elif args.backend == "geneformer":
        raw_input, tokenized = _resolve_input(train_channel, args.train_file)
    else:
        raw_input, tokenized = _resolve_input_raw(train_channel, args.train_file), None

    cfg = Config.model_validate(
        {
            "name": args.name,
            "output_dir": args.model_dir,
            # If pre-tokenized, h5ad_path is unused by finetune (we pass dataset_path);
            # point it at the channel anyway so the schema validates.
            "input": {
                "h5ad_path": str(raw_input or train_channel),
                "max_file_gib": args.max_input_gib,
                "max_cells": args.max_cells,
            },
            "model": {
                "backend": args.backend,
                "preset": args.preset,
                "geneformer_repo": args.geneformer_repo,
            },
            "geneformer": {"allow_quantize": allow_quantize},
            "classifier": {
                "kind": args.classifier_kind,
                "label_column": args.label_column,
                # SageMaker passes hyperparameters JSON-quoted; strip the quotes.
                "group_column": args.group_column.strip('"') or None,
            },
            "training": {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "freeze_layers": args.freeze_layers,
                "num_crossval_splits": args.num_crossval_splits,
                "quantize": quantize,
                "seed": args.seed,
                "split_sizes": _parse_split_sizes(args.split_sizes),
                "privacy_audit": _truthy(args.privacy_audit),
            },
            "scgpt": {
                "model_root": args.scgpt_model_root,
                "use_flash_attn": str(args.scgpt_flash_attn).lower()
                in ("1", "true", "yes", "on"),
                "grad_checkpointing": str(args.scgpt_grad_checkpointing).lower()
                in ("1", "true", "yes", "on"),
                "max_seq_len": int(args.scgpt_max_seq_len),
                "class_weighting": args.scgpt_class_weighting,
            },
            "runtime": {"nproc": max(1, os.cpu_count() or 1), "ngpu": ngpu},
        }
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # Say which split this job uses, so the log shows it (an image built before
    # --group-column existed ignores that hyperparameter and splits by cell).
    if cfg.classifier.group_column:
        print(f"[split] by group column {cfg.classifier.group_column!r}: no group on both sides", flush=True)
    else:
        print("[split] by cell (no group column)", flush=True)

    backend = get_backend(cfg.model.backend)
    backend.validate_config(cfg)

    if tokenized is not None:
        # A pre-tokenized dataset skips the tokenizer (where raw files are checked), so
        # apply the size and cell caps to it here.
        check_dataset_dir(tokenized, cfg.input.limits)
        dataset_path = tokenized
    else:
        dataset_path = backend.tokenize(cfg, cfg.input.h5ad_path).dataset_path
    result = backend.finetune(cfg, dataset_path)

    print(f"Fine-tuned model written to {result.model_dir}")
    if result.metrics:
        print("Metrics:", result.metrics)


if __name__ == "__main__":
    main()
