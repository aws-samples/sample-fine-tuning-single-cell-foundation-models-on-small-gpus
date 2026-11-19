# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Stage 1: tokenize a single-cell dataset into a Geneformer rank-value encoding.

Geneformer does not consume raw expression matrices directly. Each cell's
transcriptome is converted to a *rank value encoding* (genes ordered by expression,
normalized against the pretraining corpus) and stored as a HuggingFace Dataset on
disk. ``TranscriptomeTokenizer`` performs this conversion; the resulting ``.dataset``
is the input to both fine-tuning and evaluation.

Accepted inputs (format inferred from the file suffix):
  - ``.loom`` : ``ensembl_id`` row attribute, ``n_counts`` column attribute, labels
  - ``.h5ad`` : ``ensembl_id`` in ``.var``, ``n_counts`` in ``.obs``, labels in ``.obs``

See ``data.py`` for why the synthetic smoke data is loom.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from scfm_ft.config import Config
from scfm_ft.input_checks import check_input_file

_FORMAT_BY_SUFFIX = {".loom": "loom", ".h5ad": "h5ad"}


def tokenize(cfg: Config) -> Path:
    """Tokenize ``cfg.input.h5ad_path`` into a Geneformer ``.dataset``.

    Returns the path to the tokenized ``.dataset`` directory.
    """
    # Imported lazily so that `scfm-ft` subcommands that don't need the
    # heavy ML stack (e.g. make-sample-data, config validation) stay fast and
    # importable without geneformer installed.
    from geneformer import TranscriptomeTokenizer

    in_path = cfg.input.h5ad_path
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Header-only checks (format, size, cell and gene caps) before any reader loads the
    # matrix; rejects non-HDF5 and oversized input with a clear message.
    check_input_file(in_path, cfg.input.limits)

    file_format = _FORMAT_BY_SUFFIX.get(in_path.suffix.lower())
    if file_format is None:
        raise ValueError(
            f"Unsupported input suffix {in_path.suffix!r}; expected one of "
            f"{sorted(_FORMAT_BY_SUFFIX)}."
        )

    out_dir = cfg.tokenized_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.name

    # Carry the label column (and the group column, for donor-aware splits) through
    # tokenization so the Classifier can read them.
    custom_attr = {cfg.classifier.label_column: cfg.classifier.label_column}
    if cfg.classifier.group_column:
        custom_attr[cfg.classifier.group_column] = cfg.classifier.group_column

    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict=custom_attr,
        model_input_size=cfg.model.input_size,
        # model_version drives auto-selection of the matching gene-median / token
        # dictionaries (gc30M for V1, gc104M for V2) and the special_token setting.
        model_version=cfg.model.version,
        nproc=cfg.runtime.nproc,
    )

    # tokenize_data scans a *directory* by file_format, so isolate the input file in
    # its own temporary directory to avoid picking up unrelated files alongside it.
    with tempfile.TemporaryDirectory() as scan_dir:
        link = Path(scan_dir) / in_path.name
        shutil.copy(in_path, link)
        tokenizer.tokenize_data(
            data_directory=scan_dir,
            output_directory=str(out_dir),
            output_prefix=prefix,
            file_format=file_format,
        )

    dataset_path = out_dir / f"{prefix}.dataset"
    if not dataset_path.exists():
        raise RuntimeError(
            f"Tokenization did not produce the expected dataset at {dataset_path}. "
            f"Check that {in_path} has 'ensembl_id' and 'n_counts' (and the label column)."
        )
    return dataset_path
