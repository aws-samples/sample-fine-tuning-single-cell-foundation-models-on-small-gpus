# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Generate a small, correctly-formatted synthetic dataset for smoke tests.

This is NOT biologically meaningful data. It exists so you can exercise the full
tokenize -> finetune -> evaluate pipeline (and the test suite) without downloading
a real single-cell dataset.

Output format: **loom**. Geneformer's tokenizer requires ``ensembl_id`` as a row
attribute and ``n_counts`` as a column attribute, plus any label column. We emit
loom rather than h5ad because this pinned Geneformer version's h5ad tokenization
path indexes a pandas Series positionally (``var["ensembl_id_collapsed"][loc]``),
which raises a KeyError whenever AnnData coerces the ``.var`` index to strings (it
always does). The loom path indexes the same values as a NumPy array, which is
positional and works. Real h5ad data can still be used if its ``.var`` index is a
true integer RangeIndex at tokenization time; loom avoids that footgun entirely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# A handful of real human Ensembl gene IDs so the tokenizer maps them to tokens.
# (Geneformer's token dictionary is keyed by Ensembl ID; unknown IDs are dropped.)
_SAMPLE_ENSEMBL_IDS = [
    "ENSG00000139618", "ENSG00000141510", "ENSG00000157764", "ENSG00000171862",
    "ENSG00000146648", "ENSG00000133703", "ENSG00000169083", "ENSG00000012048",
    "ENSG00000121879", "ENSG00000134086", "ENSG00000181143", "ENSG00000198793",
    "ENSG00000105221", "ENSG00000142208", "ENSG00000165731", "ENSG00000068078",
    "ENSG00000169032", "ENSG00000100030", "ENSG00000102882", "ENSG00000177606",
]


def make_sample_data(
    out: str | Path,
    n_cells: int = 200,
    label_column: str = "cell_type",
    n_classes: int = 3,
    seed: int = 0,
) -> Path:
    """Write a synthetic ``.loom`` file with ``n_cells`` cells and ``n_classes`` labels.

    The returned path has a ``.loom`` suffix regardless of what ``out`` requested,
    so the tokenizer's ``file_format`` can be inferred.
    """
    import loompy

    rng = np.random.default_rng(seed)
    n_genes = len(_SAMPLE_ENSEMBL_IDS)

    # loom stores the matrix as genes x cells.
    labels = rng.integers(0, n_classes, size=n_cells)
    expr = rng.poisson(lam=5, size=(n_genes, n_cells)).astype(np.float32)
    # Give each class a slightly different profile so a classifier can separate them.
    for c in range(n_classes):
        boost_genes = rng.choice(n_genes, size=3, replace=False)
        cols = labels == c
        expr[np.ix_(boost_genes, cols)] += rng.poisson(
            lam=20, size=(len(boost_genes), cols.sum())
        ).astype(np.float32)

    row_attrs = {"ensembl_id": np.array(_SAMPLE_ENSEMBL_IDS)}
    col_attrs = {
        "n_counts": expr.sum(axis=0).astype(np.float32),
        label_column: np.array([f"class_{c}" for c in labels]),
        "cell_id": np.array([f"cell_{i}" for i in range(n_cells)]),
    }

    out_path = Path(out).expanduser().with_suffix(".loom")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    loompy.create(str(out_path), expr, row_attrs, col_attrs)
    return out_path
