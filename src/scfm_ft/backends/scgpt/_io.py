# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Input loading for the scGPT backend.

Reads the SAME ``.loom``/``.h5ad`` the shared loaders in ``datasets.py``/``data.py``
emit, but returns the matrix in the shape scGPT wants: gene identifiers, a dense
cells × genes raw-count matrix, and the per-cell label vector. scGPT keys its
vocabulary by **gene symbol**, whereas Geneformer keys by Ensembl ID — both are
carried in the loom (``ensembl_id`` row attr; the loaders also retain symbols), so
this prefers a gene-symbol attribute when present and falls back to Ensembl IDs.

Kept separate from ``backend.py`` so importing the backend (for discovery/validation/
dispatch tests) never pulls in ``loompy``/``anndata``.
"""

from __future__ import annotations

from pathlib import Path


def load_counts_matrix(input_path: Path, label_column: str, group_column: str | None = None):
    """Return ``(genes, counts, labels, groups)`` for scGPT tokenization.

    * ``genes``  — 1-D array of gene identifiers (symbols preferred, else Ensembl).
    * ``counts`` — dense ``float32`` matrix, cells × genes (raw counts).
    * ``labels`` — 1-D array of per-cell string labels from ``label_column``.
    * ``groups`` — 1-D array of per-cell string group ids (usually donors) from
      ``group_column``, or ``None`` when no group column is given.
    """

    suffix = input_path.suffix.lower()
    if suffix == ".loom":
        return _load_loom(input_path, label_column, group_column)
    if suffix == ".h5ad":
        return _load_h5ad(input_path, label_column, group_column)
    raise ValueError(
        f"Unsupported input suffix {input_path.suffix!r}; expected .loom or .h5ad."
    )


def _gene_names_from(names_by_attr: dict, n_genes: int):
    """Pick the best available gene-name attribute (symbol > ensembl > index)."""
    import numpy as np

    for key in ("gene_symbols", "gene_name", "var_names", "Gene", "gene_ids", "ensembl_id"):
        if key in names_by_attr and names_by_attr[key] is not None:
            return np.asarray(names_by_attr[key]).astype(str)
    return np.array([f"gene_{i}" for i in range(n_genes)])


def _load_loom(input_path: Path, label_column: str, group_column: str | None = None):
    import loompy
    import numpy as np

    with loompy.connect(str(input_path)) as ds:
        expr = ds[:, :]  # genes x cells
        counts = np.asarray(expr, dtype=np.float32).T  # cells x genes
        n_genes = counts.shape[1]
        genes = _gene_names_from(dict(ds.ra.items()), n_genes)
        if label_column not in ds.ca:
            raise KeyError(
                f"Label column {label_column!r} not found in loom column attrs "
                f"{list(ds.ca.keys())}."
            )
        labels = np.asarray(ds.ca[label_column]).astype(str)
        groups = None
        if group_column:
            if group_column not in ds.ca:
                raise KeyError(
                    f"Group column {group_column!r} not found in loom column attrs "
                    f"{list(ds.ca.keys())}."
                )
            groups = np.asarray(ds.ca[group_column]).astype(str)
    return genes, counts, labels, groups


def _load_h5ad(input_path: Path, label_column: str, group_column: str | None = None):
    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(input_path)
    X = adata.X
    counts = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(np.float32)
    names_by_attr = {
        "gene_symbols": adata.var["gene_symbols"].to_numpy()
        if "gene_symbols" in adata.var
        else None,
        "gene_ids": adata.var["gene_ids"].to_numpy() if "gene_ids" in adata.var else None,
        "ensembl_id": adata.var["ensembl_id"].to_numpy() if "ensembl_id" in adata.var else None,
        "var_names": adata.var_names.to_numpy(),
    }
    genes = _gene_names_from(names_by_attr, counts.shape[1])
    if label_column not in adata.obs:
        raise KeyError(
            f"Label column {label_column!r} not found in .obs {list(adata.obs.columns)}."
        )
    labels = adata.obs[label_column].astype(str).to_numpy()
    groups = None
    if group_column:
        if group_column not in adata.obs:
            raise KeyError(
                f"Group column {group_column!r} not found in .obs {list(adata.obs.columns)}."
            )
        groups = adata.obs[group_column].astype(str).to_numpy()
    return genes, counts, labels, groups
