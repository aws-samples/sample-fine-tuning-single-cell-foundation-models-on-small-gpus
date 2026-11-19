# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Real single-cell dataset loaders, emitted in Geneformer's expected loom format.

Unlike ``data.py`` (synthetic smoke data), these fetch and format genuine public
datasets so you can fine-tune on biologically meaningful labels.

Output format matches what ``tokenize.py`` consumes: a ``.loom`` with an
``ensembl_id`` row attribute, an ``n_counts`` column attribute, and a label column.
We emit loom rather than h5ad for the same reason as the synthetic generator — this
pinned Geneformer version's h5ad tokenization path indexes a pandas Series
positionally and breaks when AnnData stringifies the ``.var`` index.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scfm_ft.integrity import download_verified


@dataclass(frozen=True)
class PinnedFile:
    """A public file fetched over HTTPS and accepted only if it matches these values."""

    url: str
    sha256: str
    size: int


# scanpy's own loaders fetch PBMC3k from falexwolf.de (a personal domain) and from the
# `main` branch of a GitHub repo, with no integrity check, so whoever controls either
# could change what a job trains on. We download the same two files ourselves, verify
# them against these hashes, and point scanpy at the verified copies. The hashes were
# taken from the live files and match copies fetched a month earlier. The processed
# file's URL is pinned to the commit that last changed it (2019-03-06) instead of `main`.
PBMC3K_FILES: dict[str, PinnedFile] = {
    "pbmc3k_raw.h5ad": PinnedFile(
        url="https://falexwolf.de/data/pbmc3k_raw.h5ad",
        sha256="89a96f1beaa2dd83a687666d3f19a4513ac27a2a2d12581fcd77afed7ea653a1",
        size=5_855_727,
    ),
    "pbmc3k_processed.h5ad": PinnedFile(
        url=(
            "https://raw.githubusercontent.com/chanzuckerberg/cellxgene/"
            "68dfbcc2eb675e96c6a5e2a6b7a0d3465ccf46bc/example-dataset/pbmc3k.h5ad"
        ),
        sha256="0db367b991dd95809732b218539ede489bea99113807f62ebd7ccc970025fe38",
        size=24_653_425,
    ),
}


def fetch_pbmc3k_files(dest_dir: str | Path, *, opener=None) -> Path:
    """Download the two PBMC3k source files into ``dest_dir``, verifying each hash."""
    dest = Path(dest_dir)
    kwargs = {} if opener is None else {"opener": opener}
    for name, pin in PBMC3K_FILES.items():
        download_verified(pin.url, dest / name, sha256=pin.sha256, size=pin.size, **kwargs)
    return dest


def load_pbmc3k(
    out: str | Path = "data/pbmc3k.loom",
    label_column: str = "cell_type",
    min_genes: int = 200,
) -> Path:
    """Fetch 10x Genomics PBMC3k, label it with cell types, and write a Geneformer loom.

    PBMC3k is ~2,700 peripheral-blood mononuclear cells from a healthy donor — the
    canonical single-cell test dataset. The raw matrix is unlabeled, so we transfer
    the standard Louvain cell-type annotations that ``scanpy`` ships in
    ``pbmc3k_processed`` (CD4/CD8 T, B, NK, monocyte, DC, megakaryocyte), matched by
    barcode. Geneformer needs **raw counts**, which we take from ``pbmc3k()`` (the
    processed object is log-normalized/scaled and unusable for rank-value encoding).

    Args:
        out: Output path; the ``.loom`` suffix is enforced.
        label_column: Column attribute name to store the cell-type label under.
        min_genes: Drop cells expressing fewer than this many genes (QC).

    Returns:
        Path to the written ``.loom`` file.
    """
    import loompy
    import scanpy as sc

    # Download the two source files from pinned URLs, verify their SHA-256, and have
    # scanpy read those copies. It only downloads when a file is absent, so with both
    # present and verified it never touches the network. Both objects are fully read
    # into memory inside the block, so the downloaded files can be deleted afterwards.
    previous_dir = sc.settings.datasetdir
    with tempfile.TemporaryDirectory(prefix="scfm-pbmc3k-src-") as tmp:
        sc.settings.datasetdir = fetch_pbmc3k_files(tmp)
        try:
            # Raw counts (genes identified by symbol, with Ensembl IDs in var['gene_ids']).
            raw = sc.datasets.pbmc3k()
            # Processed object carries the cell-type labels (obs['louvain']).
            proc = sc.datasets.pbmc3k_processed()
        finally:
            sc.settings.datasetdir = previous_dir

    # Keep only cells that have a label (processed is a QC'd subset of raw), in the
    # processed object's order so labels line up by barcode.
    labeled_barcodes = [bc for bc in proc.obs_names if bc in set(raw.obs_names)]
    raw = raw[labeled_barcodes].copy()
    labels = proc.obs.loc[labeled_barcodes, "louvain"].astype(str).to_numpy()

    # Ensembl IDs live in var['gene_ids']; Geneformer keys its token dict by Ensembl.
    if "gene_ids" not in raw.var.columns:
        raise RuntimeError("pbmc3k raw matrix lacks var['gene_ids'] (Ensembl IDs).")
    ensembl_ids = raw.var["gene_ids"].astype(str).to_numpy()
    # Gene SYMBOLS live in var_names; scGPT keys its GeneVocab by symbol, so we must
    # carry both so either backend can match: Geneformer reads 'ensembl_id', scGPT
    # reads 'gene_symbols'. Writing only Ensembl IDs makes scGPT match 0 genes.
    gene_symbols = raw.var_names.astype(str).to_numpy()

    # Per-cell QC: require a minimum number of detected genes.
    sc.pp.filter_cells(raw, min_genes=min_genes)
    # filter_cells may have dropped cells — re-align labels to the surviving cells.
    kept = raw.obs_names
    label_lookup = dict(zip(labeled_barcodes, labels))
    labels = np.array([label_lookup[bc] for bc in kept])

    # Dense counts as float32, genes x cells (loom convention).
    counts = raw.X
    counts = (counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)).astype(
        np.float32
    )
    expr = counts.T  # genes x cells

    row_attrs = {"ensembl_id": ensembl_ids, "gene_symbols": gene_symbols}
    col_attrs = {
        "n_counts": expr.sum(axis=0).astype(np.float32),
        label_column: labels,
        "cell_id": np.asarray(list(kept)),
    }

    out_path = Path(out).expanduser().with_suffix(".loom")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    loompy.create(str(out_path), expr, row_attrs, col_attrs)
    return out_path
