# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Sample-data generator tests. Requires loompy; skips cleanly if absent."""

from __future__ import annotations

import numpy as np
import pytest

loompy = pytest.importorskip("loompy")

from scfm_ft.data import make_sample_data


def test_make_sample_data_shape_and_attrs(tmp_path):
    out = make_sample_data(tmp_path / "s", n_cells=50, n_classes=3, seed=1)
    assert out.exists()
    assert out.suffix == ".loom"

    with loompy.connect(str(out)) as ds:
        # loom is genes x cells
        _n_genes, n_cells = ds.shape
        assert n_cells == 50
        assert "ensembl_id" in ds.ra
        assert "n_counts" in ds.ca
        assert "cell_type" in ds.ca
        assert len(set(ds.ca["cell_type"])) <= 3
        # n_counts should equal the per-cell column sums of the matrix.
        assert np.allclose(ds.ca["n_counts"], ds[:, :].sum(axis=0))


def test_custom_label_column(tmp_path):
    out = make_sample_data(tmp_path / "s2", n_cells=20, label_column="disease", n_classes=2)
    with loompy.connect(str(out)) as ds:
        assert "disease" in ds.ca
