# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Header-only input checks: format, size and cell/gene caps (threat T004)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from scfm_ft.config import Config
from scfm_ft.input_checks import (
    GIB,
    InputLimits,
    InputRejected,
    check_dataset_dir,
    check_input_file,
)


def _loom(path, n_genes=3, n_cells=4):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix", data=np.ones((n_genes, n_cells), dtype="f4"))
    return path


def _h5ad_dense(path, n_cells=4, n_genes=3):
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=np.ones((n_cells, n_genes), dtype="f4"))
    return path


def _h5ad_sparse(path, n_cells=5, n_genes=6):
    with h5py.File(path, "w") as f:
        g = f.create_group("X")
        g.attrs["shape"] = (n_cells, n_genes)
        g.create_dataset("data", data=np.ones(2, dtype="f4"))
    return path


def test_accepts_loom_and_reports_cells_by_genes(tmp_path):
    info = check_input_file(_loom(tmp_path / "a.loom", n_genes=7, n_cells=11))
    assert (info.fmt, info.n_cells, info.n_genes) == ("loom", 11, 7)  # loom is genes x cells


def test_accepts_dense_and_sparse_h5ad(tmp_path):
    dense = check_input_file(_h5ad_dense(tmp_path / "d.h5ad", n_cells=4, n_genes=3))
    sparse = check_input_file(_h5ad_sparse(tmp_path / "s.h5ad", n_cells=5, n_genes=6))
    assert (dense.n_cells, dense.n_genes) == (4, 3)
    assert (sparse.n_cells, sparse.n_genes) == (5, 6)


def test_rejects_too_many_cells_and_genes(tmp_path):
    p = _loom(tmp_path / "a.loom", n_genes=3, n_cells=50)
    with pytest.raises(InputRejected, match="50 cells.*input.max_cells"):
        check_input_file(p, InputLimits(max_cells=10))
    with pytest.raises(InputRejected, match="3 genes.*input.max_genes"):
        check_input_file(p, InputLimits(max_genes=2))


def test_rejects_file_over_byte_cap(tmp_path):
    p = _loom(tmp_path / "a.loom")
    with pytest.raises(InputRejected, match="over the .* GiB limit"):
        check_input_file(p, InputLimits(max_file_bytes=10))


def test_rejects_a_header_that_claims_a_huge_matrix(tmp_path):
    # A decompression bomb: a tiny file whose header declares a ~2 TB matrix. It must be
    # refused from the shape alone, without ever reading or allocating the data.
    p = tmp_path / "bomb.h5ad"
    with h5py.File(p, "w") as f:
        f.create_dataset(
            "X", shape=(10_000_000, 50_000), dtype="f4", chunks=(1000, 1000), compression="gzip"
        )
    assert p.stat().st_size < 1_000_000
    with pytest.raises(InputRejected, match="cells, over the"):
        check_input_file(p)


def test_rejects_non_hdf5_content_behind_a_valid_suffix(tmp_path):
    p = tmp_path / "fake.h5ad"
    p.write_text("this is not HDF5, just a name\n")
    with pytest.raises(InputRejected, match="not an HDF5 file"):
        check_input_file(p)


def test_rejects_wrong_suffix_missing_file_and_directories(tmp_path):
    with pytest.raises(InputRejected, match="Unsupported input suffix"):
        check_input_file(_loom(tmp_path / "a.csv"))
    with pytest.raises(InputRejected, match="cannot be read"):
        check_input_file(tmp_path / "missing.loom")
    (tmp_path / "dir.loom").mkdir()
    with pytest.raises(InputRejected, match="not a regular file"):
        check_input_file(tmp_path / "dir.loom")


def test_rejects_hdf5_that_is_not_the_claimed_format(tmp_path):
    empty = tmp_path / "empty.loom"
    with h5py.File(empty, "w") as f:
        f.create_dataset("something_else", data=[1, 2, 3])
    with pytest.raises(InputRejected, match="no 2-D 'matrix'"):
        check_input_file(empty)
    no_x = tmp_path / "nox.h5ad"
    with h5py.File(no_x, "w") as f:
        f.create_dataset("obs", data=[1])
    with pytest.raises(InputRejected, match="no readable 'X'"):
        check_input_file(no_x)


def test_fails_closed_when_the_sparse_shape_is_missing(tmp_path):
    p = tmp_path / "s.h5ad"
    with h5py.File(p, "w") as f:
        f.create_group("X")  # a sparse-looking group with no declared shape
    with pytest.raises(InputRejected, match="no readable 'X'"):
        check_input_file(p)


def test_dataset_dir_size_cap_and_symlink_rejection(tmp_path):
    d = tmp_path / "x.dataset"
    d.mkdir()
    (d / "data.arrow").write_bytes(b"0" * 100)
    pytest.importorskip("datasets")  # row check needs a real dataset; size check does not
    with pytest.raises(InputRejected, match="over the .* GiB limit"):
        check_dataset_dir(d, InputLimits(max_file_bytes=10))
    (d / "link").symlink_to(d / "data.arrow")
    with pytest.raises(InputRejected, match="symlink"):
        check_dataset_dir(d)


def test_dataset_dir_cell_cap(tmp_path):
    datasets = pytest.importorskip("datasets")
    d = tmp_path / "x.dataset"
    datasets.Dataset.from_dict({"input_ids": [[1], [2], [3]]}).save_to_disk(str(d))
    assert check_dataset_dir(d, InputLimits(max_cells=3)) > 0
    with pytest.raises(InputRejected, match="3 cells, over the 2 limit"):
        check_dataset_dir(d, InputLimits(max_cells=2))


def test_config_exposes_the_caps_with_safe_defaults(tmp_path):
    cfg = Config.model_validate({"name": "t", "input": {"h5ad_path": "data/x.loom"}})
    lim = cfg.input.limits
    assert lim.max_file_bytes == 4 * GIB
    assert (lim.max_cells, lim.max_genes) == (1_000_000, 100_000)
    cfg = Config.model_validate(
        {"name": "t", "input": {"h5ad_path": "x.loom", "max_file_gib": 0.5, "max_cells": 9}}
    )
    assert cfg.input.limits.max_file_bytes == GIB // 2 and cfg.input.limits.max_cells == 9
    with pytest.raises(ValueError):
        Config.model_validate({"name": "t", "input": {"h5ad_path": "x.loom", "max_cells": 0}})


def test_both_backends_run_the_check_before_reading_the_matrix():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "scfm_ft"
    geneformer = (root / "tokenize.py").read_text()
    scgpt = (root / "backends" / "scgpt" / "backend.py").read_text()
    assert geneformer.index("check_input_file(") < geneformer.index("TranscriptomeTokenizer(")
    assert scgpt.index("check_input_file(") < scgpt.index("load_counts_matrix(input_path")
