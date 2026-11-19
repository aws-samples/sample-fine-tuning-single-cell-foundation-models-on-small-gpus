# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Reject malformed or oversized input files before any heavy reader touches them.

``.loom`` and ``.h5ad`` are both HDF5 containers, and the readers behind them
(``loompy``, ``anndata``, Geneformer's tokenizer, scGPT's loader) load the whole matrix.
A file that is not HDF5, or whose header declares billions of cells, either crashes the
job late (after the GPU instance is billed) or exhausts the instance's memory. This
module checks the cheap facts first, from the file system and the HDF5 *header* only:

  - it is a regular file with a supported suffix and is not larger than the byte cap,
  - it is really an HDF5 file (checked by HDF5's own signature search, not the name),
  - the matrix the header declares is within the cell and gene caps. The shape is read
    from metadata, never by loading data, so a small file that *claims* a huge matrix
    (a decompression bomb) is rejected as cheaply as an honest large one.

It fails closed: if the dimensions cannot be determined, the file is rejected.

The caps are generous defaults for single-instance fine-tuning, not a judgement about
what is biologically reasonable; raise them in ``input:`` in the config (or with the
launcher's ``--max-cells`` / ``--max-input-gib``) when you have a larger dataset and the
memory for it.
"""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path

GIB = 2**30

DEFAULT_MAX_FILE_BYTES = 4 * GIB
DEFAULT_MAX_CELLS = 1_000_000
DEFAULT_MAX_GENES = 100_000

_FORMAT_BY_SUFFIX = {".loom": "loom", ".h5ad": "h5ad"}


class InputRejected(ValueError):
    """The input file failed a safety check; the message says which and why."""


@dataclass(frozen=True)
class InputLimits:
    """Upper bounds an input must stay within."""

    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_cells: int = DEFAULT_MAX_CELLS
    max_genes: int = DEFAULT_MAX_GENES


@dataclass(frozen=True)
class InputInfo:
    """What was learned about an accepted input."""

    fmt: str
    size_bytes: int
    n_cells: int
    n_genes: int


def _hdf5_shape(path: Path, fmt: str) -> tuple[int, int]:
    """Return ``(n_cells, n_genes)`` from the HDF5 header, without loading any data."""
    import h5py

    try:
        if not h5py.is_hdf5(str(path)):
            raise InputRejected(f"{path.name} is not an HDF5 file (a .{fmt} must be).")
        with h5py.File(path, "r") as f:
            if fmt == "loom":
                matrix = f.get("matrix")
                if not isinstance(matrix, h5py.Dataset) or matrix.ndim != 2:
                    raise InputRejected(f"{path.name} has no 2-D 'matrix' dataset (not a loom file).")
                n_genes, n_cells = (int(d) for d in matrix.shape)  # loom is genes x cells
                return n_cells, n_genes
            x = f.get("X")
            if isinstance(x, h5py.Dataset) and x.ndim == 2:
                n_cells, n_genes = (int(d) for d in x.shape)
                return n_cells, n_genes
            if isinstance(x, h5py.Group):
                shape = x.attrs.get("shape", x.attrs.get("h5sparse_shape"))
                if shape is not None and len(shape) == 2:
                    return int(shape[0]), int(shape[1])
            raise InputRejected(f"{path.name} has no readable 'X' matrix (not an h5ad file).")
    except InputRejected:
        raise
    except (OSError, KeyError, ValueError, TypeError) as e:
        raise InputRejected(f"{path.name} could not be read as {fmt}: {e}") from e


def check_input_file(path: str | Path, limits: InputLimits | None = None) -> InputInfo:
    """Validate a raw ``.loom``/``.h5ad`` input; raise :class:`InputRejected` if unsafe."""
    limits = limits or InputLimits()
    path = Path(path)
    try:
        st = path.stat()
    except OSError as e:
        raise InputRejected(f"Input {path} cannot be read: {e}") from e
    if not stat.S_ISREG(st.st_mode):
        raise InputRejected(f"Input {path} is not a regular file.")

    fmt = _FORMAT_BY_SUFFIX.get(path.suffix.lower())
    if fmt is None:
        raise InputRejected(
            f"Unsupported input suffix {path.suffix!r}; expected one of {sorted(_FORMAT_BY_SUFFIX)}."
        )
    if st.st_size > limits.max_file_bytes:
        raise InputRejected(
            f"Input {path.name} is {st.st_size / GIB:.2f} GiB, over the "
            f"{limits.max_file_bytes / GIB:.2f} GiB limit (input.max_file_gib)."
        )

    n_cells, n_genes = _hdf5_shape(path, fmt)
    if n_cells <= 0 or n_genes <= 0:
        raise InputRejected(f"Input {path.name} declares an empty matrix ({n_cells} x {n_genes}).")
    if n_cells > limits.max_cells:
        raise InputRejected(
            f"Input {path.name} has {n_cells:,} cells, over the {limits.max_cells:,} limit "
            "(input.max_cells)."
        )
    if n_genes > limits.max_genes:
        raise InputRejected(
            f"Input {path.name} has {n_genes:,} genes, over the {limits.max_genes:,} limit "
            "(input.max_genes)."
        )
    return InputInfo(fmt=fmt, size_bytes=st.st_size, n_cells=n_cells, n_genes=n_genes)


def check_dataset_dir(path: str | Path, limits: InputLimits | None = None) -> int:
    """Validate a pre-tokenized ``.dataset`` directory; return its total size in bytes.

    Applies the byte cap to the whole directory and, when the ``datasets`` library is
    available, the cell cap to its row count. Symlinks and non-regular entries are
    rejected so the directory cannot point outside itself.
    """
    limits = limits or InputLimits()
    root = Path(path)
    if not root.is_dir():
        raise InputRejected(f"Dataset {root} is not a directory.")
    total = 0
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise InputRejected(f"Dataset {root.name} contains a symlink ({entry.name}).")
        if entry.is_file():
            total += entry.stat().st_size
            if total > limits.max_file_bytes:
                raise InputRejected(
                    f"Dataset {root.name} is over the {limits.max_file_bytes / GIB:.2f} GiB "
                    "limit (input.max_file_gib)."
                )
    try:
        from datasets import load_from_disk
    except ImportError:  # scGPT image: no tokenized-dataset form, nothing more to check
        return total
    try:
        rows = len(load_from_disk(str(root)))
    except Exception as e:  # any failure to open the dataset rejects it
        raise InputRejected(f"Dataset {root.name} could not be opened: {e}") from e
    if rows > limits.max_cells:
        raise InputRejected(
            f"Dataset {root.name} has {rows:,} cells, over the {limits.max_cells:,} limit "
            "(input.max_cells)."
        )
    return total
