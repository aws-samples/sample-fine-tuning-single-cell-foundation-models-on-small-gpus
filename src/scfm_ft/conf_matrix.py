# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Confusion-matrix persistence as JSON (not pickle).

The fine-tune stage writes the held-out confusion matrix next to the model so the
evaluate stage can plot it without re-running inference. JSON in pandas' "split"
layout keeps the class labels and their order, and, unlike pickle, loading it cannot
run code.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


def conf_matrix_path(model_out_dir: Path, prefix: str) -> Path:
    return Path(model_out_dir) / f"{prefix}_conf_matrix.json"


def save_conf_matrix(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.write_text(df.to_json(orient="split"))
    return path


def load_conf_matrix(path: Path) -> pd.DataFrame:
    # convert_axes=False keeps labels as strings (a class named "1" stays "1").
    return pd.read_json(io.StringIO(Path(path).read_text()), orient="split", convert_axes=False)
