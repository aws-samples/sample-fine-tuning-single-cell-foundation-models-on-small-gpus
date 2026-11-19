# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The Geneformer Classifier that `evaluate` builds to plot and score results."""

from __future__ import annotations

from scfm_ft.config import Config
from scfm_ft.evaluate import _classifier_kwargs


def _cfg(**classifier) -> Config:
    return Config.model_validate(
        {"name": "t", "input": {"h5ad_path": "data/x.h5ad"}, "classifier": classifier}
    )


def test_cell_classifier_gets_a_cell_state_dict():
    # Geneformer's Classifier.validate_options calls cell_state_dict.keys() for a cell
    # classifier, so omitting it crashed `scfm-ft evaluate` with AttributeError.
    kw = _classifier_kwargs(_cfg(label_column="cell_type"))
    assert kw["cell_state_dict"] == {"state_key": "cell_type", "states": "all"}
    assert kw["classifier"] == "cell"


def test_cell_state_dict_honours_explicit_states():
    kw = _classifier_kwargs(_cfg(label_column="ct", states=["B", "NK"]))
    assert kw["cell_state_dict"] == {"state_key": "ct", "states": ["B", "NK"]}


def test_gene_classifier_has_no_cell_state_dict():
    assert "cell_state_dict" not in _classifier_kwargs(_cfg(kind="gene"))
