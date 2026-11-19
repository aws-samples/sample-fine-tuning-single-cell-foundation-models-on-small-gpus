# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Backend registry — the single place touched when adding a model.

Maps a backend name (``"geneformer"``, ``"scgpt"``) to its :class:`ModelBackend`
implementation. Every shared layer (``cli.py``, ``sagemaker/sm_entry.py``) dispatches
through :func:`get_backend`; none of them imports a concrete backend directly.

Importing a backend CLASS here is cheap — each backend keeps its heavy ML imports
(``geneformer``, ``scgpt``, ``torch``, ``flash-attn``) lazy inside its stage methods,
so this module imports without the training stack installed.
"""

from __future__ import annotations

from scfm_ft.backends.base import (
    FinetuneResult,
    ModelBackend,
    TokenizeResult,
)
from scfm_ft.backends.geneformer import GeneformerBackend
from scfm_ft.backends.scgpt import ScgptBackend

#: Name → backend class. Adding a model is one new line here (plus the module).
REGISTRY: dict[str, type[ModelBackend]] = {
    "geneformer": GeneformerBackend,
    "scgpt": ScgptBackend,
}

__all__ = [
    "REGISTRY",
    "FinetuneResult",
    "ModelBackend",
    "TokenizeResult",
    "get_backend",
]


def get_backend(name: str) -> ModelBackend:
    """Instantiate the backend registered under ``name``.

    Raises ``SystemExit`` with a helpful message on an unknown name so the CLI /
    SageMaker entry fail clearly rather than with a ``KeyError``.
    """
    try:
        return REGISTRY[name]()
    except KeyError:
        raise SystemExit(
            f"Unknown model backend {name!r}; choose from {sorted(REGISTRY)}."
        )
