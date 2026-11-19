# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Backend conformance tests.

Prove that every backend registered in ``REGISTRY`` satisfies the ``ModelBackend``
protocol and returns sane discovery values — offline, in milliseconds, with no GPU
and no ML stack installed. A backend that drops a method or mis-types ``name`` fails
HERE in CI, not three minutes into a GPU job. This is the guard that keeps "adding a
model is one registry line" honest.
"""

from __future__ import annotations

import pytest

from scfm_ft.backends import REGISTRY, get_backend
from scfm_ft.backends.base import ModelBackend


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_backend_satisfies_protocol(name: str) -> None:
    backend = get_backend(name)
    # runtime_checkable Protocol membership: a backend missing a method fails here.
    assert isinstance(backend, ModelBackend)
    assert backend.name == name


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_backend_presets_and_instance_hint(name: str) -> None:
    backend = get_backend(name)
    presets = backend.supported_presets()
    assert isinstance(presets, list) and presets, f"{name} has no presets"
    # default_instance_type returns a plausible SageMaker type for each preset.
    for preset in presets:
        itype = backend.default_instance_type(preset)
        assert isinstance(itype, str) and itype.startswith("ml."), itype
    # docker_layers documents the image recipe.
    layers = backend.docker_layers()
    assert isinstance(layers, list) and all(isinstance(x, str) for x in layers)


def test_get_backend_unknown_name_exits() -> None:
    with pytest.raises(SystemExit, match="Unknown model backend"):
        get_backend("does-not-exist")


def test_registry_has_both_backends() -> None:
    assert set(REGISTRY) == {"geneformer", "scgpt"}
