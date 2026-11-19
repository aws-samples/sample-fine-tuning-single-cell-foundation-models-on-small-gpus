# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Training-image base images: pulled from nvcr.io and pinned by digest."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["Dockerfile", "Dockerfile.scgpt"])
def test_base_image_is_digest_pinned_and_not_from_docker_hub(name):
    froms = re.findall(r"^FROM\s+(\S+)", (ROOT / "sagemaker" / name).read_text(), re.MULTILINE)
    assert froms, f"{name} has no FROM"
    for ref in froms:
        # Docker Hub's anonymous pull limit fails CodeBuild at this line (429).
        assert ref.startswith("nvcr.io/"), f"{name}: {ref} is not from nvcr.io"
        assert re.search(r"@sha256:[0-9a-f]{64}$", ref), f"{name}: {ref} is not digest-pinned"


@pytest.mark.parametrize("name", ["Dockerfile", "Dockerfile.scgpt"])
def test_os_packages_are_upgraded_with_nvidia_libraries_held(name):
    text = (ROOT / "sagemaker" / name).read_text()
    assert "apt-get upgrade -y" in text
    assert text.index("apt-mark hold") < text.index("apt-get upgrade -y")
    assert text.index("apt-get upgrade -y") < text.index("apt-get install -y")
