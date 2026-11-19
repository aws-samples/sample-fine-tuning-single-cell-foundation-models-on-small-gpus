# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Opt-in check of image digest pinning against a real ECR repository (T005).

Skipped unless SCFM_LIVE_ECR_URI is set to a tagged image you can ``DescribeImages``, e.g.

    AWS_PROFILE=<profile> SCFM_LIVE_ECR_URI=<acct>.dkr.ecr.<region>.amazonaws.com/geneformer-ft:latest \\
        pytest tests/test_launch_live_ecr.py

It makes one read-only ECR call and needs boto3 and credentials.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest

URI = os.environ.get("SCFM_LIVE_ECR_URI")
pytestmark = pytest.mark.skipif(not URI, reason="set SCFM_LIVE_ECR_URI to run against a real ECR repository")

_LAUNCH = Path(__file__).resolve().parents[1] / "sagemaker" / "launch.py"


def _launch():
    spec = importlib.util.spec_from_file_location("launch_live", _LAUNCH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tag_resolves_to_the_digest_ecr_reports():
    pytest.importorskip("boto3")
    assert URI is not None
    pinned = _launch().resolve_image_uri(URI)
    m = re.fullmatch(r"(?P<registry>[^/]+)/(?P<repo>[^@:]+)@(?P<digest>sha256:[0-9a-f]{64})", pinned)
    assert m, pinned
    assert URI.startswith(f"{m['registry']}/{m['repo']}:")


def test_resolved_digest_is_a_fixed_point():
    pytest.importorskip("boto3")
    assert URI is not None
    launch = _launch()
    pinned = launch.resolve_image_uri(URI)
    assert launch.resolve_image_uri(pinned) == pinned  # a digest URI needs no lookup
