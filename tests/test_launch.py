# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Tests for the SageMaker launcher's scGPT weights-channel wiring.

These exercise the pure arg-parse + model-root resolution logic without importing
the SageMaker SDK (which main() imports lazily) or submitting a job.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_LAUNCH = Path(__file__).resolve().parents[1] / "sagemaker" / "launch.py"
_spec = importlib.util.spec_from_file_location("launch", _LAUNCH)
assert _spec and _spec.loader
launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)


def test_backend_and_weights_args_parse() -> None:
    old = sys.argv
    try:
        sys.argv = [
            "launch.py",
            "--image-uri", "x",
            "--role", "r",
            "--backend", "scgpt",
            "--preset", "scgpt-human",
            "--scgpt-weights-s3", "s3://bucket/scgpt/",
            "--data-source", "pbmc3k",
        ]
        args = launch._parse()
    finally:
        sys.argv = old
    assert args.backend == "scgpt"
    assert args.scgpt_weights_s3 == "s3://bucket/scgpt/"


def test_model_root_uses_channel_when_weights_s3_given() -> None:
    # Option A: weights S3 prefix -> model_root is the mounted channel path.
    root = launch.resolve_scgpt_model_root("s3://bucket/scgpt/", "/scgpt")
    assert root == launch.WEIGHTS_CHANNEL_DIR == "/opt/ml/input/data/weights"


def test_model_root_falls_back_to_explicit_when_no_weights_s3() -> None:
    # Option B / default: no weights channel -> use the baked-in/explicit root.
    assert launch.resolve_scgpt_model_root(None, "/scgpt") == "/scgpt"
    assert launch.resolve_scgpt_model_root(None, "/custom/root") == "/custom/root"


def test_s3_paths_default_under_the_scoped_prefix() -> None:
    out = launch.resolve_s3_paths("sagemaker-us-east-1-123", "scfm-ft", None)
    assert out == "s3://sagemaker-us-east-1-123/scfm-ft/output"


def test_s3_paths_respect_explicit_output_and_prefix() -> None:
    assert launch.resolve_s3_paths("b", "/custom/", "s3://elsewhere/out") == "s3://elsewhere/out"
    assert launch.resolve_s3_paths("b", "/custom/", None) == "s3://b/custom/output"


def test_launcher_turns_off_profiler_and_debugger() -> None:
    # Both write to the bucket root by default, outside the role's scoped prefixes.
    src = _LAUNCH.read_text()
    assert "disable_profiler=True" in src
    assert "debugger_hook_config=False" in src
