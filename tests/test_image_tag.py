# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The launcher wrapper defaults to the newest b<N> build tag (ECR tags are immutable)."""

from __future__ import annotations

import os
import stat
import subprocess  # nosec B404 - runs the repository's own shell helper with a stubbed aws
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "image_tag.sh"


def _run(tmp_path: Path, stub_body: str) -> subprocess.CompletedProcess:
    aws = tmp_path / "aws"
    aws.write_text("#!/usr/bin/env bash\n" + stub_body + "\n")
    aws.chmod(aws.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    return subprocess.run(  # nosec B603 B607 - fixed argv, stubbed aws on PATH
        ["bash", "-c", f"source {HELPER}; newest_build_tag geneformer-ft us-east-1 someprofile"],
        capture_output=True, text=True, env=env, check=False,
    )


def test_picks_the_highest_build_number_not_the_highest_string(tmp_path):
    out = _run(tmp_path, "printf 'b9\\tlatest\\tb40\\tb100\\nb36\\n'")
    assert out.returncode == 0
    assert out.stdout.strip() == "b100"


def test_ignores_latest_and_untagged_digests(tmp_path):
    out = _run(tmp_path, "printf 'latest\\tv1\\n'")
    assert out.returncode == 1
    assert out.stdout.strip() == ""


def test_returns_failure_when_the_ecr_call_fails(tmp_path):
    out = _run(tmp_path, "exit 254")
    assert out.returncode == 1
    assert out.stdout.strip() == ""


def test_passes_repository_region_and_profile_to_aws(tmp_path):
    log = tmp_path / "args.txt"
    out = _run(tmp_path, f"echo \"$@\" > {log}; printf 'b1'")
    assert out.stdout.strip() == "b1"
    args = log.read_text()
    assert "--repository-name geneformer-ft" in args
    assert "--region us-east-1" in args
    assert "--profile someprofile" in args


def test_wrapper_sources_the_helper_and_no_longer_hardcodes_latest():
    text = (ROOT / "scripts" / "do-fine-tuning-job.sh").read_text()
    assert "image_tag.sh" in text
    assert 'IMAGE_URI=""' in text
    assert 'IMAGE_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/geneformer-ft:latest"' not in text


def test_docs_do_not_launch_or_push_geneformer_latest():
    # `latest` is frozen once the repository is immutable, so an example that launches it runs an
    # old, possibly ungated build, and one that pushes it fails. (scgpt-ft is covered elsewhere.)
    for rel in ("README.md", "docs/SAGEMAKER.md", "sagemaker/Dockerfile", "sagemaker/launch.py"):
        text = (ROOT / rel).read_text()
        for line in text.splitlines():
            if "geneformer-ft:latest" in line:
                assert "IMAGE_URI=" not in line and "--image-uri" not in line, f"{rel}: {line}"
                assert "docker push" not in line and "docker tag" not in line, f"{rel}: {line}"


@pytest.mark.parametrize("script", ["do-fine-tuning-job.sh", "image_tag.sh"])
def test_shell_scripts_parse(script):
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], capture_output=True, text=True, check=False)  # nosec B603 B607
    assert r.returncode == 0, r.stderr


def test_ignores_gate_staging_tags_so_an_unscanned_image_is_never_the_default(tmp_path):
    # sagemaker/buildspec.yml pushes gate-b<N> first and adds b<N> only after the scan gate.
    out = _run(tmp_path, "printf 'b40\\tgate-b41\\tgate-b42\\tb39\\n'")
    assert out.returncode == 0
    assert out.stdout.strip() == "b40"
