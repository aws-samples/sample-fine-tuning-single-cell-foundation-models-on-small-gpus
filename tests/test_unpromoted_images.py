# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""scripts/unpromoted_images.sh lists images that were staged but never got a b<N> tag (M022)."""

from __future__ import annotations

import json
import os
import stat
import subprocess  # nosec B404 - runs the repository's own shell script with a stubbed aws
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "unpromoted_images.sh"


def _run(tmp_path: Path, images: list[dict] | None, *, fail: bool = False) -> subprocess.CompletedProcess:
    aws = tmp_path / "aws"
    body = "exit 254" if fail else f"cat <<'JSON'\n{json.dumps({'imageDetails': images})}\nJSON"
    aws.write_text("#!/usr/bin/env bash\n" + body + "\n")
    aws.chmod(aws.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    return subprocess.run(  # nosec B603 B607 - fixed argv, stubbed aws on PATH
        [str(SCRIPT), "geneformer-ft", "us-east-1", "someprofile"],
        capture_output=True, text=True, env=env, check=False,
    )


def _img(digest: str, *tags: str) -> dict:
    d = {"imageDigest": digest, "imagePushedAt": "2026-11-17T20:54:24+00:00"}
    if tags:
        d["imageTags"] = list(tags)
    return d


def test_lists_a_staged_image_that_was_never_promoted(tmp_path):
    out = _run(tmp_path, [_img("sha256:aaa", "gate-b44")])
    assert out.returncode == 0
    assert out.stdout.split("\t")[:2] == ["sha256:aaa", "gate-b44"]


def test_does_not_list_promoted_images_or_other_tags(tmp_path):
    out = _run(tmp_path, [
        _img("sha256:good", "gate-b43", "b43"),      # promoted: carries both tags
        _img("sha256:old", "b39"),
        _img("sha256:moving", "latest", "b40"),
        _img("sha256:untagged"),
        _img("sha256:other", "v1"),
    ])
    assert out.returncode == 0 and out.stdout == ""


def test_lists_only_the_unpromoted_among_several(tmp_path):
    out = _run(tmp_path, [_img("sha256:good", "gate-b43", "b43"), _img("sha256:bad", "gate-b44")])
    assert [line.split("\t")[0] for line in out.stdout.splitlines()] == ["sha256:bad"]


def test_a_failed_lookup_is_an_error_not_an_empty_list(tmp_path):
    out = _run(tmp_path, None, fail=True)
    assert out.returncode == 1 and "could not list images" in out.stderr


def test_script_parses_and_documents_why_there_is_no_lifecycle_rule():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)  # nosec B603 B607
    assert r.returncode == 0, r.stderr
    assert "lifecycle rule on the `gate-` prefix" in SCRIPT.read_text()
