# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""sagemaker/buildspec.yml: immutable unique tags, digest output, scan report (threat T005)."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - the tests run buildspec one-liners under bash with stubbed docker
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = yaml.safe_load((ROOT / "sagemaker" / "buildspec.yml").read_text())
PHASES = {name: [str(c) for c in SPEC["phases"][name]["commands"]] for name in SPEC["phases"]}
ALL = [c for cmds in PHASES.values() for c in cmds]

bash = shutil.which("bash")
needs_bash = pytest.mark.skipif(bash is None, reason="bash not available")


def _run(script: str, **env: str) -> subprocess.CompletedProcess:
    full_env = {"PATH": "/usr/bin:/bin", **env}
    return subprocess.run(  # nosec B603 - fixed bash, script is a repo constant, docker is stubbed
        [bash, "-c", script], capture_output=True, text=True, env=full_env, check=False
    )


def test_spec_parses_as_version_0_2_with_the_three_phases():
    assert SPEC["version"] == 0.2
    assert set(PHASES) == {"pre_build", "build", "post_build"}


def test_new_repositories_are_created_immutable_and_scanned():
    create = next(c for c in PHASES["pre_build"] if "create-repository" in c)
    assert "--image-tag-mutability IMMUTABLE" in create
    assert "scanOnPush=true" in create


def test_every_build_pushes_a_unique_tag_and_reports_the_digest():
    assert any(c.startswith('BUILD_TAG="b${CODEBUILD_BUILD_NUMBER') for c in PHASES["pre_build"])
    assert any(c.startswith('docker push "$ECR_URI/$IMAGE_REPO_NAME:$BUILD_TAG"') for c in PHASES["post_build"])
    assert any("RepoDigests" in c and c.startswith("IMAGE_DIGEST_URI=") for c in PHASES["post_build"])
    assert any("sagemaker/post_build.py" in c and '--image "$IMAGE_DIGEST_URI"' in c for c in PHASES["post_build"])
    assert PHASES["post_build"][-1].startswith("echo") and "$IMAGE_DIGEST_URI" in PHASES["post_build"][-1]


def test_the_moving_tag_is_never_pushed_unconditionally():
    # An immutable repository rejects a re-pushed tag, so any push of $IMAGE_TAG must sit
    # behind the MUTABLE check.
    for c in ALL:
        if "docker push" in c and "$IMAGE_TAG" in c:
            assert 'REPO_MUTABILITY" = "MUTABLE"' in c, c


def test_post_build_does_not_push_when_the_build_failed():
    assert "CODEBUILD_BUILD_SUCCEEDING" in PHASES["post_build"][0]
    assert PHASES["post_build"][0].rstrip().endswith("exit 1; }")


@needs_bash
def test_the_failure_guard_stops_post_build_only_when_the_build_failed():
    guard = PHASES["post_build"][0]
    ok = _run(guard, CODEBUILD_BUILD_SUCCEEDING="1")
    bad = _run(guard, CODEBUILD_BUILD_SUCCEEDING="0")
    assert ok.returncode == 0
    assert bad.returncode == 1 and "nothing to push" in bad.stdout


@needs_bash
def test_every_command_is_valid_shell():
    for c in ALL:
        r = subprocess.run([bash, "-n", "-c", c], capture_output=True, text=True, check=False)  # nosec B603
        assert r.returncode == 0, f"{c}\n{r.stderr}"


def _moving_tag_step() -> str:
    return next(c for c in PHASES["post_build"] if "REPO_MUTABILITY" in c)


@needs_bash
def test_a_mutable_repository_still_gets_the_moving_tag():
    stub = 'docker() { echo "docker $*"; }; '
    r = _run(
        stub + _moving_tag_step(),
        REPO_MUTABILITY="MUTABLE", IMAGE_TAG="latest", ECR_URI="reg", IMAGE_REPO_NAME="r", BUILD_TAG="b7",
    )
    assert r.returncode == 0
    assert "docker tag reg/r:b7 reg/r:latest" in r.stdout and "docker push reg/r:latest" in r.stdout


@needs_bash
def test_an_immutable_repository_never_gets_a_moving_tag():
    stub = 'docker() { echo "docker $*"; }; '
    for env in (
        {"REPO_MUTABILITY": "IMMUTABLE", "IMAGE_TAG": "latest"},
        {"REPO_MUTABILITY": "MUTABLE", "IMAGE_TAG": ""},  # nothing configured: nothing to push
    ):
        r = _run(stub + _moving_tag_step(), ECR_URI="reg", IMAGE_REPO_NAME="r", BUILD_TAG="b7", **env)
        assert r.returncode == 0
        assert "docker push" not in r.stdout and "Not pushing a moving tag" in r.stdout


def test_the_script_the_buildspec_calls_exists_and_is_given_the_lock_files():
    call = next(c for c in PHASES["post_build"] if "sagemaker/post_build.py" in c)
    assert (ROOT / "sagemaker" / "post_build.py").is_file()
    assert "--lock uv.lock" in call and "--dockerfile" in call


# --- M022: the b<N> tag exists only for an image that went through the scan gate ----------------
def _index(prefix: str, *, phase: str = "post_build", contains: str = "") -> int:
    return next(i for i, c in enumerate(PHASES[phase]) if c.startswith(prefix) and contains in c)


def test_the_image_is_staged_scanned_and_only_then_given_its_build_tag():
    stage = _index("docker push", contains="$STAGING_TAG")
    gate = _index("python3 sagemaker/post_build.py")
    tag = _index("docker tag", contains=":$BUILD_TAG")
    push = _index("docker push", contains=":$BUILD_TAG")
    assert stage < gate < tag < push
    # The scan is read from the staged image, so the digest must come from the staging tag.
    assert "$STAGING_TAG" in next(c for c in PHASES["post_build"] if c.startswith("IMAGE_DIGEST_URI="))
    # Nothing but the staging tag is pushed before the gate.
    assert not [c for c in PHASES["post_build"][:gate] if "docker push" in c and "$STAGING_TAG" not in c]


def test_the_moving_tag_also_waits_for_the_gate():
    gate = _index("python3 sagemaker/post_build.py")
    assert _index("if [ \"$REPO_MUTABILITY\"") > gate


def test_the_staging_tag_never_looks_like_a_build_tag():
    import re

    assert re.fullmatch(r"b[0-9]+", "b42") and not re.fullmatch(r"b[0-9]+", "gate-b42")
    assert any(c.startswith('STAGING_TAG="gate-') for c in PHASES["pre_build"])


@needs_bash
@pytest.mark.parametrize("gate_exit, tagged", [(0, True), (3, False), (4, False)])
def test_a_build_that_fails_the_gate_never_gets_a_build_tag(gate_exit, tagged):
    # CodeBuild stops a phase at the first failing command, which `set -e` reproduces.
    stub = (
        "set -e\n"
        'docker() { if [ "$1" = inspect ]; then echo "reg/r@sha256:abc"; else echo "docker $*"; fi; }\n'
        f"python3() {{ echo \"post_build $*\" >/dev/null; return {gate_exit}; }}\n"
    )
    script = stub + "\n".join(PHASES["post_build"]) + "\n"
    r = _run(
        script,
        CODEBUILD_BUILD_SUCCEEDING="1", ECR_URI="reg", IMAGE_REPO_NAME="r", BUILD_TAG="b7", STAGING_TAG="gate-b7",
        BACKEND="geneformer", DOCKERFILE="sagemaker/Dockerfile", REPO_MUTABILITY="MUTABLE", IMAGE_TAG="latest",
    )
    assert "docker push reg/r:gate-b7" in r.stdout
    assert ("docker push reg/r:b7" in r.stdout) is tagged
    assert ("docker push reg/r:latest" in r.stdout) is tagged
    assert (r.returncode == 0) is tagged
