# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The scGPT and bootstrap image installs are hash-locked (T001)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SM = Path(__file__).resolve().parents[1] / "sagemaker"
LOCKS = ["requirements-scgpt.txt", "requirements-scgpt-build.txt", "requirements-scgpt-flashattn.txt", "requirements-bootstrap.txt",
         "requirements-toolkit.txt", "requirements-toolkit-build.txt"]


def _requirements(name: str) -> dict[str, list[str]]:
    """Map 'name==version' to its sha256 list, from a pip requirements file."""
    text = (SM / name).read_text().replace("\\\n", " ")
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        spec = line.split()[0]
        out[spec] = re.findall(r"--hash=sha256:([0-9a-f]{64})", line)
    return out


@pytest.mark.parametrize("name", LOCKS)
def test_every_requirement_is_pinned_exactly_and_hashed(name):
    reqs = _requirements(name)
    assert reqs
    for spec, hashes in reqs.items():
        assert re.fullmatch(r"[A-Za-z0-9._-]+==[A-Za-z0-9.+!_-]+", spec), spec
        assert hashes, f"{spec} has no hash"


@pytest.mark.parametrize("name", LOCKS)
def test_no_urls_or_local_paths_in_locks(name):
    text = (SM / name).read_text()
    assert not re.search(r"^\s*(-e|git\+|https?://|/|\.)", text, re.MULTILINE)
    assert "/home/" not in text


def test_scgpt_lock_keeps_the_torch_stack_flash_attn_needs():
    reqs = _requirements("requirements-scgpt.txt")
    assert "torch==2.0.1+cu117" in reqs
    assert "torchtext==0.15.2+cpu" in reqs
    assert not any(s.startswith("torch==2.1") for s in reqs)
    assert any(s.startswith("numpy==1.") for s in reqs)
    assert "scgpt==0.2.1" in reqs
    assert any(s.startswith("einops==") for s in reqs)


def test_flash_attn_is_built_from_a_hashed_sdist_only():
    assert list(_requirements("requirements-scgpt-flashattn.txt")) == ["flash-attn==1.0.9"]


def _pip_install_lines(dockerfile: str) -> list[str]:
    text = (SM / dockerfile).read_text().replace("\\\n", " ")
    return [ln for ln in text.splitlines() if re.search(r"\bpip3? install\b", ln) and not ln.lstrip().startswith("#")]


@pytest.mark.parametrize("dockerfile", ["Dockerfile.scgpt", "Dockerfile"])
def test_dockerfiles_install_only_hashed_requirements_or_the_local_package(dockerfile):
    for ln in _pip_install_lines(dockerfile):
        # uv pip install lines are covered by the uv.lock export / local paths below
        for part in re.split(r"&&", ln):
            if not re.search(r"\bpip3? install\b", part) or "uv pip install" in part:
                continue
            hashed = "--require-hashes" in part
            local = bool(re.search(r"install .*\s\.\s*$", part.strip() + " ")) and "--no-deps" in part
            assert hashed or local, f"unhashed install in {dockerfile}: {part.strip()}"


def test_scgpt_dockerfile_does_not_upgrade_pip_or_use_unpinned_installs():
    text = (SM / "Dockerfile.scgpt").read_text()
    assert "install --upgrade pip" not in text
    assert "--index-url" not in text.replace("--extra-index-url", "")
    assert "--no-build-isolation" in text
    assert "import torchtext" in text


def test_geneformer_dockerfile_bootstraps_uv_from_the_hashed_file():
    text = (SM / "Dockerfile").read_text()
    assert "pip3 install uv==" not in text
    assert "requirements-bootstrap.txt" in text


def test_build_tools_keep_pkg_resources_for_torch_2_0_1_cpp_extension():
    # torch 2.0.1's cpp_extension does `from pkg_resources import packaging`; flash-attn 1.0.9
    # is built with it. setuptools 82+ removed pkg_resources, which would fail that build.
    reqs = _requirements("requirements-scgpt-build.txt")
    (spec,) = [s for s in reqs if s.startswith("setuptools==")]
    assert int(spec.split("==")[1].split(".")[0]) < 82, spec


def test_geneformer_image_installs_the_training_toolkit_from_hashed_locks():
    # `uv pip install sagemaker-training` was unpinned and unhashed (outside uv.lock).
    text = (SM / "Dockerfile").read_text().replace("\\\n", " ")
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "uv pip install sagemaker-training" not in code
    assert "uv pip install --require-hashes --no-build-isolation -r /tmp/locks/requirements-toolkit.txt" in code
    assert "uv pip install --require-hashes --no-deps -r /tmp/locks/requirements-toolkit-build.txt" in code
    reqs = _requirements("requirements-toolkit.txt")
    assert "sagemaker-training==5.1.4" in reqs


def test_every_uv_pip_install_in_the_geneformer_dockerfile_is_hashed_or_local():
    text = (SM / "Dockerfile").read_text().replace("\\\n", " ")
    for ln in text.splitlines():
        if ln.lstrip().startswith("#"):
            continue
        for part in re.split(r"&&", ln):
            if "uv pip install" not in part:
                continue
            hashed = "--require-hashes" in part
            # Local code only: the cloned Geneformer, and this package without its dependencies.
            local = "--no-deps" in part and bool(re.search(r"install --no-deps (/Geneformer|\.)\s*$", part.strip()))
            assert hashed or local, f"unhashed uv pip install: {part.strip()}"


def test_toolkit_lock_agrees_with_uv_lock_on_shared_packages():
    # The toolkit lock must not move a package the project is tested with.
    import tomllib

    uv_lock = tomllib.loads((SM.parent / "uv.lock").read_text())
    locked = {}
    for pkg in uv_lock["package"]:
        locked.setdefault(pkg["name"].lower().replace("_", "-"), set()).add(pkg["version"])
    for name in ("requirements-toolkit.txt", "requirements-toolkit-build.txt"):
        for spec in _requirements(name):
            pkg, _, ver = spec.partition("==")
            key = pkg.lower().replace("_", "-")
            if key in locked and key not in {"pip"}:
                assert ver in locked[key], f"{name}: {spec} differs from uv.lock {sorted(locked[key])}"
