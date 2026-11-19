# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The launcher venv is installed from a hash-locked file (T017)."""

from __future__ import annotations

import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
LOCK = SCRIPTS / "requirements-launcher.txt"
WRAPPER = (SCRIPTS / "do-fine-tuning-job.sh").read_text()
# Comments may describe what is forbidden; only the commands count.
CODE = "\n".join(ln for ln in WRAPPER.splitlines() if not ln.lstrip().startswith("#"))


def _requirements() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in LOCK.read_text().replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out[line.split()[0]] = re.findall(r"--hash=sha256:([0-9a-f]{64})", line)
    return out


def test_every_launcher_requirement_is_pinned_exactly_and_hashed():
    reqs = _requirements()
    assert reqs
    for spec, hashes in reqs.items():
        # Environment markers (";") are allowed; everything else must be name==version.
        assert re.fullmatch(r"[A-Za-z0-9._-]+(\[[^\]]+\])?==[A-Za-z0-9.+!_-]+;?", spec) or "==" in spec, spec
        assert hashes, f"{spec} has no hash"


def test_the_lock_pins_the_v2_sdk_and_boto3():
    names = {spec.split("==")[0].lower() for spec in _requirements()}
    assert {"sagemaker", "boto3"} <= names
    assert "sagemaker==2.219.0" in _requirements()


def test_the_lock_has_no_urls_or_local_paths():
    text = LOCK.read_text()
    assert not re.search(r"^\s*(-e |git\+|https?://|file:|\.{1,2}/)", text, re.MULTILINE)


def test_the_wrapper_installs_only_from_the_hash_lock():
    assert 'LAUNCHER_REQS="${SCRIPT_DIR}/requirements-launcher.txt"' in WRAPPER
    installs = [ln.strip() for ln in CODE.splitlines() if re.search(r"\b(uv )?pip\"? install\b", ln)]
    assert installs
    for ln in installs:
        assert "--require-hashes" in ln and '-r "${LAUNCHER_REQS}"' in ln, ln


def test_the_wrapper_never_upgrades_pip_or_installs_a_bare_package():
    assert "--upgrade" not in CODE
    assert "SAGEMAKER_SDK_PIN" not in CODE
    assert not re.search(r'pip"? install[^\n]*\bboto3\b', CODE)


def test_a_venv_built_from_a_different_lock_is_recreated():
    assert ".requirements.sha256" in WRAPPER
    assert "recreating it" in WRAPPER
    # Only the venv this script owns is ever deleted.
    assert '"${LAUNCHER_VENV}" == "${REPO_ROOT}/.sm-launcher"' in WRAPPER
