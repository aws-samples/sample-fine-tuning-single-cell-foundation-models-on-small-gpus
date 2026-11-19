# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""scripts/fetch_weights.py checks every download against hashes pinned in the script (T002, T003)."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_weights.py"
_spec = importlib.util.spec_from_file_location("fetch_weights_under_test", SCRIPT)
assert _spec and _spec.loader
fw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw)


def _repo_files() -> set[str]:
    out = set()
    for cfg in fw.PRESETS.values():
        out |= {f"{cfg['model_dir']}/{f}" for f in fw.MODEL_FILES}
        out |= {f"{cfg['dict_subdir']}/{t.format(corpus=cfg['corpus'])}" for t in fw.DICT_FILES}
    return out


def test_every_file_the_script_can_download_has_a_pinned_hash():
    assert set(fw.PINNED_SHA256) == _repo_files()
    for path, digest in fw.PINNED_SHA256.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), path


def test_a_matching_file_passes(tmp_path, monkeypatch):
    f = tmp_path / "x.bin"
    f.write_bytes(b"weights")
    monkeypatch.setitem(fw.PINNED_SHA256, "a/b.bin", hashlib.sha256(b"weights").hexdigest())
    fw.verify(str(f), "a/b.bin")


def test_a_changed_file_is_refused_with_advice(tmp_path, monkeypatch):
    f = tmp_path / "x.bin"
    f.write_bytes(b"tampered")
    monkeypatch.setitem(fw.PINNED_SHA256, "a/b.bin", hashlib.sha256(b"weights").hexdigest())
    with pytest.raises(SystemExit) as e:
        fw.verify(str(f), "a/b.bin")
    assert "does not match the pinned" in str(e.value) and "cache" in str(e.value)


def test_a_file_with_no_pinned_hash_is_refused(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"anything")
    with pytest.raises(SystemExit) as e:
        fw.verify(str(f), "geneformer/not_a_pinned_file.pkl")
    assert "No pinned SHA-256" in str(e.value)


def test_a_bad_download_is_never_copied_into_the_repo(tmp_path, monkeypatch):
    bad = tmp_path / "cache_blob"
    bad.write_bytes(b"not the weights")
    monkeypatch.setattr(fw, "hf_hub_download", lambda **kw: str(bad))
    dest = tmp_path / "vendor" / "Geneformer-V1-10M" / "model.safetensors"
    with pytest.raises(SystemExit):
        fw._download("Geneformer-V1-10M/model.safetensors", str(dest))
    assert not dest.exists()


def test_main_stops_on_a_bad_dictionary_before_writing_it(tmp_path, monkeypatch):
    bad = tmp_path / "cache_blob"
    bad.write_bytes(b"not a dictionary")
    monkeypatch.setattr(fw, "hf_hub_download", lambda **kw: str(bad))
    real_verify = fw.verify

    def verify(path, repo_file):
        if repo_file.rsplit("/", 1)[-1] in fw.MODEL_FILES:  # let the model files through
            return
        real_verify(path, repo_file)

    monkeypatch.setattr(fw, "verify", verify)
    repo = tmp_path / "repo"
    monkeypatch.setattr(sys, "argv", ["fetch_weights.py", "--preset", "V1-10M", "--repo", str(repo)])
    with pytest.raises(SystemExit):
        fw.main()
    assert not list(repo.rglob("*.pkl"))
