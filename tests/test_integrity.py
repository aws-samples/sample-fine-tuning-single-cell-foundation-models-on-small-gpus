# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Input integrity: SHA256SUMS channels (T003) and pinned downloads (T012)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

from scfm_ft import datasets
from scfm_ft.integrity import (
    MANIFEST_NAME,
    IntegrityError,
    download_verified,
    parse_manifest_pins,
    parse_sha256sums,
    sha256_file,
    verify_channel,
    write_sha256sums,
)

_SM_ENTRY = Path(__file__).resolve().parents[1] / "sagemaker" / "sm_entry.py"
_spec = importlib.util.spec_from_file_location("sm_entry", _SM_ENTRY)
assert _spec and _spec.loader
sm_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm_entry)

H = "a" * 64


def _channel(tmp_path, files: dict[str, bytes]):
    for rel, data in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return tmp_path


# --- manifest parsing -----------------------------------------------------------------
def test_parse_accepts_sha256sum_output_including_binary_marker_and_dot_slash():
    text = f"{H}  data.loom\n{'b' * 64} *./sub/model.pt\n# comment\n\n"
    assert parse_sha256sums(text) == {"data.loom": H, "sub/model.pt": "b" * 64}


@pytest.mark.parametrize(
    "line",
    [f"{H}  ../escape", f"{H}  /etc/passwd", f"{H}  a/../../b", "not a hash  file", f"{H[:-1]}  short"],
)
def test_parse_rejects_unsafe_or_malformed_lines(line):
    with pytest.raises(IntegrityError):
        parse_sha256sums(line)


def test_parse_rejects_duplicates():
    with pytest.raises(IntegrityError, match="twice"):
        parse_sha256sums(f"{H}  a\n{H}  ./a\n")


# --- channel verification -------------------------------------------------------------
def test_write_then_verify_round_trips_nested_files(tmp_path):
    root = _channel(tmp_path, {"data.loom": b"x", "t.dataset/part-0.arrow": b"yy"})
    manifest = write_sha256sums(root)
    assert manifest.name == MANIFEST_NAME and "t.dataset/part-0.arrow" in manifest.read_text()
    assert MANIFEST_NAME not in manifest.read_text()  # the manifest does not list itself
    result = verify_channel(root, require=True)
    assert (result.manifest_present, result.files_verified) == (True, 2)


def test_a_modified_file_fails_even_without_require(tmp_path):
    root = _channel(tmp_path, {"data.loom": b"original"})
    write_sha256sums(root)
    (root / "data.loom").write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="data.loom: SHA-256 does not match"):
        verify_channel(root)


def test_a_listed_but_missing_file_fails(tmp_path):
    root = _channel(tmp_path, {"data.loom": b"x"})
    (root / MANIFEST_NAME).write_text(f"{H}  gone.loom\n")
    with pytest.raises(IntegrityError, match="gone.loom: listed .* but missing"):
        verify_channel(root)


def test_unlisted_files_warn_normally_but_fail_when_required(tmp_path, capsys):
    root = _channel(tmp_path, {"data.loom": b"x"})
    write_sha256sums(root)
    (root / "extra.loom").write_bytes(b"sneaked in")
    verify_channel(root)  # default: warn only, so existing workflows keep working
    assert "WARNING" in capsys.readouterr().out
    with pytest.raises(IntegrityError, match="not listed in SHA256SUMS: extra.loom"):
        verify_channel(root, require=True)


def test_missing_manifest_warns_by_default_and_fails_when_required(tmp_path, capsys):
    root = _channel(tmp_path, {"data.loom": b"x"})
    assert verify_channel(root).manifest_present is False
    assert "unverified" in capsys.readouterr().out
    with pytest.raises(IntegrityError, match="required but missing"):
        verify_channel(root, require=True)


def test_symlinks_in_a_channel_are_refused(tmp_path):
    root = tmp_path / "chan"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    (root / "link.bin").symlink_to(outside)
    (root / MANIFEST_NAME).write_text(f"{sha256_file(outside)}  link.bin\n")
    with pytest.raises(IntegrityError, match="symlink or points outside"):
        verify_channel(root)


def test_checksums_of_an_empty_directory_is_an_error(tmp_path):
    with pytest.raises(IntegrityError, match="No files"):
        write_sha256sums(tmp_path)


# --- pinned downloads -----------------------------------------------------------------
class _Resp(io.BytesIO):
    def __init__(self, data: bytes, final_url: str):
        super().__init__(data)
        self._final = final_url
        self.bytes_read = 0

    def read(self, n=-1):
        block = super().read(n)
        self.bytes_read += len(block)
        return block

    def geturl(self):
        return self._final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _opener(data: bytes, final_url: str | None = None):
    def opener(url, timeout=None):
        return _Resp(data, final_url or url)

    return opener


def _pin(data: bytes):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def test_download_keeps_a_file_that_matches_its_pin(tmp_path):
    data = b"genuine bytes" * 1000
    dest = download_verified("https://x.test/f", tmp_path / "f.h5ad", opener=_opener(data), **_pin(data))
    assert dest.read_bytes() == data
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_tampered_content_and_leaves_nothing_behind(tmp_path):
    good = b"genuine bytes" * 1000
    evil = b"tampered bytes" * 1000
    pin = _pin(good) | {"size": len(evil)}  # same size, different content
    with pytest.raises(IntegrityError, match="does not match its pinned SHA-256"):
        download_verified("https://x.test/f", tmp_path / "f.h5ad", opener=_opener(evil), **pin)
    assert list(tmp_path.iterdir()) == []


def test_download_rejects_a_truncated_file(tmp_path):
    data = b"z" * 5000
    with pytest.raises(IntegrityError, match="does not match"):
        download_verified("https://x.test/f", tmp_path / "f", opener=_opener(data[:4000]), **_pin(data))
    assert list(tmp_path.iterdir()) == []


def test_download_stops_reading_at_the_pinned_size(tmp_path):
    data = b"z" * 5000
    big = _Resp(data * 10_000, "https://x.test/f")  # a 50 MB response for a 5 KB pin
    with pytest.raises(IntegrityError, match="larger than the pinned size"):
        download_verified("https://x.test/f", tmp_path / "f", opener=lambda *a, **k: big, **_pin(data))
    assert big.bytes_read < len(data) * 10_000 // 10  # aborted early, not drained to the end
    assert list(tmp_path.iterdir()) == []


def test_download_refuses_http_and_https_to_http_redirects(tmp_path):
    data = b"abc"
    with pytest.raises(IntegrityError, match="non-HTTPS"):
        download_verified("http://x.test/f", tmp_path / "f", opener=_opener(data), **_pin(data))
    with pytest.raises(IntegrityError, match="redirected to a non-HTTPS"):
        download_verified(
            "https://x.test/f", tmp_path / "f", opener=_opener(data, "http://evil.test/f"), **_pin(data)
        )
    assert list(tmp_path.iterdir()) == []


# --- PBMC3k pins ----------------------------------------------------------------------
def test_pbmc3k_pins_are_https_hashes_and_not_a_moving_branch():
    assert set(datasets.PBMC3K_FILES) == {"pbmc3k_raw.h5ad", "pbmc3k_processed.h5ad"}
    for name, pin in datasets.PBMC3K_FILES.items():
        assert pin.url.startswith("https://"), name
        assert re.fullmatch(r"[0-9a-f]{64}", pin.sha256), name
        assert pin.size > 1_000_000, name
        assert "/main/" not in pin.url and "/master/" not in pin.url, f"{name} must not track a branch"
    assert re.search(r"/[0-9a-f]{40}/", datasets.PBMC3K_FILES["pbmc3k_processed.h5ad"].url)


def test_pbmc3k_fetch_rejects_content_that_does_not_match_the_pins(tmp_path):
    with pytest.raises(IntegrityError):
        datasets.fetch_pbmc3k_files(tmp_path, opener=_opener(b"not the real dataset"))
    assert list(tmp_path.iterdir()) == []


def test_load_pbmc3k_verifies_before_scanpy_reads_anything():
    src = (Path(datasets.__file__)).read_text()
    assert src.index("fetch_pbmc3k_files(") < src.index("sc.datasets.pbmc3k()")
    assert "sc.datasets.pbmc3k_processed()" in src


# --- sm_entry wiring ------------------------------------------------------------------
def _args(**kw):
    ns = argparse.Namespace(
        data_source="channel", train_channel="", backend="geneformer", require_input_hashes="false"
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_entry_point_verifies_the_train_channel_before_use(tmp_path, monkeypatch):
    root = _channel(tmp_path, {"data.loom": b"x"})
    write_sha256sums(root)
    (root / "data.loom").write_bytes(b"changed after the manifest was written")
    monkeypatch.delenv("SM_CHANNEL_WEIGHTS", raising=False)
    with pytest.raises(IntegrityError):
        sm_entry._verify_input_channels(_args(train_channel=str(root)))


def test_entry_point_also_verifies_the_scgpt_weights_channel(tmp_path, monkeypatch):
    train = _channel(tmp_path / "train", {"data.loom": b"x"})
    weights = _channel(tmp_path / "weights", {"scGPT_human/model.pt": b"weights"})
    write_sha256sums(train)
    write_sha256sums(weights)
    monkeypatch.setenv("SM_CHANNEL_WEIGHTS", str(weights))
    ns = _args(train_channel=str(train), backend="scgpt", require_input_hashes="true")
    sm_entry._verify_input_channels(ns)  # clean channels pass
    (weights / "scGPT_human/model.pt").write_bytes(b"swapped checkpoint")
    with pytest.raises(IntegrityError, match="model.pt"):
        sm_entry._verify_input_channels(ns)


def test_require_flag_is_parsed_as_a_string_bool(monkeypatch):
    monkeypatch.setattr("sys.argv", ["sm_entry.py", "--require-input-hashes", "true", "--max-cells", "5"])
    args = sm_entry._parse_args()
    assert sm_entry._truthy(args.require_input_hashes) is True and args.max_cells == 5
    monkeypatch.setattr("sys.argv", ["sm_entry.py"])
    assert sm_entry._truthy(sm_entry._parse_args().require_input_hashes) is False


def test_pbmc3k_source_skips_channel_verification(tmp_path):
    # Nothing to verify for the self-fetch (it checks its own pinned hashes), so a missing
    # channel directory must not fail a --data-source pbmc3k job.
    sm_entry._verify_input_channels(_args(data_source="pbmc3k", train_channel=str(tmp_path / "none")))


# --- manifest pins (T016): the manifest itself is checked against a hash kept elsewhere ---
def _staged(tmp_path):
    (tmp_path / "data.loom").write_bytes(b"cells")
    manifest = write_sha256sums(tmp_path)
    return manifest, sha256_file(manifest)


def test_a_matching_manifest_pin_passes(tmp_path):
    _, pin = _staged(tmp_path)
    assert verify_channel(tmp_path, manifest_sha256=pin).manifest_present


def test_a_swapped_file_and_manifest_is_caught_by_the_pin(tmp_path):
    _, pin = _staged(tmp_path)
    (tmp_path / "data.loom").write_bytes(b"poisoned")
    write_sha256sums(tmp_path)  # attacker rewrites the manifest to match
    verify_channel(tmp_path)  # without a pin the swap goes unnoticed
    with pytest.raises(IntegrityError, match="pinned hash"):
        verify_channel(tmp_path, manifest_sha256=pin)


def test_a_pin_makes_the_manifest_mandatory_and_complete(tmp_path):
    manifest, pin = _staged(tmp_path)
    (tmp_path / "extra.bin").write_bytes(b"x")
    with pytest.raises(IntegrityError, match="not listed"):
        verify_channel(tmp_path, manifest_sha256=pin)
    manifest.unlink()
    with pytest.raises(IntegrityError, match="pinned but missing"):
        verify_channel(tmp_path, manifest_sha256=pin)


def test_parse_manifest_pins():
    h1, h2 = "a" * 64, "B" * 64
    assert parse_manifest_pins("") == {}
    assert parse_manifest_pins(h1) == {"train": h1}
    assert parse_manifest_pins(f"train={h1},weights={h2}") == {"train": h1, "weights": "b" * 64}
    for bad in ("train=abc", f"cache={h1}", f"train={h1},train={h1}", "train", f"train {h1}"):
        with pytest.raises(IntegrityError):
            parse_manifest_pins(bad)
