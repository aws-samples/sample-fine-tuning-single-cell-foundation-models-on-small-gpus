# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Tests for the SageMaker entry point's input resolution and arg parsing.

These do not require the SageMaker SDK, network, or GPU — they exercise the pure
logic that maps a train channel + hyperparameters onto a run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load sagemaker/sm_entry.py as a module (it lives outside the package).
_SM_ENTRY = Path(__file__).resolve().parents[1] / "sagemaker" / "sm_entry.py"
_spec = importlib.util.spec_from_file_location("sm_entry", _SM_ENTRY)
assert _spec and _spec.loader
sm_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm_entry)


def test_resolve_input_prefers_tokenized_dataset(tmp_path: Path) -> None:
    (tmp_path / "pbmc3k.dataset").mkdir()
    (tmp_path / "data.loom").write_text("ignored")
    raw, tok = sm_entry._resolve_input(tmp_path, "data.loom")
    assert raw is None
    assert tok is not None and tok.name == "pbmc3k.dataset"


def test_resolve_input_named_raw_file(tmp_path: Path) -> None:
    (tmp_path / "data.loom").write_text("x")
    raw, tok = sm_entry._resolve_input(tmp_path, "data.loom")
    assert tok is None
    assert raw is not None and raw.name == "data.loom"


def test_resolve_input_glob_fallback(tmp_path: Path) -> None:
    (tmp_path / "cells.h5ad").write_text("x")
    raw, tok = sm_entry._resolve_input(tmp_path, "missing.loom")
    assert tok is None
    assert raw is not None and raw.name == "cells.h5ad"


def test_resolve_input_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sm_entry._resolve_input(tmp_path, "data.loom")


def test_arg_parsing_hyphenated_flags() -> None:
    old = sys.argv
    try:
        sys.argv = ["sm_entry.py", "--preset", "V2-104M", "--epochs", "3", "--num-crossval-splits", "0"]
        args = sm_entry._parse_args()
    finally:
        sys.argv = old
    assert args.preset == "V2-104M"
    assert args.epochs == 3
    assert args.num_crossval_splits == 0


def test_backend_arg_defaults_and_scgpt() -> None:
    old = sys.argv
    try:
        sys.argv = ["sm_entry.py"]
        default = sm_entry._parse_args()
        sys.argv = ["sm_entry.py", "--backend", "scgpt", "--preset", "scgpt-human"]
        scgpt = sm_entry._parse_args()
    finally:
        sys.argv = old
    assert default.backend == "geneformer"
    assert scgpt.backend == "scgpt"
    assert scgpt.scgpt_model_root == "/scgpt"
    # scGPT training knob defaults: flash-attn on, grad checkpointing off.
    assert scgpt.scgpt_flash_attn == "true"
    assert scgpt.scgpt_grad_checkpointing == "false"


def test_scgpt_flags_parse() -> None:
    old = sys.argv
    try:
        sys.argv = [
            "sm_entry.py", "--backend", "scgpt", "--preset", "scgpt-human",
            "--scgpt-flash-attn", "false", "--scgpt-grad-checkpointing", "true",
        ]
        args = sm_entry._parse_args()
    finally:
        sys.argv = old
    assert args.scgpt_flash_attn == "false"
    assert args.scgpt_grad_checkpointing == "true"


def test_resolve_input_raw_named_file(tmp_path: Path) -> None:
    (tmp_path / "cells.loom").write_text("x")
    raw = sm_entry._resolve_input_raw(tmp_path, "cells.loom")
    assert raw.name == "cells.loom"


def test_resolve_input_raw_glob_fallback(tmp_path: Path) -> None:
    (tmp_path / "cells.h5ad").write_text("x")
    raw = sm_entry._resolve_input_raw(tmp_path, "missing.loom")
    assert raw.name == "cells.h5ad"


def test_resolve_input_raw_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="scgpt backend"):
        sm_entry._resolve_input_raw(tmp_path, "data.loom")


def test_main_dispatches_scgpt_backend(tmp_path: Path, monkeypatch) -> None:
    """main() for --backend scgpt resolves the raw input, builds an scgpt Config,
    and drives the fake backend's validate_config -> tokenize -> finetune."""
    from scfm_ft.backends.base import FinetuneResult, TokenizeResult

    (tmp_path / "cells.loom").write_text("x")
    recorder: dict = {}

    class _Fake:
        name = "scgpt"

        def validate_config(self, cfg):
            recorder.setdefault("calls", []).append("validate_config")
            recorder["backend"] = cfg.model.backend

        def tokenize(self, cfg, input_path):
            recorder.setdefault("calls", []).append("tokenize")
            ds = cfg.tokenized_dir / "x_scgpt.npz"
            return TokenizeResult(dataset_path=ds)

        def finetune(self, cfg, dataset_path):
            recorder.setdefault("calls", []).append("finetune")
            recorder["dataset"] = dataset_path
            return FinetuneResult(model_dir=cfg.model_out_dir, metrics={"accuracy": 0.9})

    monkeypatch.setattr(sm_entry, "get_backend", lambda name: _Fake())
    argv = [
        "sm_entry.py",
        "--backend", "scgpt",
        "--preset", "scgpt-human",
        "--train-channel", str(tmp_path),
        "--train-file", "cells.loom",
        "--model-dir", str(tmp_path / "out"),
        "--epochs", "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    sm_entry.main()

    assert recorder["calls"] == ["validate_config", "tokenize", "finetune"]
    assert recorder["backend"] == "scgpt"
    assert recorder["dataset"].name == "x_scgpt.npz"


# --- manifest pins reach the channel check --------------------------------------------
def _channel_args(tmp_path: Path, **kw):
    import argparse

    return argparse.Namespace(
        require_input_hashes="true", data_source="channel", train_channel=str(tmp_path),
        backend="geneformer", manifest_sha256=kw.get("pins", ""),
    )


def test_a_pin_for_the_wrong_manifest_stops_the_job(tmp_path: Path) -> None:
    from scfm_ft.integrity import IntegrityError, write_sha256sums

    (tmp_path / "data.loom").write_bytes(b"x")
    write_sha256sums(tmp_path)
    with pytest.raises(IntegrityError, match="pinned hash"):
        sm_entry._verify_input_channels(_channel_args(tmp_path, pins="train=" + "0" * 64))


def test_the_right_pin_passes_and_a_pin_for_an_unchecked_channel_fails(tmp_path: Path) -> None:
    from scfm_ft.integrity import IntegrityError, sha256_file, write_sha256sums

    (tmp_path / "data.loom").write_bytes(b"x")
    pin = sha256_file(write_sha256sums(tmp_path))
    sm_entry._verify_input_channels(_channel_args(tmp_path, pins=f"train={pin}"))
    with pytest.raises(IntegrityError, match="weights"):
        sm_entry._verify_input_channels(_channel_args(tmp_path, pins=f"weights={pin}"))
