# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Dispatch wiring tests.

Prove that the shared layer (the CLI's ``run`` command) drives the pipeline
``validate_config -> tokenize -> finetune -> evaluate`` strictly through the backend
registry, in order, passing the tokenized dataset path from tokenize into finetune —
with NO GPU, NO real model, NO ML stack. A fake backend recorded into the registry
stands in for a real one, so this guards the glue that makes "every shared layer
talks only to the interface" true.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from typer.testing import CliRunner

from scfm_ft.backends.base import FinetuneResult, TokenizeResult


class _FakeBackend:
    name = "fake"

    def __init__(self, recorder: dict):
        self._rec = recorder

    def supported_presets(self):
        return ["V2-104M"]

    def resolve_checkpoint(self, cfg):
        return Path("/fake/ckpt")

    def validate_config(self, cfg):
        self._rec.setdefault("calls", []).append("validate_config")

    def tokenize(self, cfg, input_path):
        self._rec.setdefault("calls", []).append("tokenize")
        self._rec["tokenize_input"] = input_path
        ds = cfg.tokenized_dir / "fake.dataset"
        return TokenizeResult(dataset_path=ds)

    def finetune(self, cfg, dataset_path):
        self._rec.setdefault("calls", []).append("finetune")
        self._rec["finetune_dataset"] = dataset_path
        return FinetuneResult(model_dir=cfg.model_out_dir, metrics={"accuracy": 0.9})

    def evaluate(self, cfg):
        self._rec.setdefault("calls", []).append("evaluate")
        return {"accuracy": 0.9, "macro_f1": 0.88}

    def default_instance_type(self, preset):
        return "ml.g5.xlarge"

    def docker_layers(self):
        return ["fake layer"]


def _config_file(tmp_path: Path) -> Path:
    body = textwrap.dedent(f"""
        name: wire
        output_dir: {tmp_path / "out"}
        input: {{ h5ad_path: {tmp_path / "x.loom"} }}
        model: {{ backend: geneformer, preset: V2-104M }}
    """)
    (tmp_path / "x.loom").write_text("placeholder")
    p = tmp_path / "cfg.yaml"
    p.write_text(body)
    return p


def test_cli_run_dispatches_through_registry(tmp_path, monkeypatch):
    recorder: dict = {}

    # Patch the registry get_backend as the CLI resolves it (imported inside the
    # command body from scfm_ft.backends).
    from scfm_ft import backends

    monkeypatch.setattr(backends, "get_backend", lambda name: _FakeBackend(recorder))

    from scfm_ft import cli

    cfg_path = _config_file(tmp_path)
    result = CliRunner().invoke(cli.app, ["run", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output

    # Order: validate_config, then tokenize -> finetune -> evaluate.
    assert recorder["calls"] == ["validate_config", "tokenize", "finetune", "evaluate"]
    # The dataset path from tokenize is what finetune received.
    assert recorder["finetune_dataset"].name == "fake.dataset"
    # tokenize was handed the configured input path.
    assert recorder["tokenize_input"].name == "x.loom"


def test_cli_tokenize_only_dispatches(tmp_path, monkeypatch):
    recorder: dict = {}
    from scfm_ft import backends

    monkeypatch.setattr(backends, "get_backend", lambda name: _FakeBackend(recorder))

    from scfm_ft import cli

    cfg_path = _config_file(tmp_path)
    result = CliRunner().invoke(cli.app, ["tokenize", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert recorder["calls"] == ["validate_config", "tokenize"]
