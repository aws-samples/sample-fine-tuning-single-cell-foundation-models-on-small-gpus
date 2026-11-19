# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Pipeline wiring tests.

These prove the tokenize -> finetune -> evaluate stages chain together correctly
and pass the right arguments through, WITHOUT installing the multi-GB geneformer +
torch stack. The ``geneformer`` package is replaced with a lightweight fake that
records how it was called and produces the on-disk artifacts the next stage expects.

The real end-to-end run (actual model weights, actual training) is exercised
separately once Geneformer is cloned via git-lfs and installed -- see README /
docs/SAGEMAKER.md. This suite guards the glue code that we CAN verify cheaply.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("loompy")

from scfm_ft.config import Config
from scfm_ft.data import make_sample_data


def _install_fake_geneformer(monkeypatch, recorder: dict):
    """Register a fake ``geneformer`` module in sys.modules."""
    mod = types.ModuleType("geneformer")

    class FakeTokenizer:
        def __init__(self, **kwargs):
            recorder["tokenizer_init"] = kwargs

        def tokenize_data(self, data_directory, output_directory, output_prefix, file_format):
            recorder["tokenize_call"] = {
                "data_directory": data_directory,
                "output_directory": output_directory,
                "output_prefix": output_prefix,
                "file_format": file_format,
            }
            # Produce the .dataset directory the finetune stage looks for.
            Path(output_directory, f"{output_prefix}.dataset").mkdir(parents=True, exist_ok=True)

    class FakeClassifier:
        def __init__(self, **kwargs):
            recorder.setdefault("classifier_init", []).append(kwargs)
            self._out = None
            self._prefix = None

        def prepare_data(self, input_data_file, output_directory, output_prefix, **kwargs):
            recorder["prepare_data"] = {
                "input_data_file": input_data_file,
                "output_directory": output_directory,
                "output_prefix": output_prefix,
                **kwargs,
            }
            self._out = Path(output_directory)
            self._prefix = output_prefix
            # Mirror Geneformer's real naming: with a held-out test fraction it writes
            # {prefix}_labeled_train + {prefix}_labeled_test; otherwise {prefix}_labeled.
            for suffix in ("_labeled", "_labeled_train", "_labeled_test"):
                (self._out / f"{output_prefix}{suffix}.dataset").mkdir(parents=True, exist_ok=True)
            (self._out / f"{output_prefix}_id_class_dict.pkl").write_bytes(b"\x80\x04N.")

        def validate(self, **kwargs):
            recorder["validate"] = kwargs
            import pandas as pd

            return {
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "conf_matrix": pd.DataFrame([[1, 0], [0, 1]]),
            }

        def train_all_data(self, **kwargs):
            recorder["train_all_data"] = kwargs

        def evaluate_saved_model(self, **kwargs):
            recorder["evaluate_saved_model"] = kwargs
            import pandas as pd

            return {"accuracy": 0.5, "conf_matrix": pd.DataFrame([[1, 0], [0, 1]])}

        def plot_conf_mat(self, **kwargs):
            recorder["plot_conf_mat"] = kwargs

    mod.TranscriptomeTokenizer = FakeTokenizer
    mod.Classifier = FakeClassifier
    monkeypatch.setitem(sys.modules, "geneformer", mod)


def _smoke_config(tmp_path: Path, h5ad: Path) -> Config:
    return Config.model_validate(
        {
            "name": "wire",
            "output_dir": str(tmp_path / "out"),
            "input": {"h5ad_path": str(h5ad)},
            "model": {"preset": "V1-10M", "geneformer_repo": str(tmp_path / "repo")},
            "classifier": {"kind": "cell", "label_column": "cell_type"},
            "training": {"epochs": 1, "batch_size": 2, "num_crossval_splits": 1},
            "runtime": {"nproc": 1, "ngpu": 0},
        }
    )


def test_full_pipeline_wiring(tmp_path, monkeypatch):
    recorder: dict = {}
    _install_fake_geneformer(monkeypatch, recorder)

    # Make the fake pretrained model path exist so finetune's guard passes.
    (tmp_path / "repo" / "Geneformer-V1-10M").mkdir(parents=True)

    h5ad = make_sample_data(tmp_path / "wire.h5ad", n_cells=40, n_classes=2, seed=3)
    cfg = _smoke_config(tmp_path, h5ad)

    from scfm_ft.evaluate import evaluate
    from scfm_ft.finetune import finetune
    from scfm_ft.tokenize import tokenize

    # 1. tokenize
    ds = tokenize(cfg)
    assert ds.exists()
    assert recorder["tokenizer_init"]["model_input_size"] == 2048  # V1 input size
    assert recorder["tokenize_call"]["file_format"] == "loom"

    # 2. finetune (split==1 -> validate path, returns metrics)
    result = finetune(cfg, dataset_path=ds)
    assert result.labeled_dataset.exists()
    assert result.id_class_dict.exists()
    assert result.metrics["accuracy"] == 0.5
    # cell_state_dict wired from classifier config
    init = recorder["classifier_init"][0]
    assert init["cell_state_dict"] == {"state_key": "cell_type", "states": "all"}
    assert init["model_version"] == "V1"
    # training args mapped correctly: validate() fine-tunes FROM the pretrained base
    # model and writes to output_directory.
    assert recorder["validate"]["model_directory"].endswith("Geneformer-V1-10M")
    assert recorder["validate"]["output_directory"].endswith("model")

    # 3. evaluate (reads persisted metrics + conf_matrix from finetune; renders plot)
    metrics = evaluate(cfg)
    assert metrics["accuracy"] == 0.5
    assert "evaluate_saved_model" not in recorder  # must NOT re-run inference
    assert "plot_conf_mat" in recorder
    # the eval metrics JSON was written
    assert (cfg.eval_dir / "wire_eval_metrics.json").exists()


def test_finetune_errors_without_pretrained_model(tmp_path, monkeypatch):
    recorder: dict = {}
    _install_fake_geneformer(monkeypatch, recorder)
    # Note: do NOT create the model dir, so the guard should fire.
    h5ad = make_sample_data(tmp_path / "w2.h5ad", n_cells=20, n_classes=2)
    cfg = _smoke_config(tmp_path, h5ad)

    from scfm_ft.finetune import finetune
    from scfm_ft.tokenize import tokenize

    ds = tokenize(cfg)
    with pytest.raises(FileNotFoundError, match="Pretrained model not found"):
        finetune(cfg, dataset_path=ds)


def test_train_all_data_path(tmp_path, monkeypatch):
    recorder: dict = {}
    _install_fake_geneformer(monkeypatch, recorder)
    (tmp_path / "repo" / "Geneformer-V1-10M").mkdir(parents=True)
    h5ad = make_sample_data(tmp_path / "w3.h5ad", n_cells=20, n_classes=2)
    cfg = _smoke_config(tmp_path, h5ad)
    cfg.training.num_crossval_splits = 0  # train on all data -> train_all_data path

    from scfm_ft.finetune import finetune
    from scfm_ft.tokenize import tokenize

    ds = tokenize(cfg)
    result = finetune(cfg, dataset_path=ds)
    assert "train_all_data" in recorder
    assert "validate" not in recorder
    assert result.metrics is None


def test_group_column_splits_geneformer_by_donor(tmp_path, monkeypatch):
    """With classifier.group_column, whole donors go to test (prepare_data) and to eval
    (validate) through Geneformer's split_id_dict, and the split is written to disk."""
    import json

    import scfm_ft.finetune as ft

    recorder: dict = {}
    _install_fake_geneformer(monkeypatch, recorder)
    (tmp_path / "repo" / "Geneformer-V1-10M").mkdir(parents=True)
    h5ad = make_sample_data(tmp_path / "wire.h5ad", n_cells=40, n_classes=2, seed=3)
    cfg = _smoke_config(tmp_path, h5ad)
    cfg.classifier.group_column = "donor_id"

    donors = [f"d{i % 8}" for i in range(40)]
    labels = ["class_0" if i % 2 else "class_1" for i in range(40)]
    monkeypatch.setattr(
        ft, "_dataset_column", lambda path, col: donors if col == "donor_id" else labels
    )

    from scfm_ft.tokenize import tokenize

    ds = tokenize(cfg)
    assert recorder["tokenizer_init"]["custom_attr_name_dict"] == {
        "cell_type": "cell_type",
        "donor_id": "donor_id",
    }
    ft.finetune(cfg, dataset_path=ds)

    test_split = recorder["prepare_data"]["split_id_dict"]
    eval_split = recorder["validate"]["split_id_dict"]
    assert test_split["attr_key"] == eval_split["attr_key"] == "donor_id"
    assert set(test_split["train"]) == set(eval_split["train"]) | set(eval_split["eval"])
    assert not set(test_split["test"]) & set(test_split["train"])
    assert not set(eval_split["train"]) & set(eval_split["eval"])
    assert set(test_split["train"]) | set(test_split["test"]) == set(donors)

    record = json.loads((cfg.model_out_dir / "wire_group_split.json").read_text())
    assert record["group_column"] == "donor_id"
    assert sum(s["n_cells"] for s in record["splits"].values()) == 40


def test_geneformer_refuses_grouped_five_fold(tmp_path, monkeypatch):
    import scfm_ft.finetune as ft

    _install_fake_geneformer(monkeypatch, {})
    (tmp_path / "repo" / "Geneformer-V1-10M").mkdir(parents=True)
    h5ad = make_sample_data(tmp_path / "wire.h5ad", n_cells=40, n_classes=2, seed=3)
    cfg = _smoke_config(tmp_path, h5ad)
    cfg.classifier.group_column = "donor_id"
    cfg.training.num_crossval_splits = 5
    monkeypatch.setattr(ft, "_dataset_column", lambda path, col: ["d0"] * 40)
    from scfm_ft.tokenize import tokenize

    with pytest.raises(ValueError, match="Grouped 5-fold"):
        ft.finetune(cfg, dataset_path=tokenize(cfg))
