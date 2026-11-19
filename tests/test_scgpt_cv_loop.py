# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Offline run of the scGPT fold loop with a tiny stand-in model.

scGPT itself only exists inside the training image, so this injects a fake
``scgpt.model.TransformerModel`` and a fake vocab to exercise the real training/eval/
aggregation code path on CPU in a second.
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn


class _FakeTransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, n_cls, **_kw):
        super().__init__()
        self.emb = nn.Embedding(ntoken, 8)
        self.cls_decoder = nn.Linear(8, n_cls)

    def forward(self, gids, vals, src_key_padding_mask=None, CLS=True):
        h = (self.emb(gids) * vals.unsqueeze(-1)).mean(1)
        return {"cls_output": self.cls_decoder(h)}


class _FakeVocab(dict):
    def set_default_index(self, _i):
        pass


@pytest.fixture
def fake_scgpt(monkeypatch):
    model_mod = types.ModuleType("scgpt.model")
    model_mod.TransformerModel = _FakeTransformerModel
    monkeypatch.setitem(sys.modules, "scgpt", types.ModuleType("scgpt"))
    monkeypatch.setitem(sys.modules, "scgpt.model", model_mod)

    from scfm_ft.backends.scgpt import backend

    vocab = _FakeVocab({**{f"g{i}": i for i in range(50)}, "<pad>": 50})
    monkeypatch.setattr(backend, "_load_gene_vocab", lambda _d: vocab)


def _cfg(n_splits, class_weighting="none", seed=42):
    ns = types.SimpleNamespace
    return ns(
        name="dry",
        training=ns(
            num_crossval_splits=n_splits,
            split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1},
            seed=seed,
            batch_size=16,
            learning_rate=1e-2,
            epochs=2,
        ),
        scgpt=ns(
            n_input_bins=51,
            use_flash_attn=False,
            do_mvc=False,
            freeze_backbone=False,
            grad_checkpointing=False,
            max_seq_len=20,
            class_weighting=class_weighting,
        ),
    )


def _data():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.array(["A", "B", "C"]), [60, 30, 10])
    values = rng.random((100, 40)).astype("float32")
    return np.arange(40), values, labels


@pytest.mark.parametrize("n_splits,expected_total", [(1, 10), (5, 100)])
def test_fold_loop_runs_and_aggregates(fake_scgpt, tmp_path, n_splits, expected_total):
    from scfm_ft.backends.base import METRIC_ACCURACY, METRIC_CONF_MATRIX, METRIC_MACRO_F1
    from scfm_ft.backends.scgpt._train import train_scgpt_classifier

    gene_ids, values, labels = _data()
    torch.save({}, tmp_path / "model.pt")  # stub checkpoint (the fake model has no encoder)
    metrics, conf, label_map = train_scgpt_classifier(
        tmp_path, gene_ids, values, labels, 50, _cfg(n_splits), tmp_path
    )

    assert 0.0 <= metrics[METRIC_ACCURACY] <= 1.0
    assert 0.0 <= metrics[METRIC_MACRO_F1] <= 1.0
    # Single split: 10% test. 5-fold: every one of the 100 cells held out once.
    assert int(conf.values.sum()) == expected_total
    assert label_map == {0: "A", 1: "B", 2: "C"}
    assert (tmp_path / "dry_scgpt_finetuned.pt").exists()

    if n_splits == 5:
        assert metrics["n_folds"] == 5
        assert len(metrics["fold_accuracy"]) == 5
        assert len(metrics["fold_macro_f1"]) == 5
        assert metrics[METRIC_ACCURACY] == pytest.approx(np.mean(metrics["fold_accuracy"]))
    else:
        assert "n_folds" not in metrics

    # The backend writes everything except the confusion matrix to JSON.
    json.dumps({k: v for k, v in metrics.items() if k != METRIC_CONF_MATRIX})


@pytest.mark.parametrize("best_model_present", [False, True])
def test_missing_model_pt_raises_instead_of_training_from_scratch(
    fake_scgpt, tmp_path, best_model_present
):
    from scfm_ft.backends.scgpt._train import train_scgpt_classifier

    if best_model_present:
        (tmp_path / "best_model.pt").write_bytes(b"")  # scGPT's download name
    gene_ids, values, labels = _data()
    with pytest.raises(FileNotFoundError, match="model.pt") as exc:
        train_scgpt_classifier(tmp_path, gene_ids, values, labels, 50, _cfg(1), tmp_path)
    assert ("rename it to model.pt" in str(exc.value)) == best_model_present


@pytest.mark.parametrize("mode", ["balanced", "sqrt"])
def test_fold_loop_runs_with_class_weighting(fake_scgpt, tmp_path, capsys, mode):
    from scfm_ft.backends.base import METRIC_MACRO_F1
    from scfm_ft.backends.scgpt._train import train_scgpt_classifier

    gene_ids, values, labels = _data()
    torch.save({}, tmp_path / "model.pt")
    metrics, _conf, _labels = train_scgpt_classifier(
        tmp_path, gene_ids, values, labels, 50, _cfg(5, mode), tmp_path
    )
    out = capsys.readouterr().out
    # One weighting line per fold, naming every class, with the rarest weighted highest.
    lines = [l for l in out.splitlines() if l.startswith(f"[scgpt] class weighting ({mode})")]
    assert len(lines) == 5
    w = dict(kv.split("=") for kv in lines[0].split(": ", 1)[1].split(", "))
    assert float(w["C"]) > float(w["B"]) > float(w["A"])
    assert 0.0 <= metrics[METRIC_MACRO_F1] <= 1.0


def _run(tmp_path, seed):
    from scfm_ft.backends.scgpt._train import train_scgpt_classifier

    tmp_path.mkdir(parents=True, exist_ok=True)
    gene_ids, values, labels = _data()
    torch.save({}, tmp_path / "model.pt")
    metrics, _conf, _labels = train_scgpt_classifier(
        tmp_path, gene_ids, values, labels, 50, _cfg(5, seed=seed), tmp_path
    )
    return metrics


def test_same_seed_reproduces_the_run(fake_scgpt, tmp_path):
    """Seeding covers head init, dropout and batch order, not just the folds."""
    a = _run(tmp_path / "a", seed=7)
    b = _run(tmp_path / "b", seed=7)
    assert a["fold_accuracy"] == b["fold_accuracy"]
    assert a["fold_macro_f1"] == b["fold_macro_f1"]


def test_each_fold_is_seeded_from_the_run_seed(fake_scgpt, tmp_path, monkeypatch):
    """training.seed seeds training (not just the folds), offset by the fold index."""
    from scfm_ft.backends.scgpt import _train

    seeds = []
    orig = _train._seed_everything
    monkeypatch.setattr(_train, "_seed_everything", lambda s: (seeds.append(s), orig(s)))
    _run(tmp_path / "a", seed=7)
    assert seeds == [7, 8, 9, 10, 11]


@pytest.mark.parametrize("n_splits", [1, 5])
def test_privacy_audit_runs_in_the_real_fold_loop(fake_scgpt, tmp_path, n_splits):
    """With training.privacy_audit, the loop scores training and held-out cells with the
    trained model and writes dry_privacy_audit.json (one audit per fold under CV), with
    per-donor results when groups are given."""
    from scfm_ft.backends.scgpt._train import train_scgpt_classifier

    gene_ids, values, labels = _data()
    groups = np.array([f"d{i % 10}" for i in range(100)])
    torch.save({}, tmp_path / "model.pt")
    cfg = _cfg(n_splits)
    cfg.training.privacy_audit = True
    cfg.training.privacy_audit_max_cells = 0
    cfg.classifier = types.SimpleNamespace(group_column="donor_id")
    metrics, _conf, _map = train_scgpt_classifier(
        tmp_path, gene_ids, values, labels, 50, cfg, tmp_path, groups=groups
    )
    rep = json.loads((tmp_path / "dry_privacy_audit.json").read_text())
    folds = rep["folds"] if n_splits > 1 else [rep]
    assert len(folds) == n_splits
    for fold in folds:
        assert fold["cells"]["n_members"] + fold["cells"]["n_nonmembers"] == 100
        assert 0.0 <= fold["cells"]["auc"] <= 1.0
        assert "error" not in fold["by_group"]  # a grouped split: no donor on both sides
    assert 0.0 <= metrics["privacy_audit_auc"] <= 1.0
