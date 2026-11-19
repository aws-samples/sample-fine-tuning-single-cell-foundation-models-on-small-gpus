# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Membership-inference audit metrics (T013). Pure NumPy/sklearn: no model needed."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scfm_ft.privacy import (
    advantage,
    attack_metrics,
    audit,
    cap_indices,
    per_cell_loss,
    roc_auc,
    summary_lines,
    tpr_at_fpr,
    write_report,
)


def test_per_cell_loss_is_cross_entropy_on_the_true_class():
    logits = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    labels = np.array([0, 0, 1])
    loss, prob, correct = per_cell_loss(logits, labels)
    expected_p = np.array([np.exp(2) / (np.exp(2) + 1), 1 / (1 + np.exp(3)), 0.5])
    np.testing.assert_allclose(prob, expected_p)
    np.testing.assert_allclose(loss, -np.log(expected_p))
    assert correct.tolist() == [True, False, False]  # a tie goes to class 0


def test_auc_is_the_probability_a_member_outscores_a_non_member():
    assert roc_auc([3, 4], [1, 2]) == 1.0
    assert roc_auc([1, 2], [3, 4]) == 0.0
    assert roc_auc([1, 3], [2, 2]) == 0.5
    assert roc_auc([2], [2]) == 0.5  # ties count half
    with pytest.raises(ValueError):
        roc_auc([], [1])


def test_tpr_at_low_fpr_and_advantage():
    members = np.array([10, 9, 8, 0.5, 0])
    nonmembers = np.arange(100) / 100.0  # 0.00 .. 0.99
    # Above every non-member: 3 of 5 members, with no false positives at all.
    assert tpr_at_fpr(members, nonmembers, 0.01) == pytest.approx(0.6)
    assert advantage(members, nonmembers) == pytest.approx(0.6)
    assert advantage([0, 1], [0, 1]) == pytest.approx(0.0)


def _losses(rng, n, overfit):
    """Member losses near 0 when the model memorised them, else the same as non-members."""
    nonmember = rng.exponential(1.0, n)
    member = rng.exponential(0.05, n) if overfit else rng.exponential(1.0, n)
    return member, nonmember


def test_a_memorising_model_is_caught_and_a_model_that_learnt_nothing_is_not():
    rng = np.random.default_rng(0)
    m, n = _losses(rng, 3000, overfit=True)
    strong = attack_metrics(m, n, n_boot=50)
    # P(member loss < the 1% non-member quantile, about 0.01) = 1 - exp(-0.01 / 0.05) = 0.18
    assert strong["auc"] > 0.9 and strong["tpr_at_fpr"]["0.01"] > 0.1
    m, n = _losses(rng, 3000, overfit=False)
    weak = attack_metrics(m, n, n_boot=50)
    assert 0.45 < weak["auc"] < 0.55
    lo, hi = weak["auc_95ci"]
    assert lo < 0.5 < hi and lo < weak["auc"] < hi


def _side(loss, groups=None, correct=None):
    loss = np.asarray(loss, dtype=float)
    side = {"loss": loss, "prob": np.exp(-loss), "correct": np.ones(len(loss), bool) if correct is None else correct}
    if groups is not None:
        side["groups"] = np.asarray(groups)
    return side


def test_audit_reports_cells_groups_and_the_accuracy_gap():
    rng = np.random.default_rng(1)
    member = _side(rng.exponential(0.1, 400), groups=np.repeat([f"m{i}" for i in range(8)], 50))
    nonmember = _side(
        rng.exponential(1.0, 200),
        groups=np.repeat([f"n{i}" for i in range(4)], 50),
        correct=np.arange(200) % 2 == 0,
    )
    rep = audit(member, nonmember, seed=0)
    assert rep["cells"]["n_members"] == 400 and rep["cells"]["n_nonmembers"] == 200
    assert rep["accuracy_gap"] == pytest.approx(0.5)
    assert rep["by_group"]["n_member_groups"] == 8 and rep["by_group"]["n_nonmember_groups"] == 4
    assert rep["by_group"]["auc"] == 1.0  # every member donor has lower mean loss
    assert any("groups: AUC 1.000" in line for line in summary_lines(rep))


def test_audit_refuses_group_results_when_a_group_is_on_both_sides():
    member = _side([0.1, 0.2], groups=["a", "b"])
    nonmember = _side([1.0, 2.0], groups=["b", "c"])
    rep = audit(member, nonmember)
    assert "error" in rep["by_group"] and "auc" not in rep["by_group"]


def test_audit_without_groups_has_no_group_section(tmp_path, capsys):
    rep = audit(_side([0.1, 0.5, 0.9]), _side([0.4, 1.2]))
    assert "by_group" not in rep
    path = write_report(rep, tmp_path / "a.json")
    assert json.loads(path.read_text())["cells"]["n_members"] == 3
    assert "[privacy-audit] cells: AUC" in capsys.readouterr().out


def test_cap_indices_samples_without_replacement():
    idx = np.arange(100, 200)
    capped = cap_indices(idx, 10, seed=0)
    assert len(capped) == 10 and len(set(capped.tolist())) == 10 and set(capped) <= set(idx)
    assert (cap_indices(idx, 10, seed=0) == capped).all()
    assert len(cap_indices(idx, 0, seed=0)) == 100 and len(cap_indices(idx, 500, seed=0)) == 100


# --- backend hooks --------------------------------------------------------------------------
def _cfg(tmp_path, **training):
    from scfm_ft.config import Config

    return Config.model_validate(
        {
            "name": "pa",
            "output_dir": str(tmp_path / "out"),
            "input": {"h5ad_path": str(tmp_path / "x.loom")},
            "classifier": {"kind": "cell", "label_column": "cell_type", "group_column": "donor_id"},
            "training": {"num_crossval_splits": 1, "privacy_audit": True, **training},
        }
    )


def test_scgpt_writes_one_report_per_fold(tmp_path):
    from scfm_ft.backends.scgpt._train import _write_privacy_audit

    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(0)
    parts = [
        {"member": _side(rng.exponential(0.2, 50)), "nonmember": _side(rng.exponential(1.0, 50))}
        for _ in range(3)
    ]
    out = _write_privacy_audit(parts, cfg, tmp_path)
    rep = json.loads((tmp_path / "pa_privacy_audit.json").read_text())
    assert len(rep["folds"]) == 3
    assert out["privacy_audit_auc"] == pytest.approx(np.mean([f["cells"]["auc"] for f in rep["folds"]]))
    _write_privacy_audit(parts[:1], cfg, tmp_path)
    assert "cells" in json.loads((tmp_path / "pa_privacy_audit.json").read_text())


def _install_fake_geneformer_eval(monkeypatch, seen):
    """Fake geneformer.evaluation_utils / perturber_utils: a 'model' whose loss is low on
    the cells it trained on (marked in the dataset), so the audit must find them."""
    import sys
    import types

    pkg = types.ModuleType("geneformer")
    eu = types.ModuleType("geneformer.evaluation_utils")
    pu = types.ModuleType("geneformer.perturber_utils")

    def classifier_predict(model, kind, ds, batch, token_dict):
        seen.setdefault("predict", []).append(len(ds))
        y = list(ds["label"])
        logits = [[4.0, -4.0] if t else [0.2, 0.0] for t in ds["trained"]]
        logits = [lg if lab == 0 else lg[::-1] for lg, lab in zip(logits, y)]
        return [0] * len(y), y, logits, None

    def load_model(model_type, num_classes, model_directory, mode):
        seen["load"] = (model_type, num_classes, model_directory, mode)
        return object()

    eu.classifier_predict = classifier_predict
    pu.load_model = load_model
    pkg.evaluation_utils, pkg.perturber_utils = eu, pu
    for name, mod in (("geneformer", pkg), ("geneformer.evaluation_utils", eu), ("geneformer.perturber_utils", pu)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_geneformer_audit_scores_training_donors_against_held_out_donors(tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    import types

    from scfm_ft.finetune import _privacy_audit
    from scfm_ft.splits import GroupSplit

    seen: dict = {}
    _install_fake_geneformer_eval(monkeypatch, seen)
    out = tmp_path / "out"
    model_dir = out / "261118_geneformer_cellClassifier_pa" / "ksplit1"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"id2label": {"0": "T", "1": "B"}}))

    def ds(donors, trained):
        n = len(donors)
        return datasets.Dataset.from_dict(
            {"donor_id": donors, "label": [i % 2 for i in range(n)], "trained": [trained] * n}
        )

    split = GroupSplit(train=["a", "b", "c"], valid=["d"], test=["e"])
    labeled = datasets.concatenate_datasets([ds(["a"] * 20 + ["b"] * 20 + ["c"] * 20, True), ds(["d"] * 20, False)])
    labeled.save_to_disk(str(out / "pa_labeled_train.dataset"))
    ds(["e"] * 20, False).save_to_disk(str(out / "pa_labeled_test.dataset"))

    cfg = _cfg(tmp_path)
    cc = types.SimpleNamespace(gene_token_dict={})
    result = _privacy_audit(
        cc, cfg, out, "pa", out / "pa_labeled_train.dataset", out / "pa_labeled_test.dataset", split
    )
    assert seen["load"] == ("CellClassifier", 2, str(model_dir), "eval")
    assert seen["predict"] == [60, 40]  # members: train donors; non-members: valid + test donors
    rep = json.loads((out / "pa_privacy_audit.json").read_text())
    assert rep["cells"]["auc"] == 1.0 and result["privacy_audit_auc"] == 1.0
    assert rep["by_group"]["n_member_groups"] == 3 and rep["by_group"]["n_nonmember_groups"] == 2


def test_geneformer_audit_is_skipped_without_a_group_split_or_with_folds(tmp_path, capsys):
    from scfm_ft.finetune import _privacy_audit

    assert _privacy_audit(None, _cfg(tmp_path), tmp_path, "pa", None, None, None) == {}
    assert "needs classifier.group_column" in capsys.readouterr().out
    assert _privacy_audit(None, _cfg(tmp_path, num_crossval_splits=5), tmp_path, "pa", None, None, None) == {}
    assert "num_crossval_splits 1 only" in capsys.readouterr().out


def test_launcher_forwards_the_audit_and_regulated_data_turns_it_on(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / "sagemaker" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    launch, sm_entry = load("launch"), load("sm_entry")
    base = ["launch.py", "--image-uri", "x", "--role", "r"]
    monkeypatch.setattr(sys, "argv", base)
    args = launch._parse()
    assert launch.build_hyperparameters(args, scgpt_model_root="/s", quantize=False)["privacy-audit"] == "false"
    monkeypatch.setattr(sys, "argv", [*base, "--privacy-audit"])
    args = launch._parse()
    assert launch.build_hyperparameters(args, scgpt_model_root="/s", quantize=False)["privacy-audit"] == "true"
    assert "args.privacy_audit = True" in (root / "sagemaker" / "launch.py").read_text()
    monkeypatch.setattr(sys, "argv", ["sm_entry.py", "--privacy-audit", "true"])
    assert sm_entry._truthy(sm_entry._parse_args().privacy_audit) is True


# --- release rule ---------------------------------------------------------------------------
def _report(cell_tpr=0.02, group_auc=0.55, n_nonmember_groups=6, group=True, group_error=None):
    rep = {"cells": {"auc": 0.52, "tpr_at_fpr": {"0.01": cell_tpr, "0.001": 0.0}}}
    if group:
        rep["by_group"] = {"n_member_groups": 20, "n_nonmember_groups": n_nonmember_groups}
        if group_error:
            rep["by_group"]["error"] = group_error
        else:
            rep["by_group"]["auc"] = group_auc
    return rep


def test_release_rule_passes_within_both_limits():
    from scfm_ft.privacy import release_check

    assert release_check(_report()) == (True, [], [])
    # The limits themselves pass.
    assert release_check(_report(cell_tpr=0.05, group_auc=0.60))[0]


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"cell_tpr": 0.051}, "caught at 1% false positives"),
        ({"group_auc": 0.61}, "donor-level AUC 0.610"),
        ({"group": False}, "run with --group-column"),
        ({"group_error": "1 group(s) are on both sides"}, "unusable"),
    ],
)
def test_release_rule_fails_on_each_limit_and_without_a_donor_split(kwargs, fragment):
    from scfm_ft.privacy import release_check

    ok, failures, _ = release_check(_report(**kwargs))
    assert not ok and any(fragment in f for f in failures)


def test_release_rule_warns_on_few_held_out_donors_and_checks_every_fold():
    from scfm_ft.privacy import release_check

    ok, failures, warnings = release_check(_report(n_nonmember_groups=2))
    assert ok and not failures and "only 2 held-out donor(s)" in warnings[0]
    ok, failures, _ = release_check({"folds": [_report(), _report(group_auc=0.7), _report()]})
    assert not ok and failures == ["fold 2: donor-level AUC 0.700 (limit 0.60)"]


def test_privacy_check_command_exits_by_the_rule(tmp_path):
    from typer.testing import CliRunner

    from scfm_ft.cli import app

    good = tmp_path / "good.json"
    good.write_text(json.dumps(_full_report(group_auc_low=True)))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_full_report(group_auc_low=False)))
    runner = CliRunner()
    res = runner.invoke(app, ["privacy-check", str(good)])
    assert res.exit_code == 0, res.output
    assert "release rule: PASS" in res.output
    res = runner.invoke(app, ["privacy-check", str(bad)])
    assert res.exit_code == 1 and "release rule: FAIL" in res.output


def _full_report(group_auc_low: bool) -> dict:
    """A real audit() report: members that the model memorised, or not, by donor."""
    rng = np.random.default_rng(3)
    member_loss = rng.exponential(1.0 if group_auc_low else 0.05, 600)
    member = _side(member_loss, groups=np.repeat([f"m{i}" for i in range(12)], 50))
    nonmember = _side(rng.exponential(1.0, 300), groups=np.repeat([f"n{i}" for i in range(6)], 50))
    return audit(member, nonmember, seed=0)
