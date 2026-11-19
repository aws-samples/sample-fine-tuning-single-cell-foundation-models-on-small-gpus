# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Membership-inference audit for a fine-tuned classifier (T013).

The audit runs the standard loss-threshold attack: an attacker who can query the model
guesses that a cell was in the training set when the model's loss on it is low. Members
are cells the model trained on; non-members are held-out cells it never saw. If the
model behaves the same on both, the attack is no better than a coin flip (ROC AUC 0.5).

This is a lower bound on leakage: stronger attacks (for example LiRA, which trains
shadow models) can do better. It also measures the model, not the data: a model trained
on few cells per donor can leak more than these numbers show.

Reported per audit:

* ``auc``: ROC AUC of the attack over cells, with a bootstrap 95% interval;
* ``tpr_at_fpr``: share of members caught at 1% and 0.1% false-positive rate (the
  numbers that matter for a few individuals, not the average);
* ``advantage``: the largest TPR - FPR over all thresholds;
* the mean loss and accuracy on each side (the generalisation gap the attack exploits);
* ``by_group``: the same attack on per-group mean loss, when groups (donors) are given.
  This only means something with a donor-aware split, where held-out donors are true
  non-members; with few donors the AUC is coarse, so the group counts are reported.

The functions here are pure NumPy, so they are tested without a model.
"""

from __future__ import annotations

import json
from pathlib import Path

FPR_POINTS = (0.01, 0.001)


def per_cell_loss(logits, labels):
    """Cross-entropy and true-class probability per cell, from raw logits."""
    import numpy as np

    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    z = z - z.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    true_log_p = log_p[np.arange(len(y)), y]
    correct = z.argmax(axis=1) == y
    return -true_log_p, np.exp(true_log_p), correct


def _roc(member_scores, nonmember_scores):
    import numpy as np
    from sklearn.metrics import roc_curve

    m = np.asarray(member_scores, dtype=np.float64)
    n = np.asarray(nonmember_scores, dtype=np.float64)
    if len(m) == 0 or len(n) == 0:
        raise ValueError("an audit needs at least one member and one non-member")
    y = np.concatenate([np.ones(len(m)), np.zeros(len(n))])
    fpr, tpr, _ = roc_curve(y, np.concatenate([m, n]))
    return m, n, fpr, tpr


def roc_auc(member_scores, nonmember_scores) -> float:
    """P(member score > non-member score), ties counted half."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    m, n, _, _ = _roc(member_scores, nonmember_scores)
    return float(roc_auc_score(np.concatenate([np.ones(len(m)), np.zeros(len(n))]), np.concatenate([m, n])))


def tpr_at_fpr(member_scores, nonmember_scores, fpr: float) -> float:
    """Highest share of members caught at a threshold whose false-positive rate is at most ``fpr``."""
    _, _, f, t = _roc(member_scores, nonmember_scores)
    ok = f <= fpr
    return float(t[ok].max()) if ok.any() else 0.0


def advantage(member_scores, nonmember_scores) -> float:
    """max over thresholds of TPR - FPR (0 = no signal, 1 = perfect separation)."""
    _, _, f, t = _roc(member_scores, nonmember_scores)
    return float(max(0.0, (t - f).max()))


def _bootstrap_auc(member_scores, nonmember_scores, n_boot: int, seed: int) -> list[float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    m = np.asarray(member_scores)
    n = np.asarray(nonmember_scores)
    aucs = [
        roc_auc(rng.choice(m, len(m)), rng.choice(n, len(n)))
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return [float(lo), float(hi)]


def attack_metrics(member_loss, nonmember_loss, *, n_boot: int = 200, seed: int = 0) -> dict:
    """Loss-threshold attack metrics. Lower loss means "member", so scores are -loss."""
    import numpy as np

    ms, ns = -np.asarray(member_loss, dtype=np.float64), -np.asarray(nonmember_loss, dtype=np.float64)
    out = {
        "n_members": len(ms),
        "n_nonmembers": len(ns),
        "auc": roc_auc(ms, ns),
        "advantage": advantage(ms, ns),
        "tpr_at_fpr": {str(f): tpr_at_fpr(ms, ns, f) for f in FPR_POINTS},
    }
    if n_boot and len(ms) > 1 and len(ns) > 1:
        out["auc_95ci"] = _bootstrap_auc(ms, ns, n_boot, seed)
    return out


def _group_means(loss, groups) -> dict:
    import numpy as np

    loss = np.asarray(loss, dtype=np.float64)
    groups = np.asarray(groups)
    return {g: float(loss[groups == g].mean()) for g in np.unique(groups).tolist()}


def audit(
    member: dict,
    nonmember: dict,
    *,
    seed: int = 0,
) -> dict:
    """Full audit. ``member`` and ``nonmember`` each hold per-cell arrays: ``loss``,
    ``prob`` (true-class probability), ``correct`` (bool), and optionally ``groups``."""
    import numpy as np

    report = {
        "attack": "loss threshold (lower loss = member); a lower bound on leakage",
        "cells": attack_metrics(member["loss"], nonmember["loss"], seed=seed),
        "members": {
            "mean_loss": float(np.mean(member["loss"])),
            "mean_true_class_prob": float(np.mean(member["prob"])),
            "accuracy": float(np.mean(member["correct"])),
        },
        "nonmembers": {
            "mean_loss": float(np.mean(nonmember["loss"])),
            "mean_true_class_prob": float(np.mean(nonmember["prob"])),
            "accuracy": float(np.mean(nonmember["correct"])),
        },
    }
    report["accuracy_gap"] = report["members"]["accuracy"] - report["nonmembers"]["accuracy"]
    mg, ng = member.get("groups"), nonmember.get("groups")
    if mg is not None and ng is not None:
        m_means, n_means = _group_means(member["loss"], mg), _group_means(nonmember["loss"], ng)
        overlap = sorted(set(m_means) & set(n_means))
        by_group = {"n_member_groups": len(m_means), "n_nonmember_groups": len(n_means)}
        if overlap:
            # A group on both sides means the split was not by group: per-group results
            # would compare a donor with itself. Report it instead of a misleading number.
            by_group["error"] = f"{len(overlap)} group(s) are on both sides; the split was not by group"
        else:
            by_group.update(
                attack_metrics(list(m_means.values()), list(n_means.values()), n_boot=0)
            )
        report["by_group"] = by_group
    return report


def summary_lines(report: dict) -> list[str]:
    c = report["cells"]
    ci = c.get("auc_95ci")
    tpr = c["tpr_at_fpr"]
    lines = [
        (
            f"cells: AUC {c['auc']:.3f}"
            + (f" (95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci else "")
            + f", TPR at 1% FPR {tpr['0.01']:.3f}, at 0.1% FPR {tpr['0.001']:.3f}, "
            + f"advantage {c['advantage']:.3f} ({c['n_members']} members, "
            + f"{c['n_nonmembers']} non-members)"
        ),
        (
            f"accuracy: members {report['members']['accuracy']:.3f}, non-members "
            f"{report['nonmembers']['accuracy']:.3f}; mean loss {report['members']['mean_loss']:.3f} vs "
            f"{report['nonmembers']['mean_loss']:.3f}"
        ),
    ]
    g = report.get("by_group")
    if g:
        if "error" in g:
            lines.append(f"groups: {g['error']}")
        else:
            lines.append(
                f"groups: AUC {g['auc']:.3f} over {g['n_member_groups']} member and "
                f"{g['n_nonmember_groups']} non-member group(s)"
            )
    return lines


def write_report(report: dict, path: Path) -> Path:
    path.write_text(json.dumps(report, indent=2))
    for line in summary_lines(report):
        print(f"[privacy-audit] {line}", flush=True)
    print_release_check(report)
    return path


def print_release_check(report: dict) -> bool:
    """Print the release rule's verdict for an audit report (see ``release_check``)."""
    ok, failures, warnings = release_check(report)
    print(f"[privacy-audit] release rule: {'PASS' if ok else 'FAIL'}", flush=True)
    for line in failures:
        print(f"[privacy-audit]   fail: {line}", flush=True)
    for line in warnings:
        print(f"[privacy-audit]   warning: {line}", flush=True)
    return ok


def cap_indices(idx, max_cells: int, seed: int):
    """At most ``max_cells`` of ``idx``, sampled without replacement (sorted)."""
    import numpy as np

    idx = np.asarray(idx)
    if max_cells <= 0 or len(idx) <= max_cells:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, max_cells, replace=False))


# --- release rule ------------------------------------------------------------------------------
# A model trained on human donor data may leave the group allowed to see that data only if its
# audit (with a donor-aware split) stays within these limits. Agreed 2026-11-18; see
# docs/security/aws-baseline.md, "Sharing a model".
RELEASE_GROUP_AUC_MAX = 0.60  # donor-level loss-threshold AUC
RELEASE_CELL_TPR_AT_1PCT_FPR_MAX = 0.05  # share of training cells caught at 1% false positives
# Below this many held-out donors the donor-level AUC moves in large steps; warn, do not fail.
RELEASE_MIN_NONMEMBER_GROUPS = 5


def release_check(audit_report: dict) -> tuple[bool, list[str], list[str]]:
    """Apply the release rule to one ``<name>_privacy_audit.json``.

    Returns ``(ok, failures, warnings)``. A cross-validated report (``{"folds": [...]}``)
    passes only if every fold passes. A report without a donor-level result fails: without a
    donor split, held-out cells can share donors with training cells, so the audit
    understates what the model leaks about its donors.
    """
    reports = audit_report.get("folds") or [audit_report]
    failures: list[str] = []
    warnings: list[str] = []
    for k, rep in enumerate(reports):
        where = f"fold {k + 1}: " if len(reports) > 1 else ""
        tpr = rep["cells"]["tpr_at_fpr"]["0.01"]
        if tpr > RELEASE_CELL_TPR_AT_1PCT_FPR_MAX:
            failures.append(
                f"{where}{tpr:.3f} of training cells caught at 1% false positives "
                f"(limit {RELEASE_CELL_TPR_AT_1PCT_FPR_MAX:.2f})"
            )
        g = rep.get("by_group")
        if not g:
            failures.append(f"{where}no donor-level result: run with --group-column (a donor split)")
        elif "error" in g:
            failures.append(f"{where}donor-level result unusable: {g['error']}")
        else:
            if g["auc"] > RELEASE_GROUP_AUC_MAX:
                failures.append(
                    f"{where}donor-level AUC {g['auc']:.3f} (limit {RELEASE_GROUP_AUC_MAX:.2f})"
                )
            if g["n_nonmember_groups"] < RELEASE_MIN_NONMEMBER_GROUPS:
                warnings.append(
                    f"{where}only {g['n_nonmember_groups']} held-out donor(s): the donor-level AUC "
                    f"is coarse below {RELEASE_MIN_NONMEMBER_GROUPS}"
                )
    return not failures, failures, warnings
