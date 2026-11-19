# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Group-aware (for example donor-aware) train/valid/test splits.

A cell-level split puts cells from the same donor in both the training and the held-out
set, so held-out accuracy partly measures how well the model recognises donors it has
already seen, and a membership-inference test against that held-out set understates
what the model leaks about the people it was trained on. With a group column, every
group (donor) lands in exactly one split.

Both backends use these functions: the scGPT backend indexes cells with them directly,
and the Geneformer backend passes the resulting group lists to Geneformer's
``split_id_dict`` (its own ``attr_to_split`` only works when each group has one label,
which is not true for cell-type classification).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class GroupSplit:
    """Which groups went to which split. Group values keep their original type."""

    train: list = field(default_factory=list)
    valid: list = field(default_factory=list)
    test: list = field(default_factory=list)

    def of(self, name: str) -> list:
        return {"train": self.train, "valid": self.valid, "test": self.test}[name]


def group_split(groups, split_sizes: dict, seed: int) -> GroupSplit:
    """Assign whole groups to test, then valid, then train, by cell count.

    Groups are shuffled with ``seed``; test and valid each take groups until they hold
    at least their fraction of the cells (so they may overshoot when groups are large),
    and each split with a non-zero fraction gets at least one group. Train keeps the
    rest and always gets at least one group.
    """
    import numpy as np

    values = np.asarray(groups)
    if values.size == 0:
        raise ValueError("group split needs at least one cell")
    uniq, counts = np.unique(values, return_counts=True)
    fr = {k: float(split_sizes.get(k, 0.0)) for k in _SPLITS}
    needed = ["train"] + [k for k in ("valid", "test") if fr[k] > 0]
    if len(uniq) < len(needed):
        raise ValueError(
            f"a group split into {', '.join(needed)} needs at least {len(needed)} groups; "
            f"the group column has {len(uniq)}"
        )
    order = list(np.random.default_rng(seed).permutation(len(uniq)))
    total = int(counts.sum())
    taken: dict[str, list] = {"valid": [], "test": []}
    for name in ("test", "valid"):
        if fr[name] <= 0:
            continue
        # Groups the later splits still need: train always, plus valid while filling test.
        reserve = 1 + (1 if name == "test" and fr["valid"] > 0 else 0)
        cells = 0
        while len(order) > reserve and (not taken[name] or cells < fr[name] * total):
            g = order.pop(0)
            taken[name].append(uniq[g].item())
            cells += int(counts[g])
    train = [uniq[g].item() for g in order]
    return GroupSplit(train=sorted(train), valid=sorted(taken["valid"]), test=sorted(taken["test"]))


def indices_for(groups, members):
    """Positions (a NumPy index array) of the cells whose group is in ``members``."""
    import numpy as np

    return np.flatnonzero(np.isin(np.asarray(groups), list(members)))


def group_folds(groups, labels, n_splits: int, seed: int) -> list:
    """``(train_idx, test_idx)`` pairs from stratified group k-fold: no group in two folds."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"{n_splits}-fold grouped CV needs at least {n_splits} groups; got {n_groups}")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in sgkf.split(np.zeros(len(groups)), np.asarray(labels), groups)]


def split_record(groups, labels, assignment: dict, column: str) -> dict:
    """JSON-serialisable record of a split: groups and cells per split, and any label
    that appears in a held-out split but not in train (the model cannot predict it)."""
    import numpy as np

    groups = np.asarray(groups)
    labels = np.asarray(labels).astype(str)
    record: dict = {"group_column": column, "splits": {}}
    train_labels: set = set()
    for name, members in assignment.items():
        idx = indices_for(groups, members)
        record["splits"][name] = {
            "groups": [str(m) for m in members],
            "n_groups": len(members),
            "n_cells": len(idx),
            "label_counts": dict(sorted(Counter(labels[idx].tolist()).items())),
        }
        if name == "train":
            train_labels = set(labels[idx].tolist())
    record["labels_missing_from_train"] = {
        name: sorted(set(info["label_counts"]) - train_labels)
        for name, info in record["splits"].items()
        if name != "train" and set(info["label_counts"]) - train_labels
    }
    return record


def write_split_record(record: dict, path: Path) -> Path:
    path.write_text(json.dumps(record, indent=2))
    missing = record.get("labels_missing_from_train") or {}
    for name, labels in missing.items():
        print(
            f"[split] WARNING: {len(labels)} label(s) in the {name} split never appear in train: "
            f"{', '.join(labels[:10])}{' ...' if len(labels) > 10 else ''}",
            flush=True,
        )
    return path
