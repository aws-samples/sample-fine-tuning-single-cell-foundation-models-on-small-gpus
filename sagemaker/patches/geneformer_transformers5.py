#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Patch Geneformer so it imports under transformers 5.x.

Problem: Geneformer pins ``transformers==4.46``. Its ``collator_for_classification.py``
and ``pretrainer.py`` import four names that transformers 5 removed:
``SpecialTokensMixin`` (folded into ``PreTrainedTokenizerBase``) and the TensorFlow
helpers ``is_tf_available``, ``_is_tensorflow`` and ``_is_torch``. Every other Geneformer
module imports ``collator_for_classification``, so the whole package fails to import.
Its default training arguments also pass ``group_by_length``, which transformers 5's
``TrainingArguments`` rejects (it became ``train_sampling_strategy``).
4.46.3 carries CVE-2026-5241 (critical) and most of the RCE advisories once accepted
as AR-1 in docs/security/accepted-risks.md; 5.10 is the lowest release with every
published transformers fix.

Fix: install ``geneformer_transformers5_compat.py`` (next to this script) into the
package as ``geneformer/_scfm_transformers5_compat.py`` and rewrite the two modules'
imports to take those four names from it; map ``group_by_length`` to the equivalent
``train_sampling_strategy`` in ``classifier_utils.py`` (``False`` -> ``"random"``, the
default) and ``pretrainer.py``. Nothing else in Geneformer changes.

Runs at image-build time against the pinned upstream clone, like
``geneformer_quantize_device_map.py``. It is idempotent, and it fails loudly if any
import line it expects is missing, so an upstream refactor can't silently leave the
image half-patched.

Usage: python3 geneformer_transformers5.py /path/to/Geneformer/geneformer
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

COMPAT_SRC = Path(__file__).with_name("geneformer_transformers5_compat.py")
COMPAT_NAME = "_scfm_transformers5_compat.py"
MARKER = "# scfm-ft patch: transformers 5"
COMPAT_IMPORT = (
    f"from ._scfm_transformers5_compat import (  {MARKER}\n"
    "    SpecialTokensMixin,\n"
    "    _is_tensorflow,\n"
    "    _is_torch,\n"
    "    is_tf_available,\n"
    ")\n"
)
IMPORT_EDITS = [
    ("    SpecialTokensMixin,\n", ""),
    (
        "from transformers.utils import is_tf_available, is_torch_available, logging, to_py_obj\n",
        "from transformers.utils import is_torch_available, logging, to_py_obj\n",
    ),
    ("from transformers.utils.generic import _is_tensorflow, _is_torch\n", COMPAT_IMPORT),
]

# file -> [(old, new)]; each `old` must occur exactly once in its file. transformers 5
# also replaced the TrainingArguments flag `group_by_length` with
# `train_sampling_strategy` ("random" is the default and the old False behaviour).
EDITS = {
    "collator_for_classification.py": IMPORT_EDITS,
    "pretrainer.py": [
        *IMPORT_EDITS,
        (
            "        if self.args.group_by_length:\n",
            f'        if self.args.train_sampling_strategy == "group_by_length":  {MARKER}\n',
        ),
    ],
    "classifier_utils.py": [
        (
            '        "group_by_length": False,\n',
            f'        "train_sampling_strategy": "random",  {MARKER}\n',
        ),
    ],
}


def patch_source(src: str, name: str) -> str:
    """Return ``src`` with this file's transformers-5 edits applied (idempotent)."""
    if MARKER in src:
        return src
    for old, new in EDITS[name]:
        count = src.count(old)
        if count != 1:
            raise ValueError(
                f"{name}: expected exactly one {old.strip()!r}, found {count}. "
                "Upstream Geneformer changed; update this patch."
            )
        src = src.replace(old, new, 1)
    return src


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: geneformer_transformers5.py <Geneformer/geneformer package dir>", file=sys.stderr)
        return 2
    pkg = Path(sys.argv[1])
    try:
        # Patch every target in memory first, so a missing anchor in one file can't
        # leave the package half-rewritten.
        results = {}
        for name in EDITS:
            src = (pkg / name).read_text()
            results[name] = (src, patch_source(src, name))
    except (OSError, ValueError) as e:
        print(f"[geneformer_transformers5] ERROR: {e}", file=sys.stderr)
        return 1
    for name, (src, patched) in results.items():
        target = pkg / name
        if patched == src:
            print(f"[geneformer_transformers5] already patched: {target}")
        else:
            target.write_text(patched)
            print(f"[geneformer_transformers5] patched imports in {target}")
    shutil.copyfile(COMPAT_SRC, pkg / COMPAT_NAME)
    print(f"[geneformer_transformers5] installed {pkg / COMPAT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
