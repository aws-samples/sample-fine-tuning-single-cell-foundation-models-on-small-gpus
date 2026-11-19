#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Patch Geneformer's perturber_utils.load_model so QLoRA (4-bit) classifier
loads place the quantized model on GPU via device_map.

Problem: load_model() calls `model_class.from_pretrained(quantization_config=...)`
with no device_map. A 4-bit load then materializes the uint8/Byte weights on CPU,
and initializing a freshly-added classification head there crashes with:
    NotImplementedError: "normal_kernel_cpu" not implemented for 'Byte'

Fix: when a quantization_config is set, also pass device_map={"": 0} so the load
(and the head init) run on CUDA, where .normal_() is implemented.

This runs at image-build time against the upstream Geneformer clone (the Dockerfile
re-clones Geneformer rather than using the repo's vendored copy). It is idempotent
and fails loudly if the anchor it expects is missing, so an upstream refactor can't
silently leave the image unpatched.

Usage: python3 patch_geneformer.py /path/to/perturber_utils.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = 'model_args["quantization_config"] = quantize_config'
SENTINEL = 'device_map'  # already-patched marker within the injected block

INJECT = '''model_args["quantization_config"] = quantize_config
        # scfm-ft patch (patch_geneformer.py): pin a quantized load to the
        # GPU so a freshly-added classifier head can be .normal_()-initialized
        # (no CPU kernel for Byte dtype). No-op without CUDA.
        import torch as _torch_patch
        if _torch_patch.cuda.is_available():
            model_args["device_map"] = {"": 0}'''


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_geneformer.py <perturber_utils.py>", file=sys.stderr)
        return 2

    target = Path(sys.argv[1])
    src = target.read_text()

    # Idempotent: if our injected device_map block is already present, do nothing.
    if 'model_args["device_map"] = {"": 0}' in src:
        print(f"[patch_geneformer] already patched: {target}")
        return 0

    if ANCHOR not in src:
        print(
            f"[patch_geneformer] ERROR: anchor not found in {target}. "
            "Upstream Geneformer may have changed load_model(); update this patch.",
            file=sys.stderr,
        )
        return 1

    # Replace the first occurrence of the anchor line with the injected block.
    patched = src.replace(ANCHOR, INJECT, 1)
    target.write_text(patched)
    print(f"[patch_geneformer] patched device_map into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
