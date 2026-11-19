#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# One-shot environment setup for geneformer-fine-tuning.
# Does NOT require git-lfs: clones Geneformer code, then fetches the needed weights
# + dictionaries over HTTPS via huggingface_hub.
set -euo pipefail

REPO_DIR="${1:-vendor/Geneformer}"
PRESET="${2:-V1-10M}"
# Pinned Geneformer commit; keep in sync with sagemaker/Dockerfile and fetch_weights.py.
GENEFORMER_REVISION="1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"

echo "==> Cloning Geneformer code into ${REPO_DIR} at ${GENEFORMER_REVISION} (LFS files become pointer stubs)"
if [ ! -d "${REPO_DIR}/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ctheodoris/Geneformer "${REPO_DIR}"
  git -C "${REPO_DIR}" checkout --detach "${GENEFORMER_REVISION}"
else
  echo "    already present ($(git -C "${REPO_DIR}" rev-parse HEAD))"
fi

echo "==> Creating uv environment and installing project (dependencies from uv.lock)"
uv venv
uv sync

echo "==> Patching Geneformer to run on transformers 5 (idempotent)"
uv run python sagemaker/patches/geneformer_transformers5.py "${REPO_DIR}/geneformer"

echo "==> Installing the Geneformer package (code only; deps come from pyproject)"
uv pip install --no-deps "./${REPO_DIR}"

echo "==> Fetching ${PRESET} weights + dictionaries over HTTPS"
uv run python scripts/fetch_weights.py --preset "${PRESET}" --repo "${REPO_DIR}"

echo "==> Done. Try the CPU smoke test:"
echo "    uv run scfm-ft make-sample-data --out data/smoke.loom --n-cells 150"
echo "    uv run scfm-ft finetune --config configs/smoke_test.yaml"
