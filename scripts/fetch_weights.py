#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Fetch Geneformer pretrained weights + token dictionaries over HTTPS.

Geneformer stores its weights and dictionaries as git-LFS files. A plain
``git clone`` (or ``GIT_LFS_SKIP_SMUDGE=1``) leaves them as tiny pointer stubs.
This script materializes the real files via ``huggingface_hub`` — no git-lfs
required — into the cloned repo at ``vendor/Geneformer`` AND into the installed
``geneformer`` package (the tokenizer loads dictionaries from the package).

Usage:
    python scripts/fetch_weights.py --preset V1-10M
    python scripts/fetch_weights.py --preset V2-104M
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "ctheodoris/Geneformer"
# Pinned commit, so every download is the same file; keep in sync with
# sagemaker/Dockerfile (GENEFORMER_REVISION) and scripts/setup.sh.
REVISION = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"

# Per-preset: the model subdir and which corpus dictionary set it uses.
PRESETS = {
    "V1-10M": {"model_dir": "Geneformer-V1-10M", "dict_subdir": "geneformer/gene_dictionaries_30m", "corpus": "gc30M"},
    "V2-104M": {"model_dir": "Geneformer-V2-104M", "dict_subdir": "geneformer", "corpus": "gc104M"},
    "V2-316M": {"model_dir": "Geneformer-V2-316M", "dict_subdir": "geneformer", "corpus": "gc104M"},
}

MODEL_FILES = ["config.json", "model.safetensors", "training_args.bin"]
DICT_FILES = [
    "ensembl_mapping_dict_{corpus}.pkl",
    "gene_median_dictionary_{corpus}.pkl",
    "gene_name_id_dict_{corpus}.pkl",
    "token_dictionary_{corpus}.pkl",
]

# SHA-256 of every file this script downloads, taken from the pinned commit above: the `oid` in
# each Git LFS pointer for the weights, dictionaries and training_args.bin, and the blob hash
# for config.json. Every download is checked against it before it is copied anywhere, so a
# corrupt or tampered file (or Hugging Face cache entry) stops the setup instead of becoming
# a model or a pickle that Geneformer loads. To regenerate after changing REVISION, read the
# pointers from the clone: `git -C vendor/Geneformer show HEAD:<path>`.
PINNED_SHA256 = {
    "Geneformer-V1-10M/config.json":
        "9cf69ca3bdb0215c4188b54c451b6f02adfe68b8f66011a57d0f32845133fd4b",
    "Geneformer-V1-10M/model.safetensors":
        "a5e33a757431643b3697de7ef6127950cdc49e06e58d4266b3a3ab191b683f14",
    "Geneformer-V1-10M/training_args.bin":
        "f0ec3459454205174c9d2e4d6c6930f6b0fbf3364fc03a6f4d99c4d3add2012b",
    "Geneformer-V2-104M/config.json":
        "467d4492f0dd53b4d60afffe20812db484ca1cf9fdbeb6a6e060e93564f70859",
    "Geneformer-V2-104M/model.safetensors":
        "fff5cba29ddd8792991fa77b4872246fbe548a178cebda3775cdc72b67780e7f",
    "Geneformer-V2-104M/training_args.bin":
        "0d8ddd9e4f35b5fe23a3adaae03aa4480705ca82eed546a488f970adb3752d9d",
    "Geneformer-V2-316M/config.json":
        "2cc4af3442644e84af71814a607b61c958d7b4e5ebde6791ab77b5f534ac6f6e",
    "Geneformer-V2-316M/model.safetensors":
        "965ceccea81953d362081ef3843560a0e4fef88d396c28017881f1e94b1246f3",
    "Geneformer-V2-316M/training_args.bin":
        "e45150f9a4ca34cb4e91ce79f65f3d99d9d66df9f66a37517a352d291008e0b8",
    "geneformer/ensembl_mapping_dict_gc104M.pkl":
        "0819bcbd869cfa14279449b037eb9ed1d09a91310e77bd1a19d927465030e95c",
    "geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl":
        "eac0fb0b3007267871b6305ac0003ceba19d4f28d85686cb9067ecf142787869",
    "geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl":
        "b3b589bb5ec75040d05fc44dd6bf0184cf87f3c362cf158d196a6ed3b7fe5f39",
    "geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl":
        "55e67962e79c0039a6c32d43c5c99f38e51964bbcfa32f736150ee1e285c438c",
    "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl":
        "ab9dc40973fa5224d77b793e2fd114cacf3d08423ed9c4c49caf0ba9c7f218f1",
    "geneformer/gene_median_dictionary_gc104M.pkl":
        "a51c53f6a771d64508dfaf61529df70e394c53bd20856926117ae5d641a24bf5",
    "geneformer/gene_name_id_dict_gc104M.pkl":
        "fabfa0c2f49c598c59ae432a32c3499a5908c033756c663b5e0cddf58deea8e1",
    "geneformer/token_dictionary_gc104M.pkl":
        "67c445f4385127adfc48dcc072320cd65d6822829bf27dd38070e6e787bc597f",
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: str, repo_file: str) -> None:
    """Exit unless ``path`` has the pinned SHA-256 for ``repo_file`` (fail closed if none is pinned)."""
    expected = PINNED_SHA256.get(repo_file)
    if expected is None:
        raise SystemExit(f"No pinned SHA-256 for {repo_file}; refusing to use it. Add it to PINNED_SHA256.")
    got = _sha256(path)
    if got != expected:
        raise SystemExit(
            f"{repo_file}: SHA-256 {got[:16]}... does not match the pinned {expected[:16]}... "
            f"(commit {REVISION[:7]}). The download is corrupt or was tampered with; delete it "
            "from the Hugging Face cache and run again."
        )


def _copy(src: str, dest: str) -> None:
    """Copy src -> dest, overwriting any existing (possibly read-only) destination."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # HF cache files are read-only; a prior copy of dest may be too. Clear it first
    # so shutil.copy does not fail with PermissionError on re-run.
    if os.path.exists(dest):
        os.chmod(dest, 0o644)
        os.remove(dest)
    shutil.copy(src, dest)
    os.chmod(dest, 0o644)


def _download(repo_file: str, dest: str) -> None:
    src = hf_hub_download(repo_id=REPO_ID, filename=repo_file, revision=REVISION)  # nosec B615 -- REVISION is a pinned commit SHA
    verify(src, repo_file)
    _copy(src, dest)
    print(f"  {repo_file}  ->  {dest}  ({os.path.getsize(dest)} bytes, SHA-256 verified)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=list(PRESETS), default="V1-10M")
    ap.add_argument("--repo", default="vendor/Geneformer", help="path to the cloned Geneformer repo")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    corpus = cfg["corpus"]
    repo_root = args.repo

    print(f"Fetching {args.preset} weights:")
    for f in MODEL_FILES:
        _download(f"{cfg['model_dir']}/{f}", os.path.join(repo_root, cfg["model_dir"], f))

    print(f"Fetching {corpus} dictionaries:")
    dict_targets = [os.path.join(repo_root, cfg["dict_subdir"])]
    # Also drop dictionaries into the INSTALLED geneformer package, which is what
    # the tokenizer actually loads from at runtime.
    try:
        import geneformer

        pkg_dir = os.path.dirname(geneformer.__file__)
        pkg_dict_dir = pkg_dir if cfg["dict_subdir"] == "geneformer" else os.path.join(pkg_dir, "gene_dictionaries_30m")
        dict_targets.append(pkg_dict_dir)
    except ImportError:
        print("  (geneformer not importable yet; skipping package copy)")

    for tmpl in DICT_FILES:
        fname = tmpl.format(corpus=corpus)
        repo_file = f"{cfg['dict_subdir']}/{fname}"
        # Download once, copy to all targets.
        src = hf_hub_download(repo_id=REPO_ID, filename=repo_file, revision=REVISION)  # nosec B615 -- REVISION is a pinned commit SHA
        verify(src, repo_file)
        for tgt_dir in dict_targets:
            dest = os.path.join(tgt_dir, fname)
            _copy(src, dest)
            print(f"  {fname}  ->  {dest}  ({os.path.getsize(dest)} bytes)")

    print("Done.")


if __name__ == "__main__":
    main()
