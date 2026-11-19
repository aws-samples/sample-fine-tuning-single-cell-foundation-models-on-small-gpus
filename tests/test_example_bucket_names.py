# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Example S3 bucket names readers might copy must not be squattable."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["README.md", "docs", "scripts", "sagemaker", "configs"]
SUFFIXES = {".md", ".sh", ".py", ".yaml", ".yml", ".json"}


def _files():
    for entry in SOURCES:
        path = ROOT / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (p for p in path.rglob("*") if p.suffix in SUFFIXES and p.is_file())


def test_example_s3_buckets_are_placeholders_or_reserved_demo_names():
    # A literal like s3://my-bucket/ can be registered by anyone; a reader who copies it
    # would read from or write to someone else's bucket. Bucket names in examples must be
    # a placeholder (<bucket>, $ACCT, {bucket}) or use the amzn-s3-demo- prefix that AWS
    # reserves for documentation.
    bad = []
    for path in _files():
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for bucket in re.findall(r"s3://([^/\s\"'`)]+)", line):
                if any(c in bucket for c in "<${") or bucket.startswith("amzn-s3-demo-"):
                    continue
                bad.append(f"{path.relative_to(ROOT)}:{n}: s3://{bucket}")
    assert not bad, "squattable example bucket names:\n" + "\n".join(bad)
