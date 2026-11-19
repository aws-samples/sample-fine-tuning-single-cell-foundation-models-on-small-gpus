# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""CodeBuild post-push step: record what was built and report the ECR image scan.

Runs after ``docker push`` (see ``buildspec.yml``). Uses only the standard library and
the ``aws`` CLI that CodeBuild provides.

1. **Build manifest.** Prints one ``BUILD_MANIFEST {json}`` log line tying the pushed
   image digest to the inputs that produced it: the Dockerfile and lock files (SHA-256),
   the base image reference, the CodeBuild build and source version, and the time. To
   check later that a running job used a known build, look the digest up in the build log.
2. **Scan report.** Waits for the ECR image scan (basic scanning on push, or started here
   when the repository does not scan on push) and prints the finding counts by severity.
   Reporting never fails the build. ``--fail-on CRITICAL,HIGH`` (or ``SCAN_FAIL_ON``)
   makes the build fail when the image has findings at those severities, and also when no
   scan result arrives in time (a gate that cannot see the scan fails closed). It is off by
   default because the pinned CUDA and torch 2.0.1 stacks are old enough to carry findings
   no rebuild can clear; choose a threshold after reading the first report.

Exit status: 0 on success, 3 when ``--fail-on`` is breached, 4 when ``--fail-on`` or
``--scan-required`` is set and no scan result could be obtained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess  # nosec B404 - runs only the aws CLI with a fixed argument list
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL", "UNDEFINED")
_TERMINAL = {"COMPLETE", "FAILED", "UNSUPPORTED_IMAGE", "FINDINGS_UNAVAILABLE", "ACTIVE"}


class AwsError(RuntimeError):
    """An ``aws`` CLI call failed; ``code`` carries the AWS error code when present."""

    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.code = code


def run_aws(args: list[str]) -> dict:
    """Run ``aws <args> --output json`` and return the parsed result (``{}`` if empty)."""
    proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell, aws from the build image
        ["aws", *args, "--output", "json"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        m = re.search(r"An error occurred \((\w+)\)", proc.stderr)
        raise AwsError(proc.stderr.strip()[-300:], m.group(1) if m else "")
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def base_image(dockerfile_text: str) -> str | None:
    """The first ``FROM`` image reference in a Dockerfile (flags such as --platform skipped)."""
    m = re.search(r"^FROM\s+(?:--\S+\s+)*(\S+)", dockerfile_text, re.MULTILINE | re.IGNORECASE)
    return m.group(1) if m else None


def build_manifest(
    *,
    image: str,
    repository: str,
    tag: str,
    backend: str,
    dockerfile: str,
    locks: list[str],
    env: dict[str, str] | None = None,
    now: datetime | None = None,
) -> dict:
    """Describe the image just pushed and the inputs it was built from."""
    env = os.environ if env is None else env
    return {
        "schema": 1,
        "image": image,
        "repository": repository,
        "tag": tag,
        "backend": backend,
        "built_at": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_id": env.get("CODEBUILD_BUILD_ID", ""),
        "source_version": env.get("CODEBUILD_RESOLVED_SOURCE_VERSION", ""),
        "dockerfile": {"path": dockerfile, "sha256": sha256_file(dockerfile)},
        "base_image": base_image(Path(dockerfile).read_text()),
        "locks": {p: sha256_file(p) for p in locks if Path(p).is_file()},
    }


def _counts(findings: dict) -> dict[str, int]:
    counts = dict(findings.get("findingSeverityCounts") or {})
    if not counts:  # enhanced scanning reports individual findings instead
        for item in findings.get("enhancedFindings") or []:
            sev = str(item.get("severity", "UNDEFINED")).upper()
            counts[sev] = counts.get(sev, 0) + 1
    return {k: int(v) for k, v in counts.items()}


def fetch_scan(
    repository: str,
    digest: str,
    *,
    aws: Callable[[list[str]], dict] = run_aws,
    timeout: float = 600.0,
    sleep: Callable[[float], None] = time.sleep,
    poll: float = 10.0,
) -> dict:
    """Return ``{"status": ..., "counts": {...}}`` for an image, starting a scan if needed."""
    image_id = f"imageDigest={digest}"
    base = ["ecr", "describe-image-scan-findings", "--repository-name", repository, "--image-id", image_id]
    started = False
    deadline = time.monotonic() + timeout
    while True:
        try:
            resp = aws(base)
        except AwsError as e:
            if e.code == "ScanNotFoundException" and not started:
                started = True
                try:
                    aws(["ecr", "start-image-scan", "--repository-name", repository, "--image-id", image_id])
                except AwsError as start_err:
                    if start_err.code != "LimitExceededException":
                        raise
            elif e.code != "ScanNotFoundException":
                raise
            resp = {}
        status = (resp.get("imageScanStatus") or {}).get("status", "PENDING")
        if status in _TERMINAL and resp:
            return {"status": status, "counts": _counts(resp.get("imageScanFindings") or {})}
        if time.monotonic() >= deadline:
            return {"status": "TIMEOUT", "counts": {}}
        sleep(poll)


def breached(counts: dict[str, int], fail_on: list[str]) -> list[str]:
    """The severities in ``fail_on`` that have at least one finding."""
    return [s for s in fail_on if counts.get(s, 0) > 0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repository", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--image", required=True, help="pushed image as <registry>/<repo>@sha256:<digest>")
    p.add_argument("--backend", default="geneformer")
    p.add_argument("--dockerfile", required=True)
    p.add_argument("--lock", action="append", default=[], help="lock/requirements file to hash (repeatable)")
    p.add_argument("--manifest-out", default=None)
    p.add_argument("--fail-on", default=os.environ.get("SCAN_FAIL_ON", ""))
    p.add_argument("--scan-required", action="store_true")
    p.add_argument("--scan-timeout", type=float, default=float(os.environ.get("SCAN_TIMEOUT", "1200")))
    args = p.parse_args(argv)

    digest = args.image.rsplit("@", 1)[-1]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        print(f"post_build: --image must end in @sha256:<digest>, got {args.image!r}", file=sys.stderr)
        return 2
    fail_on = [s.strip().upper() for s in args.fail_on.split(",") if s.strip()]
    unknown = [s for s in fail_on if s not in SEVERITIES]
    if unknown:
        print(f"post_build: unknown severities in --fail-on: {unknown}", file=sys.stderr)
        return 2

    manifest = build_manifest(
        image=args.image, repository=args.repository, tag=args.tag, backend=args.backend,
        dockerfile=args.dockerfile, locks=args.lock,
    )
    try:
        scan = fetch_scan(args.repository, digest, aws=run_aws, timeout=args.scan_timeout)
    except AwsError as e:
        scan = {"status": "UNAVAILABLE", "counts": {}, "error": str(e)[:200]}
    manifest["scan"] = scan

    print("BUILD_MANIFEST " + json.dumps(manifest, sort_keys=True))
    if args.manifest_out:
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    counts = scan["counts"]
    summary = ", ".join(f"{s}={counts[s]}" for s in SEVERITIES if s in counts) or "none"
    print(f"IMAGE_SCAN status={scan['status']} findings: {summary}")
    if scan["status"] not in ("COMPLETE", "ACTIVE"):
        print(f"post_build: no scan result ({scan.get('error', scan['status'])}).", file=sys.stderr)
        if fail_on:
            print(f"post_build: --fail-on {','.join(fail_on)} is set, so a missing scan fails the build.", file=sys.stderr)
            return 4
        return 4 if args.scan_required else 0
    hit = breached(counts, fail_on)
    if hit:
        print(f"post_build: image has findings at {', '.join(hit)} (--fail-on {','.join(fail_on)}).", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
