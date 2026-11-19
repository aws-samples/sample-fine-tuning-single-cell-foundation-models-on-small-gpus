# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""CodeBuild post-push step: build manifest and ECR scan report (threat T005)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "sagemaker" / "post_build.py"
_spec = importlib.util.spec_from_file_location("post_build", _SRC)
assert _spec and _spec.loader
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)

DIGEST = "sha256:" + "cd" * 32
IMAGE = f"123456789012.dkr.ecr.us-east-1.amazonaws.com/geneformer-ft@{DIGEST}"


class FakeAws:
    """Replays scripted ``aws`` results in order; an Exception in the script is raised."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[list[str]] = []

    def __call__(self, args):
        self.calls.append(args)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _scan(status="COMPLETE", **counts):
    return {"imageScanStatus": {"status": status}, "imageScanFindings": {"findingSeverityCounts": counts}}


def test_base_image_reads_the_first_from_and_skips_flags():
    text = "# c\nFROM --platform=linux/amd64 nvcr.io/nvidia/cuda:12.1@sha256:abc AS build\nFROM scratch\n"
    assert pb.base_image(text) == "nvcr.io/nvidia/cuda:12.1@sha256:abc"
    assert pb.base_image("RUN true\n") is None


def test_manifest_ties_the_digest_to_its_inputs(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM nvcr.io/nvidia/cuda:11.7.1@sha256:aa\nRUN true\n")
    lock = tmp_path / "req.txt"
    lock.write_text("torch==2.0.1\n")
    env = {"CODEBUILD_BUILD_ID": "proj:123", "CODEBUILD_RESOLVED_SOURCE_VERSION": "etag-9"}
    m = pb.build_manifest(
        image=IMAGE, repository="geneformer-ft", tag="b12", backend="geneformer",
        dockerfile=str(df), locks=[str(lock), str(tmp_path / "absent.lock")], env=env,
        now=datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc),
    )
    assert m["image"] == IMAGE and m["tag"] == "b12" and m["built_at"] == "2026-11-17T12:00:00Z"
    assert m["build_id"] == "proj:123" and m["source_version"] == "etag-9"
    assert m["dockerfile"]["sha256"] == hashlib.sha256(df.read_bytes()).hexdigest()
    assert m["locks"] == {str(lock): hashlib.sha256(lock.read_bytes()).hexdigest()}  # absent file skipped
    assert m["base_image"] == "nvcr.io/nvidia/cuda:11.7.1@sha256:aa"


def test_scan_is_started_when_none_exists_then_polled_to_completion():
    aws = FakeAws(
        pb.AwsError("no scan", "ScanNotFoundException"),
        {},  # start-image-scan
        _scan("IN_PROGRESS"),
        _scan("COMPLETE", CRITICAL=2, HIGH=5),
    )
    out = pb.fetch_scan("geneformer-ft", DIGEST, aws=aws, sleep=lambda s: None)
    assert out == {"status": "COMPLETE", "counts": {"CRITICAL": 2, "HIGH": 5}}
    assert aws.calls[1][:2] == ["ecr", "start-image-scan"]


def test_a_scan_rate_limit_on_start_is_tolerated_but_other_errors_are_not():
    ok = FakeAws(pb.AwsError("x", "ScanNotFoundException"), pb.AwsError("x", "LimitExceededException"),
                 _scan("COMPLETE", LOW=1))
    assert pb.fetch_scan("r", DIGEST, aws=ok, sleep=lambda s: None)["counts"] == {"LOW": 1}
    bad = FakeAws(pb.AwsError("denied", "AccessDeniedException"))
    with pytest.raises(pb.AwsError):
        pb.fetch_scan("r", DIGEST, aws=bad, sleep=lambda s: None)


def test_a_scan_that_never_finishes_times_out_instead_of_hanging():
    aws = FakeAws(*[_scan("IN_PROGRESS")] * 50)
    assert pb.fetch_scan("r", DIGEST, aws=aws, timeout=0, sleep=lambda s: None)["status"] == "TIMEOUT"


def test_enhanced_scanning_findings_are_counted_by_severity():
    resp = {"imageScanStatus": {"status": "ACTIVE"},
            "imageScanFindings": {"enhancedFindings": [{"severity": "high"}, {"severity": "HIGH"}, {}]}}
    out = pb.fetch_scan("r", DIGEST, aws=FakeAws(resp), sleep=lambda s: None)
    assert out["counts"] == {"HIGH": 2, "UNDEFINED": 1}


def test_breached_reports_only_the_requested_severities_that_have_findings():
    assert pb.breached({"CRITICAL": 1, "HIGH": 0, "LOW": 9}, ["CRITICAL", "HIGH"]) == ["CRITICAL"]
    assert pb.breached({"LOW": 9}, []) == []


def _run(monkeypatch, tmp_path, aws, *extra):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM nvcr.io/nvidia/cuda:12.1@sha256:aa\n")
    monkeypatch.setattr(pb, "run_aws", aws)
    argv = ["--repository", "geneformer-ft", "--tag", "b1", "--image", IMAGE, "--dockerfile", str(df), *extra]
    return pb.main(argv)


def test_main_prints_one_parseable_manifest_line_and_passes_by_default(monkeypatch, tmp_path, capsys):
    assert _run(monkeypatch, tmp_path, FakeAws(_scan("COMPLETE", CRITICAL=3))) == 0  # report only
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("BUILD_MANIFEST "))
    manifest = json.loads(line.removeprefix("BUILD_MANIFEST "))
    assert manifest["image"] == IMAGE and manifest["scan"]["counts"] == {"CRITICAL": 3}
    assert "IMAGE_SCAN status=COMPLETE findings: CRITICAL=3" in out


def test_main_fails_the_build_only_when_fail_on_is_breached(monkeypatch, tmp_path, capsys):
    assert _run(monkeypatch, tmp_path, FakeAws(_scan("COMPLETE", HIGH=1)), "--fail-on", "CRITICAL") == 0
    assert _run(monkeypatch, tmp_path, FakeAws(_scan("COMPLETE", CRITICAL=1)), "--fail-on", "critical,high") == 3
    assert "findings at CRITICAL" in capsys.readouterr().err


def test_main_scan_unavailable_warns_unless_a_scan_is_required(monkeypatch, tmp_path):
    def denied(args):
        raise pb.AwsError("denied", "AccessDeniedException")

    assert _run(monkeypatch, tmp_path, denied) == 0
    assert _run(monkeypatch, tmp_path, denied, "--scan-required") == 4


def test_main_rejects_bad_input(monkeypatch, tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM x\n")
    base = ["--repository", "r", "--tag", "t", "--dockerfile", str(df)]
    assert pb.main([*base, "--image", "123456789012.dkr.ecr.us-east-1.amazonaws.com/r:latest"]) == 2
    assert pb.main([*base, "--image", IMAGE, "--fail-on", "SEVERE"]) == 2


def test_a_fail_on_threshold_fails_closed_when_the_scan_never_arrives(monkeypatch, tmp_path, capsys):
    # Build b42: SCAN_FAIL_ON=CRITICAL was set, the scan was still IN_PROGRESS at the deadline,
    # and the build passed without its image being checked.
    monkeypatch.setattr(pb, "fetch_scan", lambda *a, **k: {"status": "TIMEOUT", "counts": {}})
    assert _run(monkeypatch, tmp_path, FakeAws(_scan("COMPLETE")), "--fail-on", "CRITICAL") == 4
    assert "missing scan fails the build" in capsys.readouterr().err
    assert _run(monkeypatch, tmp_path, FakeAws(_scan("COMPLETE"))) == 0  # report-only stays lenient


def test_scan_timeout_defaults_to_twenty_minutes_and_reads_the_environment(monkeypatch, tmp_path):
    seen = {}

    def fake_fetch(*a, timeout, **k):
        seen["timeout"] = timeout
        return {"status": "COMPLETE", "counts": {}}

    monkeypatch.setattr(pb, "fetch_scan", fake_fetch)
    monkeypatch.delenv("SCAN_TIMEOUT", raising=False)
    _run(monkeypatch, tmp_path, FakeAws(_scan("COMPLETE")))
    assert seen["timeout"] == 1200
    monkeypatch.setenv("SCAN_TIMEOUT", "300")
    _run(monkeypatch, tmp_path, FakeAws(_scan("COMPLETE")))
    assert seen["timeout"] == 300
