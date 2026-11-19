# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""scripts/check_audit_trail.sh reports whether a trail can attribute access to data buckets (T010)."""

from __future__ import annotations

import json
import os
import stat
import subprocess  # nosec B404 - runs the repository's own shell script with a stubbed aws
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_audit_trail.sh"

NOW = datetime.now(timezone.utc)
GOOD_STATUS = {"IsLogging": True, "LatestDeliveryTime": NOW.isoformat()}
GOOD_TRAIL = {"trailList": [{"IsMultiRegionTrail": True, "LogFileValidationEnabled": True}]}
ADV = {"AdvancedEventSelectors": [
    {"FieldSelectors": [
        {"Field": "eventCategory", "Equals": ["Data"]},
        {"Field": "resources.type", "Equals": ["AWS::S3::Object"]},
        {"Field": "resources.ARN", "StartsWith": ["arn:aws:s3:::data-b/"]},
    ]},
    {"FieldSelectors": [{"Field": "eventCategory", "Equals": ["Management"]}]},
]}
MGMT = "management write events are recorded from every source"


def _run(tmp_path, *, status=GOOD_STATUS, trail=GOOD_TRAIL, selectors=ADV, fail=False, buckets=("data-b",), profile="p"):
    aws = tmp_path / "aws"
    answers = {"get-trail-status": status, "describe-trails": trail, "get-event-selectors": selectors}
    lines = ["#!/usr/bin/env bash", "[ " + ("1" if fail else "0") + " -eq 1 ] && exit 254"]
    # a dash for the profile means no --profile at all; the stub fails any call that has one
    lines.append('case " $* " in *" --profile "*) [ "' + profile + '" = "-" ] && exit 254;; esac')
    for sub, payload in answers.items():
        lines.append(f"[ \"$2\" = {sub} ] && cat <<'JSON'\n{json.dumps(payload)}\nJSON")
    aws.write_text("\n".join(lines) + "\nexit 0\n")
    aws.chmod(aws.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    return subprocess.run(  # nosec B603 B607 - fixed argv, stubbed aws on PATH
        [str(SCRIPT), "t", "us-east-1", profile, *buckets], capture_output=True, text=True, env=env, check=False
    )


def test_a_healthy_trail_passes_every_check(tmp_path):
    out = _run(tmp_path)
    assert out.returncode == 0, out.stdout
    assert "FAIL" not in out.stdout and out.stdout.count("PASS") == 7


def test_not_logging_fails(tmp_path):
    out = _run(tmp_path, status={**GOOD_STATUS, "IsLogging": False})
    assert out.returncode == 1 and "FAIL trail is logging" in out.stdout


def test_a_delivery_error_or_stale_delivery_fails(tmp_path):
    out = _run(tmp_path, status={**GOOD_STATUS, "LatestDeliveryError": "AccessDenied"})
    assert out.returncode == 1 and "FAIL no log delivery error" in out.stdout
    old = (NOW - timedelta(days=3)).isoformat()
    out = _run(tmp_path, status={**GOOD_STATUS, "LatestDeliveryTime": old})
    assert out.returncode == 1 and "FAIL logs were delivered" in out.stdout
    out = _run(tmp_path, status={"IsLogging": True})
    assert out.returncode == 1 and "never delivered" in out.stdout


def test_a_bucket_without_data_events_fails(tmp_path):
    out = _run(tmp_path, buckets=("data-b", "other-b"))
    assert out.returncode == 1
    assert "PASS S3 data events (reads and writes) are recorded for data-b" in out.stdout
    assert "FAIL S3 data events (reads and writes) are recorded for other-b" in out.stdout


def test_a_prefix_that_only_looks_similar_does_not_count(tmp_path):
    sel = json.loads(json.dumps(ADV))
    sel["AdvancedEventSelectors"][0]["FieldSelectors"][2]["StartsWith"] = ["arn:aws:s3:::data-b-extra/"]
    assert _run(tmp_path, selectors=sel).returncode == 1


def test_write_only_selectors_do_not_count(tmp_path):
    sel = json.loads(json.dumps(ADV))
    sel["AdvancedEventSelectors"][0]["FieldSelectors"].append({"Field": "readOnly", "Equals": ["false"]})
    assert _run(tmp_path, selectors=sel).returncode == 1
    basic = {"EventSelectors": [{"ReadWriteType": "WriteOnly", "IncludeManagementEvents": True,
             "DataResources": [{"Type": "AWS::S3::Object", "Values": ["arn:aws:s3:::data-b/"]}]}]}
    assert _run(tmp_path, selectors=basic).returncode == 1
    basic["EventSelectors"][0]["ReadWriteType"] = "All"
    assert _run(tmp_path, selectors=basic).returncode == 0


def test_an_unreadable_trail_is_an_error_not_a_pass(tmp_path):
    out = _run(tmp_path, fail=True)
    assert out.returncode == 3 and "PASS" not in out.stdout


def test_usage_error(tmp_path):
    out = subprocess.run([str(SCRIPT)], capture_output=True, text=True, check=False)  # nosec B603
    assert out.returncode == 2


def test_a_dash_profile_uses_the_ambient_credentials(tmp_path):
    out = _run(tmp_path, profile="-")
    assert out.returncode == 0 and "FAIL" not in out.stdout
    assert _run(tmp_path, profile="p").returncode == 0


def test_a_data_only_selector_misses_management_events(tmp_path):
    # the live scfm-audit trail before 2026-11-18: one advanced selector, S3 data events only
    sel = {"AdvancedEventSelectors": ADV["AdvancedEventSelectors"][:1]}
    out = _run(tmp_path, selectors=sel)
    assert out.returncode == 1 and "FAIL " + MGMT in out.stdout
    assert "PASS S3 data events (reads and writes) are recorded for data-b" in out.stdout


def test_partial_management_selectors_do_not_count(tmp_path):
    for extra in ({"Field": "readOnly", "Equals": ["true"]},
                  {"Field": "eventSource", "NotEquals": ["kms.amazonaws.com"]}):
        sel = json.loads(json.dumps(ADV))
        sel["AdvancedEventSelectors"][1]["FieldSelectors"].append(extra)
        assert "FAIL " + MGMT in _run(tmp_path, selectors=sel).stdout
    sel = json.loads(json.dumps(ADV))
    sel["AdvancedEventSelectors"][1]["FieldSelectors"].append({"Field": "readOnly", "Equals": ["false"]})
    assert "PASS " + MGMT in _run(tmp_path, selectors=sel).stdout


def test_basic_selectors_need_management_writes_from_every_source(tmp_path):
    def basic(**kw):
        sel = {"ReadWriteType": "All", "IncludeManagementEvents": True,
               "DataResources": [{"Type": "AWS::S3::Object", "Values": ["arn:aws:s3:::data-b/"]}]}
        return {"EventSelectors": [{**sel, **kw}]}
    assert _run(tmp_path, selectors=basic()).returncode == 0
    for kw in ({"IncludeManagementEvents": False}, {"ReadWriteType": "ReadOnly"},
               {"ExcludeManagementEventSources": ["kms.amazonaws.com"]}):
        assert "FAIL " + MGMT in _run(tmp_path, selectors=basic(**kw)).stdout
