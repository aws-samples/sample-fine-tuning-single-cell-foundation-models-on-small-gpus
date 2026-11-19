#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Read-only check that a CloudTrail trail can attribute access to the buckets that hold training
# data (T010). It does not create or change anything.
#
#   scripts/check_audit_trail.sh TRAIL REGION PROFILE BUCKET [BUCKET...]
#
# Pass - as PROFILE to use the credentials already in the environment (for example in
# AWS CloudShell, which has no named profile).
#
# Prints one PASS or FAIL line per check and exits 1 if any check fails, 2 on bad usage, 3 if
# CloudTrail could not be read. Checks: the trail is logging, is multi-region, has log file
# validation, delivered logs in the last 24 hours with no delivery error, records management
# write events from every source (IAM, KMS, S3 bucket, ECR and CodeBuild changes), and records S3
# object (data) events for every BUCKET named. It does not check alarms or who can read the log
# bucket.

audit_trail_check() {
  local trail="$1" region="$2" profile="$3"; shift 3
  local status selectors desc
  local -a prof=(--profile "${profile}")
  [[ "${profile}" == "-" ]] && prof=()
  status="$(aws cloudtrail get-trail-status --name "${trail}" --region "${region}" ${prof[@]+"${prof[@]}"} --output json 2>/dev/null)" || return 3
  selectors="$(aws cloudtrail get-event-selectors --trail-name "${trail}" --region "${region}" ${prof[@]+"${prof[@]}"} --output json 2>/dev/null)" || return 3
  desc="$(aws cloudtrail describe-trails --trail-name-list "${trail}" --region "${region}" ${prof[@]+"${prof[@]}"} --output json 2>/dev/null)" || return 3
  STATUS="${status}" SELECTORS="${selectors}" DESC="${desc}" python3 - "$@" <<'PY'
import json, os, sys
from datetime import datetime, timedelta, timezone

buckets = sys.argv[1:]
status = json.loads(os.environ["STATUS"])
selectors = json.loads(os.environ["SELECTORS"])
trails = json.loads(os.environ["DESC"]).get("trailList", [])
trail = trails[0] if trails else {}
failed = False


def report(ok, text):
    global failed
    failed = failed or not ok
    print(("PASS " if ok else "FAIL ") + text)


report(bool(status.get("IsLogging")), "trail is logging")
report(bool(trail.get("IsMultiRegionTrail")), "trail is multi-region")
report(bool(trail.get("LogFileValidationEnabled")), "log file validation is on")
error = status.get("LatestDeliveryError")
report(not error, "no log delivery error" + (f" ({error})" if error else ""))
latest = status.get("LatestDeliveryTime")
fresh = False
if latest:
    when = datetime.fromisoformat(str(latest).replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    fresh = datetime.now(timezone.utc) - when < timedelta(hours=24)
report(fresh, "logs were delivered in the last 24 hours" + ("" if latest else " (never delivered)"))


def covers(bucket):
    arn = f"arn:aws:s3:::{bucket}"
    for sel in selectors.get("EventSelectors") or []:
        if sel.get("ReadWriteType") == "WriteOnly":
            continue
        for res in sel.get("DataResources") or []:
            if res.get("Type") == "AWS::S3::Object" and any(
                v in ("arn:aws:s3", arn + "/", arn + "/*") or v.startswith(arn + "/")
                for v in res.get("Values", [])
            ):
                return True
    for sel in selectors.get("AdvancedEventSelectors") or []:
        fields = {f["Field"]: f for f in sel.get("FieldSelectors", [])}
        if fields.get("eventCategory", {}).get("Equals") != ["Data"]:
            continue
        if fields.get("resources.type", {}).get("Equals") != ["AWS::S3::Object"]:
            continue
        if "readOnly" in fields and fields["readOnly"].get("Equals") == ["false"]:
            continue
        arns = fields.get("resources.ARN")
        if arns is None:
            return True
        if any(a == arn + "/" or a.startswith(arn + "/") for a in arns.get("StartsWith", [])):
            return True
        if any(a == arn + "/" or a.startswith(arn + "/") for a in arns.get("Equals", [])):
            return True
    return False


def management():
    for sel in selectors.get("EventSelectors") or []:
        if (sel.get("IncludeManagementEvents") and sel.get("ReadWriteType") != "ReadOnly"
                and not sel.get("ExcludeManagementEventSources")):
            return True
    for sel in selectors.get("AdvancedEventSelectors") or []:
        fields = {f["Field"]: f for f in sel.get("FieldSelectors", [])}
        if fields.get("eventCategory", {}).get("Equals") != ["Management"]:
            continue
        if fields.get("readOnly", {}).get("Equals") == ["true"]:
            continue
        if "eventSource" in fields:  # a selector limited to some sources is not full coverage
            continue
        return True
    return False


report(management(), "management write events are recorded from every source")
for b in buckets:
    report(covers(b), f"S3 data events (reads and writes) are recorded for {b}")
sys.exit(1 if failed else 0)
PY
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  [[ $# -ge 4 ]] || { echo "usage: $0 TRAIL REGION PROFILE BUCKET [BUCKET...]" >&2; exit 2; }
  set -uo pipefail
  audit_trail_check "$@"
  rc=$?
  [[ ${rc} -eq 3 ]] && echo "ERROR: could not read the trail with CloudTrail" >&2
  exit "${rc}"
fi
