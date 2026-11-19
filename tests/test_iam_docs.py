# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Shape checks for the optional IAM policy documents (no whole-resource wildcards
except the ones that are documented as accepted)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

IAM = Path(__file__).resolve().parents[1] / "docs" / "iam"
POLICIES = [
    "codebuild-service-role-policy.json",
    "sagemaker-execution-kms-policy.json",
    "sagemaker-execution-vpc-policy.json",
    "sagemaker-launcher-policy.json",
]


def _statements(name):
    return json.loads((IAM / name).read_text())["Statement"]


def _as_list(v):
    return v if isinstance(v, list) else [v]


@pytest.mark.parametrize("name", POLICIES)
def test_wildcard_resources_are_named_as_accepted_risks(name):
    for st in _statements(name):
        if st["Effect"] == "Allow" and "*" in _as_list(st["Resource"]):
            assert "AcceptedRisk" in st["Sid"] or st["Action"] == "ecr:GetAuthorizationToken", st["Sid"]


@pytest.mark.parametrize("name", POLICIES)
def test_no_service_wide_actions(name):
    for st in _statements(name):
        for action in _as_list(st["Action"]):
            assert not action.endswith(":*") and action != "*", action
            assert not action.startswith("iam:") or action == "iam:PassRole", action


def test_codebuild_policy_is_scoped_to_this_project_and_repos():
    text = (IAM / "codebuild-service-role-policy.json").read_text()
    assert "/aws/codebuild/geneformer-ft-build" in text
    for st in _statements("codebuild-service-role-policy.json"):
        if st["Sid"].startswith(("Push", "Manage")):
            assert all(r.rsplit("/", 1)[-1] in {"geneformer-ft", "scgpt-ft"} for r in _as_list(st["Resource"]))
        if st["Sid"].startswith("ReadBuildSource"):
            assert "geneformer-src.zip" in st["Resource"]


def test_codebuild_trust_is_pinned_to_account_and_project():
    st = json.loads((IAM / "codebuild-trust-policy.json").read_text())["Statement"]
    assert len(st) == 1
    assert st[0]["Principal"] == {"Service": "codebuild.amazonaws.com"}
    cond = st[0]["Condition"]
    assert "aws:SourceAccount" in cond["StringEquals"]
    assert cond["ArnLike"]["aws:SourceArn"].endswith(":project/geneformer-ft-build")


def test_kms_policy_names_a_single_key():
    (st,) = _statements("sagemaker-execution-kms-policy.json")
    assert st["Resource"].startswith("arn:aws:kms:") and "*" not in st["Resource"]


def test_s3_bucket_policy_template_denies_plain_http_and_has_placeholders():
    policy = json.loads((IAM.parent / "security" / "s3-bucket-policy.json").read_text())
    tls = next(s for s in policy["Statement"] if s["Sid"] == "DenyNonTlsAccess")
    assert tls["Effect"] == "Deny"
    assert tls["Condition"] == {"Bool": {"aws:SecureTransport": "false"}}
    assert "arn:aws:s3:::BUCKET_NAME/*" in tls["Resource"]


_KEY = "arn:aws:kms:REGION:ACCOUNT_ID:key/KEY_ID"
SSE = "s3:x-amz-server-side-encryption"
SSE_KEY = "s3:x-amz-server-side-encryption-aws-kms-key-id"


def _denied(policy: dict, action: str, context: dict) -> bool:
    """Whether any Deny statement matches, for the operators the template uses.

    Conditions in one statement are ANDed. StringEquals is false for a missing key and
    StringNotEquals is true for one (IAM docs, "Condition operators"); Null tests presence.
    """
    def matches(op: str, key: str, want) -> bool:
        have = context.get(key)
        if op == "Null":
            return (have is None) == (want == "true")
        if op == "StringEquals":
            return have == want
        if op == "StringNotEquals":
            return have != want
        if op == "StringNotEqualsIfExists":
            return have is None or have != want
        if op == "ArnNotLike":
            import fnmatch

            wants = want if isinstance(want, list) else [want]
            return have is None or not any(fnmatch.fnmatchcase(have, w) for w in wants)
        if op == "Bool":
            return have == want
        raise AssertionError(f"evaluator does not know {op}")

    for st in policy["Statement"]:
        actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
        if st["Effect"] != "Deny" or not any(a == action or a == "s3:*" for a in actions):
            continue
        if all(matches(op, k, v) for op, kv in st.get("Condition", {}).items() for k, v in kv.items()):
            return True
    return False


@pytest.mark.parametrize(
    ("headers", "denied"),
    [
        ({}, False),  # no header: bucket default key applies, so plain `aws s3 sync` works
        ({SSE: "aws:kms", SSE_KEY: _KEY}, False),  # SageMaker output with the data key
        ({SSE: "AES256"}, True),  # SSE-S3 would bypass the data key
        ({SSE: "aws:kms"}, True),  # no key id: S3 would use aws/s3
        ({SSE: "aws:kms", SSE_KEY: "arn:aws:kms:REGION:ACCOUNT_ID:key/OTHER"}, True),
        ({SSE: "aws:kms", SSE_KEY: "KEY_ID"}, True),  # a bare key id does not match the ARN
        ({SSE: "aws:kms:dsse", SSE_KEY: _KEY}, True),
    ],
)
def test_s3_bucket_policy_keeps_every_upload_under_the_data_key(headers, denied):
    policy = json.loads((IAM.parent / "security" / "s3-bucket-policy.json").read_text())
    context = {"aws:SecureTransport": "true", **headers}
    assert _denied(policy, "s3:PutObject", context) is denied


def test_s3_bucket_policy_still_denies_plain_http_reads():
    policy = json.loads((IAM.parent / "security" / "s3-bucket-policy.json").read_text())
    assert _denied(policy, "s3:GetObject", {"aws:SecureTransport": "false"})
    assert not _denied(policy, "s3:GetObject", {"aws:SecureTransport": "true"})


TRAIL = "arn:aws:cloudtrail:REGION:ACCOUNT_ID:trail/TRAIL_NAME"
READER = "arn:aws:iam::ACCOUNT_ID:role/LOG_READER_ROLE"
OTHER_ROLE = "arn:aws:iam::ACCOUNT_ID:role/cdk-hnb659fds-lookup-role-ACCOUNT_ID-REGION"


def _log_policy() -> dict:
    return json.loads((IAM.parent / "security" / "log-bucket-policy.json").read_text())


@pytest.mark.parametrize(
    ("action", "context", "denied"),
    [
        # CloudTrail delivering for this trail: the one writer the bucket must keep.
        ("s3:PutObject", {"aws:SourceArn": TRAIL}, False),
        # Any person or role writing, or a different trail: refused, so a log cannot be replaced.
        ("s3:PutObject", {"aws:PrincipalArn": READER}, True),
        ("s3:PutObject", {"aws:PrincipalArn": OTHER_ROLE}, True),
        ("s3:PutObject", {"aws:SourceArn": "arn:aws:cloudtrail:REGION:ACCOUNT_ID:trail/other"}, True),
        # Nobody deletes, not even the readers (lifecycle expiry is S3's own and not affected).
        ("s3:DeleteObject", {"aws:PrincipalArn": READER}, True),
        ("s3:DeleteObjectVersion", {"aws:PrincipalArn": READER}, True),
        ("s3:DeleteObject", {"aws:SourceArn": TRAIL}, True),
        # Nobody bypasses a governance-mode Object Lock retention, so a lock cannot be shortened
        # or a locked version deleted without first changing this policy (which alerts).
        ("s3:BypassGovernanceRetention", {"aws:PrincipalArn": READER}, True),
        ("s3:BypassGovernanceRetention", {"aws:PrincipalArn": OTHER_ROLE}, True),
        # Only the listed reader roles read log files.
        ("s3:GetObject", {"aws:PrincipalArn": READER}, False),
        ("s3:GetObjectVersion", {"aws:PrincipalArn": READER}, False),
        ("s3:GetObject", {"aws:PrincipalArn": OTHER_ROLE}, True),
        ("s3:GetObject", {"aws:PrincipalArn": "arn:aws:iam::ACCOUNT_ID:role/GeneformerSageMakerExecution"}, True),
    ],
)
def test_log_bucket_policy_keeps_logs_append_only_and_admin_readable(action, context, denied):
    assert _denied(_log_policy(), action, {"aws:SecureTransport": "true", **context}) is denied


def test_log_bucket_policy_keeps_the_cloudtrail_grants_and_tls():
    by_sid = {s["Sid"]: s for s in _log_policy()["Statement"]}
    write = by_sid["AWSCloudTrailWrite"]
    assert write["Effect"] == "Allow" and write["Principal"] == {"Service": "cloudtrail.amazonaws.com"}
    assert write["Condition"]["StringEquals"]["aws:SourceArn"] == TRAIL
    assert by_sid["AWSCloudTrailAclCheck"]["Action"] == "s3:GetBucketAcl"
    assert _denied(_log_policy(), "s3:GetObject", {"aws:SecureTransport": "false", "aws:PrincipalArn": READER})
    # The write exception names the same trail as the allow, so the two cannot drift apart.
    deny = by_sid["DenyLogWritesExceptTheTrail"]
    assert deny["Condition"]["StringNotEqualsIfExists"]["aws:SourceArn"] == TRAIL


def test_security_docs_link_to_files_that_exist():
    import re

    for md in (IAM.parent / "security").glob("*.md"):
        for target in re.findall(r"\]\(([^)#]+)\)", md.read_text()):
            if not target.startswith("http"):
                assert (md.parent / target).exists(), f"{md.name}: {target}"


def test_launcher_policy_scopes_passrole_to_one_role_and_service():
    stmts = {s["Sid"]: s for s in _statements("sagemaker-launcher-policy.json")}
    pr = stmts["PassOnlyTheTrainingRoleToSageMaker"]
    assert pr["Action"] == "iam:PassRole"
    assert "*" not in pr["Resource"]
    assert pr["Condition"] == {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}
    assert stmts["DenyUnapprovedInstanceTypes"]["Effect"] == "Deny"
    assert stmts["DenyJobsLongerThanTwelveHours"]["Effect"] == "Deny"
    # Only actions that support no resource ARN may use "*" (ListTrainingJobs has no resource type).
    star = [s for s in stmts.values() if s["Effect"] == "Allow" and "*" in _as_list(s["Resource"])]
    assert [_as_list(s["Action"]) for s in star] == [["sagemaker:ListTrainingJobs"]]
    assert not any(a.endswith(":*") or a == "*" for s in stmts.values() for a in _as_list(s["Action"]))


def test_launcher_policy_lets_the_sdk_find_the_default_bucket():
    # launch.py calls Session.default_bucket(). SageMaker SDK 2.219.0 then calls HeadBucket on
    # sagemaker-REGION-ACCOUNT_ID (s3:ListBucket), twice; boto3 ignores a denied ListBuckets.
    # Without s3:ListBucket every launch fails before CreateTrainingJob.
    stmt = next(s for s in _statements("sagemaker-launcher-policy.json") if s["Sid"] == "FindTheDefaultBucket")
    assert stmt["Effect"] == "Allow" and _as_list(stmt["Action"]) == ["s3:ListBucket"]
    assert _as_list(stmt["Resource"]) == ["arn:aws:s3:::sagemaker-REGION-ACCOUNT_ID"]
    assert "default_bucket()" in (IAM.parents[1] / "sagemaker" / "launch.py").read_text()


def test_budget_template_takes_the_email_and_limit_as_parameters():
    # Anyone setting this project up in their own account passes their own address and limit;
    # nothing account-specific is baked in.
    import yaml

    path = IAM.parent / "security" / "budget.yaml"
    text = path.read_text()
    t = yaml.safe_load(text)
    params = t["Parameters"]
    assert "Default" not in params["AlertEmail"] and "Default" not in params["MonthlyLimitUsd"]
    assert params["MonthlyLimitUsd"]["Type"] == "Number"
    (res,) = t["Resources"].values()
    assert res["Type"] == "AWS::Budgets::Budget"
    b = res["Properties"]["Budget"]
    assert (b["BudgetType"], b["TimeUnit"]) == ("COST", "MONTHLY")
    assert b["BudgetLimit"] == {"Amount": {"Ref": "MonthlyLimitUsd"}, "Unit": "USD"}
    assert b["CostFilters"] == {"Service": ["Amazon SageMaker"]}
    notes = res["Properties"]["NotificationsWithSubscribers"]
    kinds = {(n["Notification"]["NotificationType"], n["Notification"]["Threshold"]) for n in notes}
    assert kinds == {("ACTUAL", 80), ("FORECASTED", 100), ("ACTUAL", 100)}
    for n in notes:
        assert n["Notification"]["ThresholdType"] == "PERCENTAGE"
        assert n["Subscribers"] == [{"SubscriptionType": "EMAIL", "Address": {"Ref": "AlertEmail"}}]
    import re

    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    assert all(e.endswith("@example.com") for e in emails), emails
    assert not re.search(r"\b\d{12}\b", text)


def test_launcher_policy_allows_the_bucket_reads_that_regulated_data_preflight_makes():
    # launch.py --regulated-data calls these S3 APIs; without the matching read-only actions the
    # launch fails closed with "cannot check ...". Keep the policy and the code in step.
    import re

    api_to_action = {
        "get_bucket_versioning": "s3:GetBucketVersioning",
        "get_public_access_block": "s3:GetBucketPublicAccessBlock",
        "get_bucket_encryption": "s3:GetEncryptionConfiguration",
        "get_bucket_policy": "s3:GetBucketPolicy",
    }
    src = (IAM.parents[1] / "sagemaker" / "launch.py").read_text()
    used = set(re.findall(r"s3\.(get_[a-z_]+)\(", src))
    assert used == set(api_to_action), used
    stmt = next(s for s in _statements("sagemaker-launcher-policy.json") if s["Sid"] == "CheckBucketBaselineForRegulatedData")
    assert stmt["Effect"] == "Allow"
    assert set(_as_list(stmt["Action"])) == set(api_to_action.values())
    assert "*" not in _as_list(stmt["Resource"]) and all(r.startswith("arn:aws:s3:::") for r in _as_list(stmt["Resource"]))
