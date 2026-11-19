# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The audit-alert template (T010): every rule matches the events it is meant to catch and
ignores the ones it must not.

EventBridge is not called. ``matches`` implements the subset of EventBridge pattern syntax the
template uses (lists of exact values, nested objects, lists of objects, and
``anything-but``/``prefix``), so the events below are checked against the patterns as written.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

TEMPLATE = Path(__file__).resolve().parents[1] / "docs" / "security" / "audit-alerts.yaml"
ACCOUNT, REGION = "123456789012", "us-east-1"
PARAMS = {
    "AlertEmail": "alerts@example.com",
    "DataBucket": "amzn-s3-demo-data",
    "ProtectedBuckets": ["amzn-s3-demo-data", "amzn-s3-demo-weights"],
    "DataKeyArn": f"arn:aws:kms:{REGION}:{ACCOUNT}:key/1111aaaa-2222-3333-4444-555566667777",
    "WeightsBucket": "amzn-s3-demo-weights",
    "WeightsKeyArn": f"arn:aws:kms:{REGION}:{ACCOUNT}:key/9999aaaa-2222-3333-4444-555566667777",
    "ExecutionRoleName": "GeneformerSageMakerExecution",
    "WatchedRoleNames": ["GeneformerSageMakerExecution", "geneformer-codebuild-role", "GeneformerLauncher"],
    "RepositoryNames": ["geneformer-ft", "scgpt-ft"],
    "BuildProjectName": "geneformer-ft-build",
}


def _template() -> dict:
    return yaml.safe_load(TEMPLATE.read_text())


def _resolve(node):
    """Substitute Ref and Fn::Sub with PARAMS (enough for this template's EventPatterns)."""
    if isinstance(node, dict):
        if set(node) == {"Ref"}:
            return copy.deepcopy(PARAMS[node["Ref"]])
        if set(node) == {"Fn::Sub"}:
            text = (node["Fn::Sub"].replace("${AWS::AccountId}", ACCOUNT).replace("${AWS::Region}", REGION)
                    .replace("${AWS::Partition}", "aws"))
            for k, v in PARAMS.items():
                if isinstance(v, str):
                    text = text.replace("${" + k + "}", v)
            assert "${" not in text, text
            return text
        return {k: _resolve(v) for k, v in node.items()}
    if isinstance(node, list):
        out = []
        for item in node:
            value = _resolve(item)
            # A Ref to a list parameter inside a list flattens, as CloudFormation does.
            out.extend(value) if isinstance(item, dict) and "Ref" in item and isinstance(value, list) else out.append(value)
        return out
    return node


_MISSING = object()


def _value_matches(rule, value) -> bool:
    if isinstance(rule, dict) and "exists" in rule:
        return (value is not _MISSING) is rule["exists"]
    if value is _MISSING:
        return False
    if isinstance(rule, dict) and "anything-but" in rule:
        spec = rule["anything-but"]
        if value is None:
            return False
        if isinstance(spec, dict) and "prefix" in spec:
            return not str(value).startswith(spec["prefix"])
        return value not in (spec if isinstance(spec, list) else [spec])
    return value == rule


def matches(pattern: dict, event) -> bool:
    for key, rule in pattern.items():
        if key == "$or":
            if not any(matches(branch, event) for branch in rule):
                return False
            continue
        value = event.get(key, _MISSING) if isinstance(event, dict) else _MISSING
        if isinstance(rule, dict):
            items = value if isinstance(value, list) else [value]
            if not any(isinstance(v, dict) and matches(rule, v) for v in items):
                return False
            continue
        values = value if isinstance(value, list) else [value]
        if not any(_value_matches(r, v) for r in rule for v in values):
            return False
    return True


def _rules() -> dict[str, dict]:
    res = _template()["Resources"]
    return {name: _resolve(r["Properties"]["EventPattern"]) for name, r in res.items() if r["Type"] == "AWS::Events::Rule"}


def _event(source, name, *, params=None, arn=f"arn:aws:sts::{ACCOUNT}:assumed-role/Admin/someone", resources=None,
           identity=None):
    service = source.split(".", 1)[1]
    detail = {"eventSource": f"{service}.amazonaws.com", "eventName": name, "requestParameters": params or {},
              "userIdentity": identity if identity is not None else {"arn": arn}}
    if resources is not None:
        detail["resources"] = resources
    return {"source": source, "detail-type": "AWS API Call via CloudTrail", "detail": detail}


JOB = f"arn:aws:sts::{ACCOUNT}:assumed-role/GeneformerSageMakerExecution/SageMaker"

# (rule, event, should alert)
CASES = [
    ("BucketProtectionChanged", _event("aws.s3", "DeleteBucketPolicy", params={"bucketName": "amzn-s3-demo-data"}), True),
    ("BucketProtectionChanged", _event("aws.s3", "PutBucketVersioning", params={"bucketName": "amzn-s3-demo-weights"}), True),
    ("BucketProtectionChanged", _event("aws.s3", "PutBucketPolicy", params={"bucketName": "someone-elses-bucket"}), False),
    # Turning on Object Lock, or changing or removing its default retention. CloudTrail records the
    # PutObjectLockConfiguration API as PutBucketObjectLockConfiguration (live event, 2026-11-18).
    ("BucketProtectionChanged", _event("aws.s3", "PutBucketObjectLockConfiguration", params={"bucketName": "amzn-s3-demo-data"}), True),
    ("BucketProtectionChanged", _event("aws.s3", "GetBucketObjectLockConfiguration", params={"bucketName": "amzn-s3-demo-data"}), False),
    ("BucketProtectionChanged", _event("aws.s3", "GetBucketPolicy", params={"bucketName": "amzn-s3-demo-data"}), False),
    ("DataAccessedOutsideJob", _event("aws.s3", "GetObject", params={"bucketName": "amzn-s3-demo-data"}), True),
    ("DataAccessedOutsideJob", _event("aws.s3", "DeleteObject", params={"bucketName": "amzn-s3-demo-data"}), True),
    ("DataAccessedOutsideJob", _event("aws.s3", "GetObject", params={"bucketName": "amzn-s3-demo-data"}, arn=JOB), False),
    ("DataAccessedOutsideJob", _event("aws.s3", "PutObject", params={"bucketName": "amzn-s3-demo-data"}, arn=JOB), False),
    ("DataAccessedOutsideJob", _event("aws.s3", "GetObject", params={"bucketName": "amzn-s3-demo-weights"}), False),
    ("DataAccessedOutsideJob", _event("aws.s3", "GetObject", params={"bucketName": "amzn-s3-demo-data"},
                                      arn=f"arn:aws:sts::{ACCOUNT}:assumed-role/GeneformerSageMakerExecutionX/s"), True),
    # Metadata reads and listings by anyone but the job (CloudTrail records ListObjectsV2 as ListObjects).
    ("DataAccessedOutsideJob", _event("aws.s3", "HeadObject", params={"bucketName": "amzn-s3-demo-data"}), True),
    ("DataAccessedOutsideJob", _event("aws.s3", "ListObjects", params={"bucketName": "amzn-s3-demo-data"}), True),
    ("DataAccessedOutsideJob", _event("aws.s3", "HeadObject", params={"bucketName": "amzn-s3-demo-data"}, arn=JOB), False),
    ("DataAccessedOutsideJob", _event("aws.s3", "ListObjects", params={"bucketName": "amzn-s3-demo-data"}, arn=JOB), False),
    ("DataAccessedOutsideJob", _event("aws.s3", "ListObjects", params={"bucketName": "amzn-s3-demo-weights"}), False),
    # Listing versions also shows deleted and overwritten objects (the trail's logs show this name).
    ("DataAccessedOutsideJob", _event("aws.s3", "ListObjectVersions", params={"bucketName": "amzn-s3-demo-data"}), True),
    ("DataAccessedOutsideJob", _event("aws.s3", "ListObjectVersions", params={"bucketName": "amzn-s3-demo-data"}, arn=JOB), False),
    # Callers CloudTrail records without an ARN: an AWS service, another account, anonymous.
    ("DataAccessedOutsideJob", _event("aws.s3", "GetObject", params={"bucketName": "amzn-s3-demo-data"},
                                      identity={"type": "AWSService", "invokedBy": "s3.amazonaws.com"}), True),
    ("DataAccessedOutsideJob", _event("aws.s3", "GetObject", params={"bucketName": "amzn-s3-demo-data"},
                                      identity={"type": "AWSAccount", "accountId": "111122223333",
                                                "principalId": "AIDAEXAMPLE"}), True),
    ("DataAccessedOutsideJob", _event("aws.s3", "GetObject", params={"bucketName": "amzn-s3-demo-data"},
                                      identity={"type": "AWSAccount", "accountId": "ANONYMOUS_PRINCIPAL"}), True),
    ("DataAccessedOutsideJob", _event("aws.s3", "GetObject", params={"bucketName": "amzn-s3-demo-weights"},
                                      identity={"type": "AWSService", "invokedBy": "s3.amazonaws.com"}), False),
    ("WeightsChanged", _event("aws.s3", "PutObject", params={"bucketName": "amzn-s3-demo-weights"}), True),
    ("WeightsChanged", _event("aws.s3", "CompleteMultipartUpload", params={"bucketName": "amzn-s3-demo-weights"}), True),
    ("WeightsChanged", _event("aws.s3", "DeleteObjects", params={"bucketName": "amzn-s3-demo-weights"}), True),
    # Even the job's role: it only reads weights, so a write from it is unexpected too.
    ("WeightsChanged", _event("aws.s3", "PutObject", params={"bucketName": "amzn-s3-demo-weights"}, arn=JOB), True),
    ("WeightsChanged", _event("aws.s3", "GetObject", params={"bucketName": "amzn-s3-demo-weights"}, arn=JOB), False),
    ("WeightsChanged", _event("aws.s3", "GetObject", params={"bucketName": "amzn-s3-demo-weights"}), False),
    ("WeightsChanged", _event("aws.s3", "PutObject", params={"bucketName": "amzn-s3-demo-data"}), False),
    ("AuditTrailChanged", _event("aws.cloudtrail", "StopLogging", params={"name": "scfm-audit"}), True),
    ("AuditTrailChanged", _event("aws.cloudtrail", "PutEventSelectors"), True),
    ("AuditTrailChanged", _event("aws.cloudtrail", "LookupEvents"), False),
    ("DataKeyChanged", _event("aws.kms", "ScheduleKeyDeletion", resources=[{"ARN": PARAMS["DataKeyArn"]}]), True),
    ("DataKeyChanged", _event("aws.kms", "DisableKeyRotation", resources=[{"ARN": PARAMS["DataKeyArn"]}]), True),
    ("DataKeyChanged", _event("aws.kms", "ScheduleKeyDeletion", resources=[{"ARN": PARAMS["DataKeyArn"][:-1] + "8"}]), False),
    ("DataKeyChanged", _event("aws.kms", "Decrypt", resources=[{"ARN": PARAMS["DataKeyArn"]}]), False),
    ("DataKeyChanged", _event("aws.kms", "PutKeyPolicy", resources=[{"ARN": PARAMS["WeightsKeyArn"]}]), True),
    ("RolePermissionsChanged", _event("aws.iam", "AttachRolePolicy", params={
        "roleName": "GeneformerSageMakerExecution", "policyArn": "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"}), True),
    ("RolePermissionsChanged", _event("aws.iam", "UpdateAssumeRolePolicy", params={"roleName": "geneformer-codebuild-role"}), True),
    ("RolePermissionsChanged", _event("aws.iam", "AttachRolePolicy", params={"roleName": "unrelated-role"}), False),
    ("ImageRepositoryChanged", _event("aws.ecr", "PutImageTagMutability", params={"repositoryName": "geneformer-ft"}), True),
    ("ImageRepositoryChanged", _event("aws.ecr", "BatchDeleteImage", params={"repositoryName": "scgpt-ft"}), True),
    ("ImageRepositoryChanged", _event("aws.ecr", "PutImage", params={"repositoryName": "geneformer-ft"}), False),
    ("BuildProjectChanged", _event("aws.codebuild", "UpdateProject", params={"name": "geneformer-ft-build"}), True),
    # The live UpdateProject event of 2026-11-18 recorded the project ARN, not the name.
    ("BuildProjectChanged", _event("aws.codebuild", "UpdateProject", params={
        "name": f"arn:aws:codebuild:{REGION}:{ACCOUNT}:project/geneformer-ft-build", "timeoutInMinutes": 60}), True),
    ("BuildProjectChanged", _event("aws.codebuild", "DeleteProject", params={
        "name": f"arn:aws:codebuild:{REGION}:{ACCOUNT}:project/geneformer-ft-build"}), True),
    ("BuildProjectChanged", _event("aws.codebuild", "UpdateProject", params={
        "name": f"arn:aws:codebuild:{REGION}:{ACCOUNT}:project/some-other-project"}), False),
    ("BuildProjectChanged", _event("aws.codebuild", "StartBuild", params={"projectName": "geneformer-ft-build"}), False),
]


@pytest.mark.parametrize("rule, event, alert", CASES)
def test_rule_alerts_on_exactly_the_intended_events(rule, event, alert):
    assert matches(_rules()[rule], event) is alert


def test_every_rule_has_cases_both_ways():
    names = set(_rules())
    assert {c[0] for c in CASES if c[2]} == names
    assert {c[0] for c in CASES if not c[2]} == names


def test_no_rule_alerts_on_an_unrelated_event():
    unrelated = _event("aws.sagemaker", "CreateTrainingJob", params={"trainingJobName": "x"})
    assert not any(matches(p, unrelated) for p in _rules().values())


def test_every_rule_targets_the_topic_and_the_topic_only_accepts_these_rules():
    t = _template()
    res = t["Resources"]
    for name, r in res.items():
        if r["Type"] == "AWS::Events::Rule":
            assert r["Properties"]["Name"].startswith("scfm-alert-"), name
            assert r["Properties"]["Targets"] == [{"Id": "sns", "Arn": {"Ref": "AlertTopic"}}], name
    (stmt,) = res["AlertTopicPolicy"]["Properties"]["PolicyDocument"]["Statement"]
    assert stmt["Principal"] == {"Service": "events.amazonaws.com"} and stmt["Action"] == "sns:Publish"
    assert stmt["Condition"]["ArnLike"]["aws:SourceArn"]["Fn::Sub"].endswith(":rule/scfm-alert-*")


def test_template_needs_no_custom_yaml_tags():
    assert "!Ref" not in TEMPLATE.read_text() and "!Sub" not in TEMPLATE.read_text()


def test_weights_rule_is_created_only_when_a_weights_bucket_is_given():
    t = _template()
    assert t["Parameters"]["WeightsBucket"]["Default"] == ""
    assert t["Resources"]["WeightsChanged"]["Condition"] == "HasWeightsBucket"
    assert t["Conditions"]["HasWeightsBucket"] == {"Fn::Not": [{"Fn::Equals": [{"Ref": "WeightsBucket"}, ""]}]}
    # Every other rule stays unconditional, so an existing stack keeps them on update.
    others = [n for n, r in t["Resources"].items() if r["Type"] == "AWS::Events::Rule" and n != "WeightsChanged"]
    assert others and not any("Condition" in t["Resources"][n] for n in others)


def test_key_rule_watches_nothing_extra_when_there_is_no_weights_key():
    # WeightsKeyArn defaults to "", which no KMS key ARN can equal.
    t = _template()
    assert t["Parameters"]["WeightsKeyArn"]["Default"] == ""
    arns = t["Resources"]["DataKeyChanged"]["Properties"]["EventPattern"]["detail"]["resources"]["ARN"]
    assert arns == [{"Ref": "DataKeyArn"}, {"Ref": "WeightsKeyArn"}]
