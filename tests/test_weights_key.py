# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The weights bucket key (T005, T007): rotated, kept if the stack goes, and used by the training
job only to decrypt, only through S3."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "security" / "weights-key.yaml"
GRANT = ROOT / "docs" / "iam" / "sagemaker-execution-weights-kms-policy.json"


def _resources() -> dict:
    return yaml.safe_load(TEMPLATE.read_text())["Resources"]


def test_key_is_symmetric_rotated_and_retained():
    key = _resources()["WeightsKey"]
    assert key["Type"] == "AWS::KMS::Key"
    assert key["DeletionPolicy"] == "Retain" and key["UpdateReplacePolicy"] == "Retain"
    props = key["Properties"]
    assert props["EnableKeyRotation"] is True
    assert props["KeySpec"] == "SYMMETRIC_DEFAULT" and props["KeyUsage"] == "ENCRYPT_DECRYPT"


def test_key_policy_only_delegates_to_this_account():
    (st,) = _resources()["WeightsKey"]["Properties"]["KeyPolicy"]["Statement"]
    assert st["Effect"] == "Allow"
    assert st["Principal"] == {"AWS": {"Fn::Sub": "arn:${AWS::Partition}:iam::${AWS::AccountId}:root"}}


def test_alias_points_at_the_key_and_the_arn_is_an_output():
    t = yaml.safe_load(TEMPLATE.read_text())
    assert t["Resources"]["WeightsKeyAlias"]["Properties"]["TargetKeyId"] == {"Ref": "WeightsKey"}
    assert t["Outputs"]["WeightsKeyArn"]["Value"] == {"Fn::GetAtt": ["WeightsKey", "Arn"]}


def test_job_grant_is_decrypt_only_on_one_key_via_s3():
    (st,) = json.loads(GRANT.read_text())["Statement"]
    assert sorted(st["Action"]) == ["kms:Decrypt", "kms:DescribeKey"]
    assert st["Resource"].startswith("arn:aws:kms:") and "*" not in st["Resource"]
    assert st["Condition"] == {"StringEquals": {"kms:ViaService": "s3.REGION.amazonaws.com"}}
