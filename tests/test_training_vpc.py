# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The --regulated-data training subnets have no way out except S3, and S3 only for the project buckets,
whether the stack creates its own VPC or adds the subnets to an existing one."""

from __future__ import annotations

from pathlib import Path

import yaml

TEMPLATE = Path(__file__).resolve().parents[1] / "docs" / "security" / "training-vpc.yaml"
VPC_REF = {"Fn::If": ["NewVpc", {"Ref": "Vpc"}, {"Ref": "ExistingVpcId"}]}


def _template() -> dict:
    return yaml.safe_load(TEMPLATE.read_text())


def _resources() -> dict:
    return _template()["Resources"]


def test_there_is_no_route_to_the_internet():
    types = {r["Type"] for r in _resources().values()}
    for forbidden in ("AWS::EC2::InternetGateway", "AWS::EC2::NatGateway", "AWS::EC2::VPCGatewayAttachment",
                      "AWS::EC2::Route", "AWS::EC2::EgressOnlyInternetGateway", "AWS::EC2::TransitGatewayAttachment",
                      "AWS::EC2::VPCPeeringConnection"):
        assert forbidden not in types, forbidden
    for name, r in _resources().items():
        if r["Type"] == "AWS::EC2::Subnet":
            assert r["Properties"]["MapPublicIpOnLaunch"] is False, name


def test_subnets_use_their_own_route_table_in_either_mode():
    res = _resources()
    assoc = {r["Properties"]["SubnetId"]["Ref"]: r["Properties"]["RouteTableId"] for r in res.values()
             if r["Type"] == "AWS::EC2::SubnetRouteTableAssociation"}
    assert assoc == {"SubnetA": {"Ref": "RouteTable"}, "SubnetB": {"Ref": "RouteTable"}}
    for name in ("SubnetA", "SubnetB", "RouteTable", "S3Endpoint", "SecurityGroup"):
        assert res[name]["Properties"]["VpcId"] == VPC_REF, name


def test_the_new_vpc_is_conditional_and_existing_mode_requires_cidrs():
    t = _template()
    assert t["Resources"]["Vpc"]["Condition"] == "NewVpc"
    assert t["Conditions"]["NewVpc"] == {"Fn::Equals": [{"Ref": "ExistingVpcId"}, ""]}
    asserts = t["Rules"]["ExistingVpcNeedsSubnetCidrs"]["Assertions"]
    assert {a["Assert"]["Fn::Not"][0]["Fn::Equals"][0]["Ref"] for a in asserts} == {"SubnetACidr", "SubnetBCidr"}


def test_the_only_endpoint_is_s3_and_its_policy_names_only_the_project_buckets():
    endpoints = [r for r in _resources().values() if r["Type"] == "AWS::EC2::VPCEndpoint"]
    assert len(endpoints) == 1
    props = endpoints[0]["Properties"]
    assert props["VpcEndpointType"] == "Gateway"
    assert props["ServiceName"] == {"Fn::Sub": "com.amazonaws.${AWS::Region}.s3"}
    (stmt,) = props["PolicyDocument"]["Statement"]
    assert stmt["Effect"] == "Allow"
    assert sorted(stmt["Action"]) == ["s3:AbortMultipartUpload", "s3:GetObject", "s3:ListBucket", "s3:PutObject"]
    resources = [r["Fn::Sub"] for r in stmt["Resource"]]
    assert all(r.startswith(("arn:aws:s3:::${DataBucket}", "arn:aws:s3:::${WeightsBucket}")) for r in resources)


def test_security_group_allows_https_to_s3_only_and_no_inbound_from_outside_the_job():
    res = _resources()
    sg = res["SecurityGroup"]["Properties"]
    assert "SecurityGroupIngress" not in sg
    (https,) = sg["SecurityGroupEgress"]
    assert (https["FromPort"], https["ToPort"], https["IpProtocol"]) == (443, 443, "tcp")
    assert https["DestinationPrefixListId"] == {"Ref": "S3PrefixListId"}
    assert "CidrIp" not in https
    ingress = [r["Properties"] for r in res.values() if r["Type"] == "AWS::EC2::SecurityGroupIngress"]
    assert ingress and all(i["SourceSecurityGroupId"] == {"Ref": "SecurityGroup"} for i in ingress)


def test_new_vpc_uses_default_tenancy_which_sagemaker_training_requires():
    assert _resources()["Vpc"]["Properties"]["InstanceTenancy"] == "default"
