# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Launcher security options: image digest pinning (T005), isolation/VPC (T006), KMS (T007)."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

_LAUNCH = Path(__file__).resolve().parents[1] / "sagemaker" / "launch.py"
_spec = importlib.util.spec_from_file_location("launch_sec", _LAUNCH)
assert _spec and _spec.loader
launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)

REG = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
DIGEST = "sha256:" + "ab" * 32


def _parse(*argv: str):
    old = sys.argv
    sys.argv = ["launch.py", "--image-uri", f"{REG}/geneformer-ft:latest", "--role", "arn:r", *argv]
    try:
        return launch._parse()
    finally:
        sys.argv = old


class _Ecr:
    def __init__(self, digest=DIGEST, error: Exception | None = None):
        self.calls: list[dict] = []
        self._digest, self._error = digest, error

    def describe_images(self, **kw):
        self.calls.append(kw)
        if self._error:
            raise self._error
        return {"imageDetails": [{"imageDigest": self._digest}]}


# --- T005: tags are resolved to digests ------------------------------------------------
def test_a_tag_is_resolved_to_its_digest(capsys):
    ecr = _Ecr()
    out = launch.resolve_image_uri(f"{REG}/geneformer-ft:v3", ecr_client=ecr)
    assert out == f"{REG}/geneformer-ft@{DIGEST}"
    assert ecr.calls == [
        {"registryId": "123456789012", "repositoryName": "geneformer-ft", "imageIds": [{"imageTag": "v3"}]}
    ]
    assert "resolved to" in capsys.readouterr().out  # printed, for the audit trail


def test_an_untagged_uri_means_latest():
    ecr = _Ecr()
    launch.resolve_image_uri(f"{REG}/scgpt-ft", ecr_client=ecr)
    assert ecr.calls[0]["imageIds"] == [{"imageTag": "latest"}]


def test_a_digest_uri_is_used_unchanged_without_calling_ecr():
    ecr = _Ecr()
    uri = f"{REG}/geneformer-ft@{DIGEST}"
    assert launch.resolve_image_uri(uri, ecr_client=ecr) == uri
    assert ecr.calls == []


def test_a_failed_lookup_stops_the_launch_with_advice():
    with pytest.raises(SystemExit, match=r"(?s)Could not resolve geneformer-ft:latest.*--allow-mutable-image"):
        launch.resolve_image_uri(f"{REG}/geneformer-ft:latest", ecr_client=_Ecr(error=PermissionError("denied")))


def test_non_ecr_images_are_refused_unless_explicitly_allowed():
    with pytest.raises(SystemExit, match="not an ECR image URI"):
        launch.resolve_image_uri("docker.io/library/python:3.11")
    assert launch.resolve_image_uri("docker.io/library/python:3.11", allow_mutable=True).startswith("docker.io")


def test_allow_mutable_skips_the_lookup():
    ecr = _Ecr()
    uri = f"{REG}/geneformer-ft:latest"
    assert launch.resolve_image_uri(uri, allow_mutable=True, ecr_client=ecr) == uri
    assert ecr.calls == []


# --- T006: network isolation and VPC ---------------------------------------------------
def test_defaults_add_no_security_options():
    kwargs, env = launch.security_options(_parse("--allow-network"))
    assert (kwargs, env) == ({}, {})


def test_network_isolation_sets_the_flag_and_makes_libraries_fail_fast_offline():
    kwargs, env = launch.security_options(_parse("--network-isolation", "--train-s3", "s3://amzn-s3-demo-bucket/d/"))
    assert kwargs == {"enable_network_isolation": True}
    assert env == {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_network_isolation_is_refused_with_the_self_fetching_dataset():
    with pytest.raises(SystemExit, match="cannot be combined with --data-source pbmc3k"):
        launch.security_options(_parse("--network-isolation", "--data-source", "pbmc3k"))


def test_vpc_needs_subnets_and_security_groups_together():
    with pytest.raises(SystemExit, match="must be given together"):
        launch.security_options(_parse("--subnets", "subnet-1"))
    with pytest.raises(SystemExit, match="must be given together"):
        launch.security_options(_parse("--security-group-ids", "sg-1"))
    kwargs, _ = launch.security_options(
        _parse("--subnets", "subnet-1", "subnet-2", "--security-group-ids", "sg-1", "--encrypt-inter-container-traffic",
               "--allow-network")
    )
    assert kwargs == {
        "subnets": ["subnet-1", "subnet-2"],
        "security_group_ids": ["sg-1"],
        "encrypt_inter_container_traffic": True,
    }


# --- T007: KMS ------------------------------------------------------------------------
def test_kms_keys_are_passed_through():
    kwargs, _ = launch.security_options(
        _parse("--volume-kms-key", "key-v", "--output-kms-key", "arn:aws:kms:x", "--allow-network",
               "--instance-type", "ml.p3.2xlarge")
    )
    assert kwargs == {"volume_kms_key": "key-v", "output_kms_key": "arn:aws:kms:x"}


# --- input safety hyperparameters (T003, T004) -------------------------------------------
def test_input_safety_settings_reach_the_entry_point():
    hp = launch.build_hyperparameters(
        _parse("--max-cells", "5000", "--max-input-gib", "0.5"),
        scgpt_model_root="/scgpt",
        quantize=False,
    )
    assert (hp["require-input-hashes"], hp["max-cells"], hp["max-input-gib"]) == ("true", "5000", "0.5")
    off = launch.build_hyperparameters(_parse("--allow-unverified-inputs"), scgpt_model_root="/scgpt", quantize=False)
    assert (off["require-input-hashes"], off["max-cells"], off["max-input-gib"]) == ("false", "1000000", "4.0")


def test_every_sm_entry_flag_the_launcher_sends_is_one_the_entry_point_accepts():
    entry_spec = importlib.util.spec_from_file_location(
        "sm_entry_sec", Path(__file__).resolve().parents[1] / "sagemaker" / "sm_entry.py"
    )
    entry = importlib.util.module_from_spec(entry_spec)
    entry_spec.loader.exec_module(entry)
    hp = launch.build_hyperparameters(_parse(), scgpt_model_root="/scgpt", quantize=False)
    argv = ["sm_entry.py"] + [x for k, v in hp.items() for x in (f"--{k}", str(v))]
    old = sys.argv
    sys.argv = argv
    try:
        ns = vars(entry._parse_args())
    finally:
        sys.argv = old
    for key in hp:
        assert key.replace("-", "_") in ns, f"launcher sends --{key} but sm_entry does not define it"


def test_the_sagemaker_estimator_accepts_every_option_we_pass():
    # `import sagemaker` alone would find this repo's own sagemaker/ scripts directory.
    pytest.importorskip("sagemaker.estimator")
    from sagemaker.estimator import EstimatorBase

    params = set(inspect.signature(EstimatorBase.__init__).parameters)
    kwargs, _ = launch.security_options(
        _parse(
            "--network-isolation", "--subnets", "s-1", "--security-group-ids", "sg-1",
            "--encrypt-inter-container-traffic", "--volume-kms-key", "k", "--output-kms-key", "k",
            "--instance-type", "ml.p3.2xlarge",
        )
    )
    assert set(kwargs) <= params, f"unknown Estimator kwargs: {set(kwargs) - params}"


# --- T016/T003/T006: manifest pins and --regulated-data ------------------------------------
H = "a" * 64
FULL = ["--regulated-data", "--instance-type", "ml.p3.2xlarge", "--train-s3", "s3://b/p/",
        "--volume-kms-key", "k1", "--output-kms-key", "k2",
        "--subnets", "subnet-1", "--security-group-ids", "sg-1", "--manifest-sha256", f"train={H}",
        "--group-column", "donor_id"]


def test_manifest_pins_are_validated_and_forwarded():
    args = _parse("--manifest-sha256", f"train={H}", f"weights={'B' * 64}")
    assert launch.manifest_pins(args) == {"train": H, "weights": "b" * 64}
    hp = launch.build_hyperparameters(args, scgpt_model_root="/scgpt", quantize=False)
    assert hp["manifest-sha256"] == f"train={H},weights={'b' * 64}"
    for bad in (["train=abc"], [f"cache={H}"], [f"train={H}", f"train={H}"], [H]):
        with pytest.raises(SystemExit):
            launch.manifest_pins(_parse("--manifest-sha256", *bad))


def test_regulated_data_turns_the_controls_on():
    args = _parse(*FULL)
    launch.apply_regulated_data(args)
    assert args.network_isolation and args.require_input_hashes and args.encrypt_inter_container_traffic
    kwargs, env = launch.security_options(args)
    assert kwargs["enable_network_isolation"] is True
    assert kwargs["volume_kms_key"] == "k1" and kwargs["subnets"] == ["subnet-1"]
    assert env["HF_HUB_OFFLINE"] == "1"


def test_regulated_data_lists_everything_that_is_missing():
    args = _parse("--regulated-data", "--instance-type", "ml.p3.2xlarge")
    with pytest.raises(SystemExit) as e:
        launch.apply_regulated_data(args)
    msg = str(e.value)
    for needed in ("--train-s3", "--volume-kms-key", "--output-kms-key", "--subnets", "--manifest-sha256 train"):
        assert needed in msg


def test_regulated_data_refuses_self_fetch_mutable_image_and_unpinned_weights():
    args = _parse(*FULL, "--data-source", "pbmc3k", "--allow-mutable-image", "--scgpt-weights-s3", "s3://w/x/")
    with pytest.raises(SystemExit) as e:
        launch.apply_regulated_data(args)
    msg = str(e.value)
    assert "--data-source channel" in msg and "--allow-mutable-image" in msg and "weights=<hash>" in msg


def test_regulated_data_needs_a_group_column_or_an_explicit_cell_level_split():
    no_group = [a for a in FULL if a not in ("--group-column", "donor_id")]
    with pytest.raises(SystemExit) as e:
        launch.apply_regulated_data(_parse(*no_group))
    assert "--group-column" in str(e.value) and "--allow-cell-level-split" in str(e.value)
    args = _parse(*no_group, "--allow-cell-level-split")
    launch.apply_regulated_data(args)  # allowed, but warned
    (warning,) = launch.cell_level_split_warnings(args)
    assert "both training and held-out" in warning
    assert launch.cell_level_split_warnings(_parse(*FULL)) == []
    assert launch.cell_level_split_warnings(_parse()) == []  # not regulated: no warning


def test_group_column_is_forwarded_only_when_set():
    hp = launch.build_hyperparameters(_parse(*FULL), scgpt_model_root="/scgpt", quantize=False)
    assert hp["group-column"] == "donor_id"
    hp = launch.build_hyperparameters(_parse(), scgpt_model_root="/scgpt", quantize=False)
    assert "group-column" not in hp  # SageMaker hyperparameters must not be empty strings


def test_without_the_flag_nothing_changes():
    args = _parse()
    launch.apply_regulated_data(args)
    assert args.network_isolation and args.require_input_hashes  # both are on by default


def test_hyperparameters_omit_the_pin_when_unset():
    hp = launch.build_hyperparameters(_parse(), scgpt_model_root="/scgpt", quantize=False)
    assert "manifest-sha256" not in hp


# --- T018/T019: the launcher uploads no job code -----------------------------------------------
def _estimator_keywords() -> set[str]:
    import ast

    tree = ast.parse(_LAUNCH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Estimator":
            return {kw.arg for kw in node.keywords if kw.arg}
    raise AssertionError("no Estimator(...) call found in launch.py")


def test_estimator_is_built_without_any_code_to_upload():
    # entry_point, source_dir, dependencies and git_config each make the SDK tar a directory
    # and upload it to S3 for the job to fetch. That uploads whatever sits under sagemaker/ and
    # lets anyone who can write the S3 code prefix choose what runs as root (T018, T019).
    kw = _estimator_keywords()
    assert not kw & {"entry_point", "source_dir", "dependencies", "git_config", "code_location"}, kw


def test_the_images_carry_the_entry_script_the_job_runs():
    root = _LAUNCH.parents[1]
    for name in ("Dockerfile", "Dockerfile.scgpt"):
        text = (root / "sagemaker" / name).read_text()
        assert "SAGEMAKER_PROGRAM=sm_entry.py" in text, name
        assert "COPY sagemaker/sm_entry.py /opt/ml/code/sm_entry.py" in text, name


def test_the_sdk_stages_nothing_for_the_launcher_arguments():
    # Skipped where `sagemaker` resolves to this repo's scripts directory instead of the SDK.
    pytest.importorskip("sagemaker.estimator")
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from sagemaker.estimator import Estimator

    session = MagicMock(local_mode=False, config={}, sagemaker_config={}, boto_region_name="us-east-1")
    session.settings = SimpleNamespace(local_download_dir=None, include_jumpstart_tags=False)
    session.default_bucket_prefix = None
    with patch("sagemaker.estimator.tar_and_upload_dir") as upload:
        est = Estimator(
            sagemaker_session=session,
            image_uri=f"{REG}/geneformer-ft@{DIGEST}",
            role="arn:aws:iam::123456789012:role/r",
            instance_count=1,
            instance_type="ml.g5.xlarge",
            hyperparameters={"preset": "V1-10M"},
            output_path="s3://b/scfm-ft/output",
            disable_profiler=True,
            debugger_hook_config=False,
        )
        est._prepare_for_training(job_name="j")
    upload.assert_not_called()
    assert est.uploaded_code is None
    assert {"sagemaker_program", "sagemaker_submit_directory"}.isdisjoint(est.hyperparameters())


# --- T009/T011: --max-run is capped at what the launcher policy allows -----------------------
def test_max_run_default_and_ceiling_equal_the_launcher_policy_limit():
    import json

    policy = json.loads((_LAUNCH.parents[1] / "docs" / "iam" / "sagemaker-launcher-policy.json").read_text())
    deny = next(st for st in policy["Statement"] if st["Sid"] == "DenyJobsLongerThanTwelveHours")
    limit = int(deny["Condition"]["NumericGreaterThan"]["sagemaker:MaxRuntimeInSeconds"])
    assert launch.MAX_RUN_CEILING == limit == 12 * 3600
    # A default launch must not be the one the policy denies.
    assert _parse().max_run == limit


@pytest.mark.parametrize("value, ok", [("3600", True), ("43200", True), ("43201", False), ("86400", False), ("0", False), ("-5", False)])
def test_max_run_above_the_ceiling_or_not_positive_is_refused(value, ok):
    args = _parse("--max-run", value)
    if ok:
        launch.check_max_run(args)
    else:
        with pytest.raises(SystemExit) as e:
            launch.check_max_run(args)
        assert "--max-run" in str(e.value)


def test_allow_long_run_lifts_the_ceiling_but_not_the_positive_check():
    launch.check_max_run(_parse("--max-run", "86400", "--allow-long-run"))
    with pytest.raises(SystemExit):
        launch.check_max_run(_parse("--max-run", "0", "--allow-long-run"))


# --- T003/T016: say so when channel data runs without required hashes ------------------------
def test_channel_data_without_required_hashes_gets_a_warning():
    w = launch.unverified_input_warnings(_parse("--train-s3", "s3://b/p/", "--allow-unverified-inputs"))
    assert len(w) == 1 and "--train-s3" in w[0] and "--allow-unverified-inputs" in w[0]
    both = launch.unverified_input_warnings(
        _parse("--train-s3", "s3://b/p/", "--scgpt-weights-s3", "s3://w/x/", "--allow-unverified-inputs")
    )
    assert "--train-s3 and --scgpt-weights-s3" in both[0]


def test_no_warning_when_hashes_are_required_or_nothing_comes_from_s3():
    assert launch.unverified_input_warnings(_parse("--train-s3", "s3://b/p/")) == []
    assert launch.unverified_input_warnings(_parse("--data-source", "pbmc3k")) == []


# --- T005/T007: --regulated-data checks the buckets against the baseline ---------------------
class _AwsError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


_TLS_POLICY = '{"Statement":[{"Effect":"Deny","Principal":"*","Action":"s3:*","Resource":"*",' \
              '"Condition":{"Bool":{"aws:SecureTransport":"false"}}}]}'


class _FakeS3:
    """Answers like S3; a value that is an exception instance is raised instead."""

    def __init__(self, **overrides):
        self.answers = {
            "get_bucket_versioning": {"Status": "Enabled"},
            "get_public_access_block": {"PublicAccessBlockConfiguration": dict.fromkeys(
                ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets"), True)},
            "get_bucket_encryption": {"ServerSideEncryptionConfiguration": {"Rules": [
                {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "aws:kms"}}]}},
            "get_bucket_policy": {"Policy": _TLS_POLICY},
        }
        self.answers.update(overrides)
        self.calls: list[str] = []

    def __getattr__(self, name):
        def call(Bucket):  # noqa: N803 - boto3 keyword
            self.calls.append(f"{name}:{Bucket}")
            ans = self.answers[name]
            if isinstance(ans, Exception):
                raise ans
            return ans

        return call


def test_a_bucket_that_meets_the_baseline_has_no_problems():
    assert launch.bucket_baseline_problems(_FakeS3(), "data") == []


@pytest.mark.parametrize("override, expect", [
    ({"get_bucket_versioning": {"Status": "Suspended"}}, "versioning is not enabled"),
    ({"get_bucket_versioning": {}}, "versioning is not enabled"),
    ({"get_public_access_block": {"PublicAccessBlockConfiguration": {"BlockPublicAcls": True}}}, "Block Public Access is off for IgnorePublicAcls"),
    ({"get_public_access_block": _AwsError("NoSuchPublicAccessBlockConfiguration")}, "Block Public Access is not configured"),
    ({"get_bucket_encryption": {"ServerSideEncryptionConfiguration": {"Rules": [
        {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}}}, "not SSE-KMS"),
    ({"get_bucket_encryption": _AwsError("ServerSideEncryptionConfigurationNotFoundError")}, "no default encryption"),
    ({"get_bucket_policy": _AwsError("NoSuchBucketPolicy")}, "nothing denies requests made without TLS"),
    ({"get_bucket_policy": {"Policy": '{"Statement":[{"Effect":"Allow","Action":"s3:GetObject","Resource":"*"}]}'}}, "does not deny requests made without TLS"),
])
def test_each_baseline_gap_is_reported(override, expect):
    problems = launch.bucket_baseline_problems(_FakeS3(**override), "data")
    assert len(problems) == 1 and expect in problems[0] and problems[0].startswith("data:")


def test_a_check_that_cannot_run_fails_closed():
    problems = launch.bucket_baseline_problems(_FakeS3(get_bucket_versioning=_AwsError("AccessDenied")), "data")
    assert problems == ["data: cannot check versioning (AccessDenied)"]


def test_preflight_is_skipped_unless_regulated_data_and_not_overridden():
    s3 = _FakeS3(get_bucket_versioning={"Status": "Suspended"})
    launch.preflight_buckets(_parse("--train-s3", "s3://data/p/"), "s3://out/o", s3=s3)  # not regulated
    assert s3.calls == []
    regulated = [*FULL, "--skip-bucket-check"]
    launch.preflight_buckets(_parse(*regulated), "s3://out/o", s3=s3)
    assert s3.calls == []


def test_preflight_checks_every_bucket_once_and_lists_all_problems():
    s3 = _FakeS3(get_bucket_versioning={"Status": "Suspended"})
    with pytest.raises(SystemExit) as e:
        launch.preflight_buckets(_parse(*FULL, "--scgpt-weights-s3", "s3://w/x/"), "s3://b/scfm-ft/output", s3=s3)
    msg = str(e.value)
    # FULL's --train-s3 is s3://b/p/ and the output is also in b, so b is checked once.
    assert s3.calls.count("get_bucket_versioning:b") == 1 and "get_bucket_versioning:w" in s3.calls
    assert "b: versioning is not enabled" in msg and "w: versioning is not enabled" in msg
    assert "--skip-bucket-check" in msg


def test_preflight_passes_when_every_bucket_meets_the_baseline():
    launch.preflight_buckets(_parse(*FULL), "s3://b/scfm-ft/output", s3=_FakeS3())


# --- hashes are required by default (T003, T016) ---------------------------------------------
def test_input_hashes_are_required_unless_opted_out():
    assert _parse("--train-s3", "s3://b/p/").require_input_hashes is True
    assert _parse("--train-s3", "s3://b/p/", "--require-input-hashes").require_input_hashes is True
    assert _parse("--allow-unverified-inputs").require_input_hashes is False


def test_opt_out_and_the_old_flag_together_are_refused():
    with pytest.raises(SystemExit):
        _parse("--allow-unverified-inputs", "--require-input-hashes")


def test_regulated_data_refuses_the_opt_out():
    args = _parse(*FULL, "--allow-unverified-inputs")
    with pytest.raises(SystemExit) as e:
        launch.apply_regulated_data(args)
    assert "--allow-unverified-inputs" in str(e.value)


# --- network isolation is the default for S3 data (T006) -------------------------------------
def test_isolation_is_on_by_default_for_s3_data():
    args = _parse("--train-s3", "s3://b/p/")
    assert args.network_isolation is True
    kwargs, env = launch.security_options(args)
    assert kwargs["enable_network_isolation"] is True and env == launch.OFFLINE_ENV


def test_allow_network_turns_isolation_off():
    args = _parse("--train-s3", "s3://b/p/", "--allow-network")
    assert args.network_isolation is False
    kwargs, env = launch.security_options(args)
    assert "enable_network_isolation" not in kwargs and env == {}


def test_the_self_fetching_dataset_runs_with_network_and_explicit_isolation_is_still_refused():
    args = _parse("--data-source", "pbmc3k")
    assert args.network_isolation is False
    launch.security_options(args)  # no error
    with pytest.raises(SystemExit):
        launch.security_options(_parse("--data-source", "pbmc3k", "--network-isolation"))


def test_allow_network_and_network_isolation_together_are_refused():
    with pytest.raises(SystemExit):
        _parse("--allow-network", "--network-isolation")


def test_regulated_data_refuses_allow_network():
    args = _parse(*FULL, "--allow-network")
    with pytest.raises(SystemExit) as e:
        launch.apply_regulated_data(args)
    assert "--allow-network" in str(e.value)


# --- the job name must satisfy SageMaker even when the image is a digest URI ----------------------
def test_default_job_name_is_valid_for_sagemaker_and_does_not_use_the_image():
    from datetime import datetime, timezone

    args = _parse("--preset", "V1-10M")
    name = launch.default_job_name(args, datetime(2026, 11, 17, 22, 42, 2, tzinfo=timezone.utc))
    assert name == "geneformer-V1-10M-20261117-224202"
    launch.check_job_name(name)


def test_job_names_with_bad_characters_are_refused():
    for bad in ("geneformer-ft@sha256-2026", "has space", "-leading", "a" * 64, ""):
        with pytest.raises(SystemExit):
            launch.check_job_name(bad)
    launch.check_job_name("geneformer-v1-10m-regulated-1")


# --- local-NVMe instances encrypt storage in hardware and refuse a volume key ---------------------
def test_local_storage_instance_families():
    for t in ("ml.g5.xlarge", "ml.g4dn.xlarge", "ml.g6e.2xlarge", "ml.p4d.24xlarge", "ml.p5.48xlarge", "ml.m5d.xlarge"):
        assert launch.has_local_storage(t), t
    for t in ("ml.p3.2xlarge", "ml.m5.xlarge", "ml.c5.xlarge", "ml.p2.xlarge"):
        assert not launch.has_local_storage(t), t


def test_a_volume_key_is_refused_on_a_local_storage_instance():
    with pytest.raises(SystemExit) as e:
        launch.security_options(_parse("--volume-kms-key", "k", "--allow-network"))  # default is ml.g5.xlarge
    assert "local NVMe" in str(e.value)


def test_regulated_data_does_not_ask_for_a_volume_key_on_a_local_storage_instance():
    args = _parse(*[a for a in FULL if a not in ("--volume-kms-key", "k1", "--instance-type", "ml.p3.2xlarge")])
    launch.apply_regulated_data(args)  # g5 default: no volume key needed
    assert args.network_isolation
    ebs = _parse(*[a for a in FULL if a not in ("--volume-kms-key", "k1")])
    with pytest.raises(SystemExit) as e:
        launch.apply_regulated_data(ebs)
    assert "--volume-kms-key" in str(e.value)


# --- T007/T013: model output uses the output bucket's own KMS key ----------------------------
_CMK = "arn:aws:kms:us-east-1:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab"


def _enc(algo, key=None):
    default = {"SSEAlgorithm": algo}
    if key:
        default["KMSMasterKeyID"] = key
    return {"ServerSideEncryptionConfiguration": {"Rules": [{"ApplyServerSideEncryptionByDefault": default}]}}


def test_output_key_defaults_to_the_buckets_customer_managed_key(capsys):
    # Seen on 2026-11-18: with no OutputDataConfig.KmsKeyId SageMaker wrote model.tar.gz under
    # alias/aws/s3, not the data bucket's customer-managed default key.
    args = _parse("--output-s3", "s3://data/output/")
    launch.apply_output_kms_default(args, "s3://data/output/", _FakeS3(get_bucket_encryption=_enc("aws:kms", _CMK)))
    assert args.output_kms_key == _CMK
    assert "default key" in capsys.readouterr().out


def test_an_explicit_output_key_is_never_replaced():
    args = _parse("--output-kms-key", "explicit")
    s3 = _FakeS3(get_bucket_encryption=_enc("aws:kms", _CMK))
    launch.apply_output_kms_default(args, "s3://data/output/", s3)
    assert args.output_kms_key == "explicit" and s3.calls == []


@pytest.mark.parametrize("encryption", [_enc("AES256"), _enc("aws:kms"), _enc("aws:kms", "alias/aws/s3"),
                                        _enc("aws:kms", "arn:aws:kms:us-east-1:123456789012:alias/aws/s3")])
def test_no_key_is_set_for_sse_s3_or_the_aws_managed_key(encryption, capsys):
    args = _parse()
    launch.apply_output_kms_default(args, "s3://data/output/", _FakeS3(get_bucket_encryption=encryption))
    assert args.output_kms_key is None
    assert capsys.readouterr().err == ""


def test_an_unreadable_bucket_leaves_the_key_unset_and_says_so(capsys):
    args = _parse()
    launch.apply_output_kms_default(args, "s3://data/output/", _FakeS3(get_bucket_encryption=_AwsError("AccessDenied")))
    assert args.output_kms_key is None
    err = capsys.readouterr().err
    assert "AccessDenied" in err and "aws/s3" in err and "--output-kms-key" in err


def test_main_passes_the_filled_key_to_the_estimator():
    src = (Path(__file__).resolve().parents[1] / "sagemaker" / "launch.py").read_text()
    body = src[src.index("def main()"):]
    fill = body.index("apply_output_kms_default(args, output_path, s3)")
    assert body.index('security_kwargs["output_kms_key"] = args.output_kms_key', fill) < body.index("Estimator(", fill)


# --- T005/T016: every S3 input channel needs a manifest pin by default -----------------------
def test_s3_channel_without_a_pin_is_refused():
    with pytest.raises(SystemExit) as e:
        launch.check_manifest_pins(_parse("--train-s3", "s3://b/p/"))
    msg = str(e.value)
    assert "train channel" in msg and "--manifest-sha256 train=<hash>" in msg
    assert "--allow-unpinned-manifest" in msg


def test_every_unpinned_channel_is_named():
    args = _parse("--train-s3", "s3://b/p/", "--scgpt-weights-s3", "s3://w/x/", "--manifest-sha256", f"train={H}")
    with pytest.raises(SystemExit) as e:
        launch.check_manifest_pins(args)
    assert "weights channel" in str(e.value) and "train=<hash>" not in str(e.value)


def test_pinned_channels_and_self_fetch_pass():
    launch.check_manifest_pins(_parse("--train-s3", "s3://b/p/", "--manifest-sha256", f"train={H}"))
    launch.check_manifest_pins(_parse("--data-source", "pbmc3k"))
    launch.check_manifest_pins(
        _parse("--train-s3", "s3://b/p/", "--scgpt-weights-s3", "s3://w/x/",
               "--manifest-sha256", f"train={H}", f"weights={H}")
    )


def test_opt_outs_lift_the_pin_requirement_with_a_warning():
    args = _parse("--train-s3", "s3://b/p/", "--allow-unpinned-manifest")
    launch.check_manifest_pins(args)
    w = launch.unverified_input_warnings(args)
    assert len(w) == 1 and "--allow-unpinned-manifest" in w[0] and "train channel" in w[0]
    # No manifest required at all implies no pin; that path has its own warning.
    launch.check_manifest_pins(_parse("--train-s3", "s3://b/p/", "--allow-unverified-inputs"))


def test_regulated_data_refuses_the_unpinned_opt_out():
    with pytest.raises(SystemExit) as e:
        launch.apply_regulated_data(_parse(*FULL, "--allow-unpinned-manifest"))
    assert "--allow-unpinned-manifest" in str(e.value)


def test_main_checks_pins_before_building_the_job():
    src = _LAUNCH.read_text()
    body = src[src.index("def main()"):]
    assert body.index("check_manifest_pins(args)") < body.index("build_hyperparameters(")


def test_wrapper_forwards_the_manifest_pin():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "do-fine-tuning-job.sh").read_text()
    assert '--manifest-sha256)     MANIFEST_SHA256="$2"' in text
    assert 'ARGS+=( --manifest-sha256 "train=${MANIFEST_SHA256}" )' in text
