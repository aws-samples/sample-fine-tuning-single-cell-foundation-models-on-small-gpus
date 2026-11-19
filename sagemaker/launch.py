# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Launch an scfm-ft fine-tuning job on SageMaker.

Submits ``sm_entry.py`` as a SageMaker training job on a GPU instance, using a
prebuilt training image (see ``Dockerfile``). The image bakes in Geneformer + its
weights and the ``scfm_ft`` package, so the job does not clone or download at
runtime.

Usage:
  python sagemaker/launch.py \
      --image-uri <acct>.dkr.ecr.<region>.amazonaws.com/geneformer-ft:b<N> \
      --role arn:aws:iam::<acct>:role/<SageMakerExecutionRole> \
      --train-s3 s3://<bucket>/pbmc3k/ --manifest-sha256 train=<sha256 of its SHA256SUMS> \
      --instance-type ml.g5.xlarge \
      --preset V2-104M --epochs 10 --batch-size 24

The ``--train-s3`` prefix must contain either a raw ``.loom``/``.h5ad`` or a
pre-tokenized ``.dataset`` directory (tokenizing once on CPU and shipping the
``.dataset`` keeps GPU time spent only on fine-tuning).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

# Every job's model output lives under this prefix of the SageMaker default bucket, so the
# execution role's S3 grants can be scoped to it (docs/iam/sagemaker-execution-policy.json)
# rather than to the whole bucket. No job code is uploaded (see main()), so there is no
# code prefix.
S3_PREFIX = "scfm-ft"

# docs/iam/sagemaker-launcher-policy.json denies CreateTrainingJob when MaxRuntimeInSeconds is
# above this, so a job asking for more is refused at the API. Keeping the launcher's default
# and ceiling equal to it means a default launch is never the one that gets denied.
MAX_RUN_CEILING = 12 * 3600


def resolve_s3_paths(bucket: str, prefix: str, output_s3: str | None) -> str:
    """Return the estimator's ``output_path``.

    The SDK appends ``/<job-name>/output/model.tar.gz``, giving
    ``s3://<bucket>/<prefix>/output/<job>/output/model.tar.gz``.
    """
    prefix = prefix.strip("/")
    return output_s3 or f"s3://{bucket}/{prefix}/output"


# Sensible instance <-> model guidance (not enforced; override as you like):
#   ml.g4dn.xlarge  (T4  16GB) -> V1-10M
#   ml.g5.xlarge    (A10G 24GB) -> V2-104M
#   ml.p4d.24xlarge (A100 80GB) -> V2-316M or large batches
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch scfm-ft on SageMaker.")
    p.add_argument("--image-uri", required=True, help="ECR URI of the training image.")
    p.add_argument("--role", required=True, help="SageMaker execution role ARN.")
    p.add_argument(
        "--train-s3",
        default=None,
        help="S3 prefix with the train data (required unless --data-source pbmc3k).",
    )
    p.add_argument("--instance-type", default="ml.g5.xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--job-name", default=None)
    p.add_argument(
        "--output-s3",
        default=None,
        help="S3 prefix for model output (default: s3://<default bucket>/<s3-prefix>/output). "
        "A path outside that prefix needs a matching s3:PutObject grant on the role.",
    )
    p.add_argument(
        "--s3-prefix",
        default=S3_PREFIX,
        help="Key prefix in the SageMaker default bucket for model output. The scoped "
        "execution policy (docs/iam) grants access to this prefix only.",
    )
    p.add_argument(
        "--max-run",
        type=int,
        default=MAX_RUN_CEILING,
        help=f"Max runtime in seconds (default and ceiling {MAX_RUN_CEILING}, the limit in "
        "docs/iam/sagemaker-launcher-policy.json). A runaway job is billed until this.",
    )
    p.add_argument(
        "--skip-bucket-check",
        action="store_true",
        help="With --regulated-data, launch even if the S3 buckets cannot be checked or fail the "
        "baseline (versioning, Block Public Access, SSE-KMS, TLS-only). Weakens --regulated-data.",
    )
    p.add_argument(
        "--allow-long-run",
        action="store_true",
        help="Allow --max-run above the ceiling. The launcher policy denies such a job if you "
        "have applied it.",
    )
    p.add_argument("--use-spot", action="store_true", help="Use managed spot instances.")
    # Hyperparameters forwarded to sm_entry.py:
    p.add_argument("--name", default="sm_run")
    p.add_argument(
        "--backend",
        default="geneformer",
        choices=["geneformer", "scgpt"],
        help="Model backend. Pair with the matching --image-uri (geneformer-ft or scgpt-ft).",
    )
    p.add_argument("--preset", default="V2-104M")
    p.add_argument("--label-column", default="cell_type")
    p.add_argument(
        "--group-column",
        default=None,
        help="obs column naming each cell's donor (or other group). Splits and CV folds then "
        "keep every group on one side, so no donor has cells in both training and held-out "
        "data. --regulated-data requires it unless --allow-cell-level-split.",
    )
    p.add_argument("--classifier-kind", default="cell", choices=["cell", "gene"])
    p.add_argument(
        "--privacy-audit",
        action="store_true",
        help="After training, run a membership-inference audit: compare the model's loss on "
        "training cells and held-out cells, and write <name>_privacy_audit.json. Needs "
        "--num-crossval-splits 1 or 5 (Geneformer: 1, with --group-column). On with "
        "--regulated-data.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--freeze-layers", type=int, default=0)
    p.add_argument("--num-crossval-splits", type=int, default=1, choices=[0, 1, 5])
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="scgpt: sets the CV folds and the training randomness (head init, dropout, "
        "batch order). geneformer: sets the trainer seed only; its folds come from the "
        "vendored Classifier's fixed shuffle.",
    )
    p.add_argument(
        "--split-sizes",
        default="0.8,0.1,0.1",
        help="train,valid,test fractions (sum to 1). For geneformer 5-fold CV use "
        "0.8,0.2,0 so the folds hold out every cell once; the default evaluates on only "
        "about half the cells.",
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="QLoRA: 4-bit base model + LoRA adapters (needs a CUDA GPU + peft/bitsandbytes).",
    )
    p.add_argument(
        "--force-quantize",
        action="store_true",
        help="Bypass the entry-point QLoRA guard (sets GENEFORMER_FT_ALLOW_QUANTIZE=1 "
        "in the container). Experimental -- QLoRA has not been re-validated on a GPU "
        "since the move to transformers 5. Implies --quantize.",
    )
    p.add_argument("--train-file", default="data.loom")
    p.add_argument(
        "--data-source",
        default="channel",
        choices=["channel", "pbmc3k"],
        help="'channel' reads the train S3 input; 'pbmc3k' self-fetches in-container.",
    )
    p.add_argument(
        "--scgpt-model-root",
        default="/scgpt",
        help="In-container root holding scGPT preset checkpoint dirs (scgpt backend). "
        "When --scgpt-weights-s3 is given, this is set automatically to the mounted "
        "weights channel path.",
    )
    p.add_argument(
        "--scgpt-weights-s3",
        default=None,
        help="S3 prefix holding the scGPT preset checkpoint dirs (e.g. "
        "s3://amzn-s3-demo-bucket/scgpt/ containing scGPT_human/...). Mounted as a SageMaker "
        "'weights' channel at /opt/ml/input/data/weights so the image carries no "
        "baked weights (option A). scgpt backend only.",
    )
    p.add_argument(
        "--scgpt-no-flash-attn",
        action="store_true",
        help="Disable scGPT FlashAttention (use vanilla attention). Default: flash-attn on "
        "(falls back automatically if CUDA/flash-attn is unavailable). scgpt backend only.",
    )
    p.add_argument(
        "--scgpt-grad-checkpointing",
        action="store_true",
        help="Enable scGPT activation (gradient) checkpointing — trades compute for a large "
        "activation-memory saving so a bigger batch fits. scgpt backend only.",
    )
    p.add_argument(
        "--scgpt-max-seq-len",
        type=int,
        default=1200,
        help="Cap the gene-token sequence length to the top-N highest-variance genes "
        "(HVG) to bound attention memory. 0 keeps all genes. scgpt backend only.",
    )
    p.add_argument(
        "--scgpt-class-weighting",
        default="none",
        choices=["none", "balanced", "sqrt"],
        help="Weight the classification loss toward rare cell types: 'balanced' "
        "(inverse class frequency) or 'sqrt' (gentler). Weights come from each fold's "
        "training cells only. scgpt backend only.",
    )
    # --- Security options (see docs/security/aws-baseline.md) ---
    sec = p.add_argument_group("security options")
    sec.add_argument(
        "--allow-mutable-image",
        action="store_true",
        help="Run --image-uri as given even when it is a mutable tag that cannot be resolved "
        "to a digest. By default a tagged ECR URI is resolved to its @sha256 digest before "
        "launch, so the job runs exactly the image that was checked, and anything else is "
        "refused.",
    )
    sec.add_argument(
        "--allow-network",
        action="store_true",
        help="Give the container outbound network access. By default a job that reads S3 data "
        "(--data-source channel) runs under SageMaker network isolation, so a malicious input or "
        "dependency cannot send data out. Needed only if the job must download something. "
        "--regulated-data refuses it. A --data-source pbmc3k job always has network access, "
        "because it downloads its dataset.",
    )
    sec.add_argument(
        "--network-isolation",
        action="store_true",
        help="Kept for old command lines: isolation is the default for S3 data now. Refused "
        "with --data-source pbmc3k.",
    )
    sec.add_argument(
        "--subnets",
        nargs="+",
        default=None,
        help="Run the job inside your VPC on these subnet IDs (needs --security-group-ids).",
    )
    sec.add_argument(
        "--security-group-ids",
        nargs="+",
        default=None,
        help="Security group IDs for the VPC job (needs --subnets).",
    )
    sec.add_argument(
        "--encrypt-inter-container-traffic",
        action="store_true",
        help="Encrypt traffic between containers of a multi-instance job.",
    )
    sec.add_argument(
        "--volume-kms-key",
        default=None,
        help="KMS key ID/ARN that encrypts the training instance's storage volume. Not accepted "
        "for instance types with local NVMe storage (g4dn, g5, g6, p4d, ...), which are "
        "encrypted in hardware.",
    )
    sec.add_argument(
        "--output-kms-key",
        default=None,
        help="KMS key ID/ARN that encrypts the model output written to S3. Default: the output "
        "bucket's own default SSE-KMS key, when the launcher can read it. Without either, "
        "SageMaker uses the account's AWS-managed aws/s3 key, not the bucket's.",
    )
    sec.add_argument(
        "--allow-unverified-inputs",
        action="store_true",
        help="Do not require a SHA256SUMS for each S3 input channel. By default the job refuses "
        "to start unless every channel carries one that lists all its files and matches them "
        "(write one with `scfm-ft checksums DIR`). A swapped pickle or checkpoint runs code, so "
        "turn this off only for data you wrote yourself. --regulated-data refuses it.",
    )
    sec.add_argument(
        "--require-input-hashes",
        action="store_true",
        help="Kept for old command lines: input hashes are required by default now.",
    )
    sec.add_argument(
        "--manifest-sha256",
        nargs="+",
        default=None,
        metavar="CHANNEL=SHA256",
        help="Pin each channel's SHA256SUMS to a hash you recorded outside the bucket, e.g. "
        "train=<64 hex>. Someone who can write the bucket can otherwise replace a file and its "
        "manifest together. `scfm-ft checksums DIR` prints the value. CHANNEL is train or weights. "
        "Required for every S3 input channel unless --allow-unpinned-manifest.",
    )
    sec.add_argument(
        "--allow-unpinned-manifest",
        action="store_true",
        help="Launch without a --manifest-sha256 for an S3 input channel. The job still checks "
        "the channel against the SHA256SUMS beside it, but trusts that manifest, so anyone who "
        "can write the prefix can replace a file and its manifest together. --regulated-data "
        "refuses it.",
    )
    sec.add_argument(
        "--allow-cell-level-split",
        action="store_true",
        help="With --regulated-data, launch without --group-column (for example for data from "
        "a single donor). Cells from one donor then land in both training and held-out data, "
        "which inflates held-out scores. Printed as a warning.",
    )
    sec.add_argument(
        "--regulated-data",
        action="store_true",
        help="Launch only with every control for regulated (PHI/PII) data: turns on network "
        "isolation, --require-input-hashes and encrypted inter-container traffic, and refuses "
        "to start without KMS keys for the volume and output, a VPC (--subnets and "
        "--security-group-ids), --manifest-sha256 for each input channel, and a digest-pinned image.",
    )
    sec.add_argument(
        "--max-input-gib",
        type=float,
        default=4.0,
        help="Largest input file accepted, in GiB (checked before the file is read).",
    )
    sec.add_argument(
        "--max-cells",
        type=int,
        default=1_000_000,
        help="Most cells accepted in the input (checked from the file header).",
    )
    args = p.parse_args()
    if args.allow_unverified_inputs and args.require_input_hashes:
        p.error("--allow-unverified-inputs contradicts --require-input-hashes")
    # Verification is the default; --allow-unverified-inputs is the explicit opt-out (T003, T016).
    args.require_input_hashes = not args.allow_unverified_inputs
    if args.allow_network and args.network_isolation:
        p.error("--allow-network contradicts --network-isolation")
    # Isolation is the default for S3 data; only the self-fetching dataset needs the network (T006).
    if args.data_source == "channel" and not args.allow_network:
        args.network_isolation = True
    return args
#: Where SageMaker mounts the optional scGPT 'weights' channel inside the container.
WEIGHTS_CHANNEL_DIR = "/opt/ml/input/data/weights"


def resolve_scgpt_model_root(scgpt_weights_s3: str | None, scgpt_model_root: str) -> str:
    """Effective scGPT ``model_root`` for the job.

    When a weights S3 prefix is given (option A), the weights are mounted as a
    channel and the root is the channel mount path; otherwise the baked-in /
    explicit ``--scgpt-model-root`` is used (option B / default).
    """
    return WEIGHTS_CHANNEL_DIR if scgpt_weights_s3 else scgpt_model_root


_ECR_URI = re.compile(
    r"^(?P<registry>(?P<account>\d{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com(?:\.cn)?)"
    r"/(?P<repo>[^:@]+)(?::(?P<tag>[^@]+))?(?:@(?P<digest>sha256:[0-9a-f]{64}))?$"
)


def resolve_image_uri(image_uri: str, *, allow_mutable: bool = False, ecr_client=None) -> str:
    """Return ``image_uri`` pinned to its content digest.

    ECR tags can be re-pointed by anyone who can push to the repository, so a job that
    names ``geneformer-ft:latest`` runs whatever that tag points at *when SageMaker pulls
    it*. Resolving the tag to ``@sha256:...`` here fixes the image at launch time, so the
    job runs exactly the bytes that were resolved (and printed, for the audit trail).
    An image that is already a digest is returned unchanged.
    """
    m = _ECR_URI.match(image_uri)
    if m is None:
        if allow_mutable:
            return image_uri
        raise SystemExit(
            f"--image-uri {image_uri!r} is not an ECR image URI, so it cannot be pinned. "
            "Use an ECR image, or pass --allow-mutable-image to run it as given."
        )
    if m["digest"]:
        return image_uri
    if allow_mutable:
        return image_uri
    tag = m["tag"] or "latest"
    try:
        if ecr_client is None:
            import boto3

            ecr_client = boto3.client("ecr", region_name=m["region"])
        details = ecr_client.describe_images(
            registryId=m["account"], repositoryName=m["repo"], imageIds=[{"imageTag": tag}]
        )["imageDetails"]
        digest = details[0]["imageDigest"]
    except Exception as e:  # any lookup failure must stop the launch, with advice
        raise SystemExit(
            f"Could not resolve {m['repo']}:{tag} to a digest ({type(e).__name__}: {e}).\n"
            "Pass the digest yourself (--image-uri <registry>/<repo>@sha256:...; list it with "
            "`aws ecr describe-images`), grant the launcher ecr:DescribeImages, or pass "
            "--allow-mutable-image to run the tag as given."
        ) from e
    pinned = f"{m['registry']}/{m['repo']}@{digest}"
    print(f"[launch] image {m['repo']}:{tag} resolved to {pinned}")
    return pinned


#: Offline switches that make a network-isolated job fail fast instead of hanging on a
#: connection attempt (Hugging Face libraries otherwise try to reach the Hub).
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


_JOB_NAME_OK = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")


def default_job_name(args: argparse.Namespace, now: datetime | None = None) -> str:
    """A valid SageMaker job name. Left to itself the SDK names the job after the image URI, and
    a digest-pinned URI contains ``@`` and ``:``, which SageMaker rejects."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    base = re.sub(r"[^a-zA-Z0-9]+", "-", f"{args.backend}-{args.preset}").strip("-")
    return f"{base[:50]}-{stamp}"


def check_job_name(name: str) -> None:
    if not _JOB_NAME_OK.match(name):
        raise SystemExit(
            f"--job-name {name!r} is not valid: use 1 to 63 letters, digits and hyphens, "
            "starting with a letter or digit."
        )


#: Instance families with local NVMe storage. SageMaker encrypts that storage in hardware and
#: rejects a VolumeKmsKeyId for them (docs: "Certain Nitro-based instances include local storage").
_LOCAL_STORAGE_FAMILIES = (
    "g4dn", "g5", "g6", "g6e", "p3dn", "p4d", "p4de", "p5", "p5e", "p5en", "p6-b200", "trn1", "trn1n", "trn2",
)


def has_local_storage(instance_type: str) -> bool:
    """True for instance types whose storage SageMaker encrypts in hardware (no volume KMS key)."""
    family = instance_type.removeprefix("ml.").split(".")[0]
    return family in _LOCAL_STORAGE_FAMILIES or bool(re.fullmatch(r"[a-z]+\d+[a-z]*d", family))


def check_max_run(args: argparse.Namespace) -> None:
    """Refuse a non-positive ``--max-run``, or one above the ceiling without ``--allow-long-run``."""
    if args.max_run <= 0:
        raise SystemExit(f"--max-run must be a positive number of seconds, got {args.max_run}.")
    if args.max_run > MAX_RUN_CEILING and not args.allow_long_run:
        raise SystemExit(
            f"--max-run {args.max_run} is above the {MAX_RUN_CEILING} s ceiling "
            "(docs/iam/sagemaker-launcher-policy.json denies longer jobs). Lower it, or pass "
            "--allow-long-run if you have not applied that policy and mean it."
        )


def manifest_pins(args: argparse.Namespace) -> dict[str, str]:
    """Return ``{"train": hex, ...}`` from ``--manifest-sha256``, rejecting malformed values."""
    pins: dict[str, str] = {}
    for item in args.manifest_sha256 or []:
        name, sep, value = item.partition("=")
        value = value.strip().lower()
        if not sep or name not in ("train", "weights") or not _SHA256_HEX.match(value):
            raise SystemExit(f"--manifest-sha256 {item!r}: use train=<64 hex> or weights=<64 hex>.")
        if name in pins:
            raise SystemExit(f"--manifest-sha256 names the {name} channel twice.")
        pins[name] = value
    return pins


def apply_regulated_data(args: argparse.Namespace) -> None:
    """Turn on the controls that ``--regulated-data`` implies, or refuse to launch.

    Each missing control is listed, so one run shows everything that is still needed.
    """
    if not args.regulated_data:
        return
    pins = manifest_pins(args)
    missing: list[str] = []
    if args.data_source != "channel":
        missing.append("--data-source channel (a self-fetched dataset has no manifest to pin)")
    if not args.train_s3:
        missing.append("--train-s3")
    if not args.volume_kms_key and not has_local_storage(args.instance_type):
        missing.append("--volume-kms-key (or an instance type with hardware-encrypted local storage)")
    if not args.output_kms_key:
        missing.append("--output-kms-key")
    if not (args.subnets and args.security_group_ids):
        missing.append("--subnets and --security-group-ids")
    if "train" not in pins:
        missing.append("--manifest-sha256 train=<hash>")
    if args.scgpt_weights_s3 and "weights" not in pins:
        missing.append("--manifest-sha256 weights=<hash>")
    if args.allow_mutable_image:
        missing.append("remove --allow-mutable-image")
    if args.allow_unverified_inputs:
        missing.append("remove --allow-unverified-inputs")
    if args.allow_unpinned_manifest:
        missing.append("remove --allow-unpinned-manifest")
    if args.allow_network:
        missing.append("remove --allow-network")
    if not args.group_column and not args.allow_cell_level_split:
        missing.append(
            "--group-column <donor column> (or --allow-cell-level-split for single-donor data)"
        )
    if missing:
        raise SystemExit("--regulated-data needs:\n  " + "\n  ".join(missing))
    args.network_isolation = True
    args.require_input_hashes = True
    args.encrypt_inter_container_traffic = True
    args.privacy_audit = True


def unpinned_channels(args: argparse.Namespace) -> list[str]:
    """S3 input channels the job will read that have no ``--manifest-sha256`` pin."""
    pins = manifest_pins(args)
    channels = []
    if args.data_source == "channel" and args.train_s3:
        channels.append("train")
    if args.scgpt_weights_s3:
        channels.append("weights")
    return [c for c in channels if c not in pins]


def check_manifest_pins(args: argparse.Namespace) -> None:
    """Refuse to launch an S3 input channel whose ``SHA256SUMS`` is not pinned (T005, T016).

    Without a pin the job checks the files against the manifest beside them, so a writer of
    the prefix can replace both. ``--allow-unpinned-manifest`` is the explicit opt-out, and
    ``--allow-unverified-inputs`` (no manifest required at all) implies it.
    """
    if args.allow_unpinned_manifest or args.allow_unverified_inputs:
        return
    unpinned = unpinned_channels(args)
    if unpinned:
        flags = " ".join(f"{c}=<hash>" for c in unpinned)
        raise SystemExit(
            f"No --manifest-sha256 for the {' and '.join(unpinned)} channel. Run "
            "`scfm-ft checksums DIR` before uploading, record the manifest hash it prints "
            f"outside the bucket, and pass --manifest-sha256 {flags}. To launch without a pin "
            "(the job then trusts the SHA256SUMS in the bucket), pass --allow-unpinned-manifest."
        )


def cell_level_split_warnings(args: argparse.Namespace) -> list[str]:
    """Warning for human data split by cell rather than by donor."""
    if args.regulated_data and not args.group_column and args.allow_cell_level_split:
        return [
            (
                "--allow-cell-level-split: cells from one donor can land in both training and "
                "held-out data, so held-out scores are optimistic and a membership-inference test "
                "would understate what the model leaks. Pass --group-column if the data has donors."
            )
        ]
    return []


def unverified_input_warnings(args: argparse.Namespace) -> list[str]:
    """Warnings for S3 inputs that the job will not insist on verifying (T003, T016)."""
    channels = []
    if args.data_source == "channel" and args.train_s3:
        channels.append("--train-s3")
    if args.scgpt_weights_s3:
        channels.append("--scgpt-weights-s3")
    if not channels or args.require_input_hashes:
        unpinned = unpinned_channels(args)
        if unpinned and args.allow_unpinned_manifest:
            return [
                (
                    f"the {' and '.join(unpinned)} channel's SHA256SUMS is not pinned "
                    "(--allow-unpinned-manifest): the job trusts the manifest it finds in the bucket, "
                    "so anyone who can write that prefix can replace a file and its manifest together. "
                    "Pass --manifest-sha256 with the hash `scfm-ft checksums DIR` printed."
                )
            ]
        return []
    return [
        f"{' and '.join(channels)} will be read without requiring a SHA256SUMS: a missing manifest "
        "only prints a warning in the job, so swapped or malicious files (pickles run code when "
        "loaded) are not stopped. Write a manifest with `scfm-ft checksums DIR`, upload it with the "
        "data and drop --allow-unverified-inputs (add --manifest-sha256 too), or use --regulated-data."
    ]


def _s3_bucket(uri: str) -> str:
    return uri.removeprefix("s3://").split("/", 1)[0]


def _aws_error_code(exc: Exception) -> str:
    return getattr(exc, "response", {}).get("Error", {}).get("Code") or type(exc).__name__


def bucket_baseline_problems(s3, bucket: str) -> list[str]:
    """Check one bucket against docs/security/aws-baseline.md; fail closed when a check cannot run.

    Versioning on, every Block Public Access setting on, default encryption with KMS, and a
    policy that denies non-TLS requests. ``s3`` is a boto3 S3 client (a stub in tests).
    """
    problems: list[str] = []

    def cannot(check: str, exc: Exception) -> None:
        problems.append(f"{bucket}: cannot check {check} ({_aws_error_code(exc)})")

    try:
        if s3.get_bucket_versioning(Bucket=bucket).get("Status") != "Enabled":
            problems.append(f"{bucket}: versioning is not enabled")
    except Exception as e:  # noqa: BLE001 - any failure means we cannot vouch for the bucket
        cannot("versioning", e)

    try:
        cfg = s3.get_public_access_block(Bucket=bucket)["PublicAccessBlockConfiguration"]
        off = [k for k in ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets") if not cfg.get(k)]
        if off:
            problems.append(f"{bucket}: Block Public Access is off for {', '.join(off)}")
    except Exception as e:  # noqa: BLE001
        if _aws_error_code(e) == "NoSuchPublicAccessBlockConfiguration":
            problems.append(f"{bucket}: Block Public Access is not configured")
        else:
            cannot("Block Public Access", e)

    try:
        rules = s3.get_bucket_encryption(Bucket=bucket)["ServerSideEncryptionConfiguration"]["Rules"]
        algos = {r.get("ApplyServerSideEncryptionByDefault", {}).get("SSEAlgorithm") for r in rules}
        if not algos & {"aws:kms", "aws:kms:dsse"}:
            problems.append(f"{bucket}: default encryption is {', '.join(sorted(a or 'none' for a in algos))}, not SSE-KMS")
    except Exception as e:  # noqa: BLE001
        if _aws_error_code(e) == "ServerSideEncryptionConfigurationNotFoundError":
            problems.append(f"{bucket}: no default encryption")
        else:
            cannot("default encryption", e)

    try:
        policy = json.loads(s3.get_bucket_policy(Bucket=bucket)["Policy"])
        statements = policy.get("Statement", [])
        statements = [statements] if isinstance(statements, dict) else statements
        tls_only = any(
            st.get("Effect") == "Deny"
            and str(st.get("Condition", {}).get("Bool", {}).get("aws:SecureTransport", "")).lower() == "false"
            for st in statements
        )
        if not tls_only:
            problems.append(f"{bucket}: the bucket policy does not deny requests made without TLS")
    except Exception as e:  # noqa: BLE001
        if _aws_error_code(e) == "NoSuchBucketPolicy":
            problems.append(f"{bucket}: no bucket policy, so nothing denies requests made without TLS")
        else:
            cannot("the bucket policy", e)
    return problems


def preflight_buckets(args: argparse.Namespace, output_path: str, s3=None) -> None:
    """With --regulated-data, refuse to launch if any bucket the job touches misses the baseline."""
    if not args.regulated_data or args.skip_bucket_check:
        return
    buckets = [_s3_bucket(u) for u in (args.train_s3, args.scgpt_weights_s3, output_path) if u]
    if s3 is None:
        import boto3

        s3 = boto3.client("s3")
    problems: list[str] = []
    for bucket in dict.fromkeys(buckets):
        problems += bucket_baseline_problems(s3, bucket)
    if problems:
        raise SystemExit(
            "--regulated-data: the S3 buckets do not meet docs/security/aws-baseline.md:\n  "
            + "\n  ".join(problems)
            + "\nFix them, or pass --skip-bucket-check to launch anyway (this weakens --regulated-data)."
        )


_AWS_MANAGED_S3_KEY = ("aws/s3", "alias/aws/s3")


def default_output_kms_key(s3, bucket: str) -> tuple[str | None, str | None]:
    """Return ``(key, note)``: the output bucket's default customer-managed KMS key, if any.

    SageMaker does not use a bucket's default encryption for model output. Without
    ``OutputDataConfig.KmsKeyId`` it encrypts ``model.tar.gz`` with the account's AWS-managed
    ``aws/s3`` key, so a bucket set up with a customer-managed key still gets output under a
    key whose policy cannot be changed. Reading the bucket's default lets the launcher pass that
    key instead. ``note`` explains why no key was found; it is ``None`` when one was.
    """
    try:
        rules = s3.get_bucket_encryption(Bucket=bucket)["ServerSideEncryptionConfiguration"]["Rules"]
    except Exception as e:  # noqa: BLE001 - any failure means we cannot tell
        return None, (
            f"could not read the default encryption of {bucket} ({_aws_error_code(e)}), so model "
            "output will be encrypted with the account's aws/s3 key. Pass --output-kms-key to choose the key."
        )
    for rule in rules:
        default = rule.get("ApplyServerSideEncryptionByDefault", {})
        key = default.get("KMSMasterKeyID")
        if default.get("SSEAlgorithm") in ("aws:kms", "aws:kms:dsse") and key:
            if key.split(":")[-1] in _AWS_MANAGED_S3_KEY:
                return None, None
            return key, None
    return None, None


def apply_output_kms_default(args: argparse.Namespace, output_path: str, s3) -> None:
    """Fill ``args.output_kms_key`` from the output bucket's default key when none was given."""
    if args.output_kms_key:
        return
    key, note = default_output_kms_key(s3, _s3_bucket(output_path))
    if key:
        args.output_kms_key = key
        print(f"[launch] model output will be encrypted with the output bucket's default key {key}")
    elif note:
        print(f"[launch] NOTE: {note}", file=sys.stderr)


def security_options(args: argparse.Namespace) -> tuple[dict, dict]:
    """Map the security flags to ``(Estimator kwargs, container environment)``.

    Only options the user set are returned, so SageMaker's own defaults apply otherwise.
    """
    if args.network_isolation and args.data_source == "pbmc3k":
        raise SystemExit(
            "--network-isolation cannot be combined with --data-source pbmc3k (the self-fetch "
            "downloads the dataset). Stage it instead: `scfm-ft load-pbmc3k`, then "
            "`scfm-ft checksums <dir>`, upload both to S3 and pass --train-s3."
        )
    if bool(args.subnets) != bool(args.security_group_ids):
        raise SystemExit("--subnets and --security-group-ids must be given together.")
    if args.volume_kms_key and has_local_storage(args.instance_type):
        raise SystemExit(
            f"--volume-kms-key cannot be used with {args.instance_type}: its storage is local NVMe, "
            "which SageMaker encrypts in hardware and does not accept a KMS key for. Remove the "
            "flag, or pick an instance type with EBS storage."
        )

    kwargs: dict = {}
    env: dict = {}
    if args.network_isolation:
        kwargs["enable_network_isolation"] = True
        env.update(OFFLINE_ENV)
    if args.subnets:
        kwargs["subnets"] = list(args.subnets)
        kwargs["security_group_ids"] = list(args.security_group_ids)
    if args.encrypt_inter_container_traffic:
        kwargs["encrypt_inter_container_traffic"] = True
    if args.volume_kms_key:
        kwargs["volume_kms_key"] = args.volume_kms_key
    if args.output_kms_key:
        kwargs["output_kms_key"] = args.output_kms_key
    return kwargs, env


def build_hyperparameters(args: argparse.Namespace, *, scgpt_model_root: str, quantize: bool) -> dict:
    """Hyperparameters forwarded to ``sm_entry.py`` (SageMaker passes them as strings)."""
    hp = {
        "name": args.name,
        "backend": args.backend,
        "preset": args.preset,
        "label-column": args.label_column,
        "classifier-kind": args.classifier_kind,
        "epochs": args.epochs,
        "learning-rate": args.learning_rate,
        "batch-size": args.batch_size,
        "freeze-layers": args.freeze_layers,
        "num-crossval-splits": args.num_crossval_splits,
        "seed": args.seed,
        "split-sizes": args.split_sizes,
        "quantize": "true" if quantize else "false",
        "train-file": args.train_file,
        "data-source": args.data_source,
        "scgpt-model-root": scgpt_model_root,
        "scgpt-flash-attn": "false" if args.scgpt_no_flash_attn else "true",
        "scgpt-grad-checkpointing": "true" if args.scgpt_grad_checkpointing else "false",
        "scgpt-max-seq-len": str(args.scgpt_max_seq_len),
        "scgpt-class-weighting": args.scgpt_class_weighting,
        "require-input-hashes": "true" if args.require_input_hashes else "false",
        "privacy-audit": "true" if args.privacy_audit else "false",
        "max-input-gib": str(args.max_input_gib),
        "max-cells": str(args.max_cells),
    }
    pins = manifest_pins(args)
    if pins:  # omitted when unset: SageMaker hyperparameters must not be empty strings
        hp["manifest-sha256"] = ",".join(f"{k}={v}" for k, v in sorted(pins.items()))
    if args.group_column:  # omitted when unset, for the same reason
        hp["group-column"] = args.group_column
    return hp


def main() -> None:
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput

    from sagemaker import Session

    args = _parse()
    check_max_run(args)
    args.job_name = args.job_name or default_job_name(args)
    check_job_name(args.job_name)
    apply_regulated_data(args)
    check_manifest_pins(args)
    pins = manifest_pins(args)
    if pins and not args.require_input_hashes:
        args.require_input_hashes = True  # a pinned manifest only makes sense with verification on
    for warning in unverified_input_warnings(args) + cell_level_split_warnings(args):
        print(f"[launch] WARNING: {warning}", file=sys.stderr)

    # --force-quantize implies --quantize and bypasses the entry-point guard.
    quantize = args.quantize or args.force_quantize

    # Option A: when a weights S3 prefix is given, SageMaker mounts it as the
    # 'weights' channel at /opt/ml/input/data/weights; point the backend's model
    # root there so the image needs no baked weights.
    scgpt_model_root = resolve_scgpt_model_root(args.scgpt_weights_s3, args.scgpt_model_root)

    hyperparameters = build_hyperparameters(
        args, scgpt_model_root=scgpt_model_root, quantize=quantize
    )

    # Run exactly the image that was checked: pin a tag to its digest now (and print it).
    image_uri = resolve_image_uri(args.image_uri, allow_mutable=args.allow_mutable_image)

    # Network isolation / VPC / KMS options, validated before anything is created.
    security_kwargs, security_env = security_options(args)

    # The entry point refuses --quantize by default (unsupported on the pinned
    # transformers). --force-quantize lifts that guard inside the container.
    environment: dict = dict(security_env)
    if args.force_quantize:
        environment["GENEFORMER_FT_ALLOW_QUANTIZE"] = "1"

    spot_kwargs: dict = {}
    if args.use_spot:
        spot_kwargs = {
            "use_spot_instances": True,
            "max_wait": args.max_run + 3600,
        }

    session = Session()
    output_path = resolve_s3_paths(session.default_bucket(), args.s3_prefix, args.output_s3)
    s3 = session.boto_session.client("s3")
    preflight_buckets(args, output_path, s3=s3 if args.regulated_data else None)
    apply_output_kms_default(args, output_path, s3)
    if args.output_kms_key:
        security_kwargs["output_kms_key"] = args.output_kms_key

    estimator = Estimator(
        sagemaker_session=session,
        image_uri=image_uri,
        role=args.role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        # No entry_point, source_dir, dependencies or git_config, on purpose. Giving the SDK any
        # of them makes it tar that directory, upload it to S3 and have the job fetch it, which
        # (a) uploads whatever happens to sit under sagemaker/ (data, weights) and (b) makes the
        # job's code depend on who can write the S3 code prefix. The image already bakes
        # sm_entry.py into /opt/ml/code and sets SAGEMAKER_PROGRAM, and the training toolkit
        # runs that copy when no submit directory is given. tests/test_launch_security.py
        # fails if one of these arguments comes back.
        hyperparameters=hyperparameters,
        environment=environment,
        output_path=output_path,
        # The profiler and debugger hook are on by default and write to the bucket
        # root; nothing here reads them, and leaving them off keeps every write the
        # role makes under <s3-prefix>/output.
        disable_profiler=True,
        debugger_hook_config=False,
        max_run=args.max_run,
        **security_kwargs,
        **spot_kwargs,
    )

    if args.data_source == "pbmc3k":
        # Self-fetch in-container: no train channel needed.
        fit_inputs = None
    else:
        if not args.train_s3:
            raise SystemExit("--train-s3 is required unless --data-source pbmc3k")
        fit_inputs = {"train": TrainingInput(args.train_s3)}

    # Option A: mount the scGPT weights prefix as a 'weights' channel (scgpt backend).
    if args.scgpt_weights_s3:
        if args.backend != "scgpt":
            raise SystemExit("--scgpt-weights-s3 applies only to --backend scgpt")
        fit_inputs = fit_inputs or {}
        fit_inputs["weights"] = TrainingInput(args.scgpt_weights_s3)

    estimator.fit(
        inputs=fit_inputs,
        job_name=args.job_name,
    )
    print("Image:", image_uri)
    print("Model artifacts:", estimator.model_data)


if __name__ == "__main__":
    main()
