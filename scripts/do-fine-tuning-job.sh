#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#
# do-fine-tuning-job.sh -- one-command launcher for a Geneformer fine-tuning job
# on SageMaker (GPU training job, held-out eval, model.tar.gz to S3).
#
# It packages everything the manual launch needs:
#   * bootstraps an isolated launcher venv pinned to the v2 SageMaker SDK
#     (the project .venv does NOT carry sagemaker; the v3 SDK dropped the
#     sagemaker.estimator module launch.py imports, so the pin matters)
#   * resolves creds via a cross-account AWS profile (default: $AWS_PROFILE_NAME, below)
#   * fills in the ECR image + SageMaker execution role for your AWS account
#     (the account is taken from the required AWS_ACCOUNT_ID environment variable)
#   * picks a sensible GPU instance + batch size per model preset
#     (V2-104M OOMs on g5.xlarge at batch 24 -- the per-preset default is 12)
#   * generates a timestamped job name and blocks until the job completes
#
# Every default is overridable with a flag; run with -h for the full list.
#
# REQUIRED ENVIRONMENT:
#   AWS_ACCOUNT_ID   Your 12-digit AWS account ID. The script fails fast if it is
#                    unset or malformed. It is used to build the ECR image URI and
#                    the SageMaker execution-role ARN. Example:
#                        export AWS_ACCOUNT_ID=123456789012
#                    Optionally also set AWS_PROFILE_NAME (default: default) to the
#                    named AWS CLI profile that resolves to that account.
#
# Examples:
#   scripts/do-fine-tuning-job.sh                       # V2-104M PBMC3k, eval on
#   scripts/do-fine-tuning-job.sh --preset V1-10M       # smaller/cheaper
#   scripts/do-fine-tuning-job.sh --data-s3 s3://amzn-s3-demo-bucket/mydata/  # your own bucket + data
#   scripts/do-fine-tuning-job.sh --preset V2-316M --instance ml.p4d.24xlarge
#   scripts/do-fine-tuning-job.sh --dry-run             # print the command, launch nothing

set -euo pipefail

# --------------------------------------------------------------------------- #
# Required: AWS account ID from the environment (no hardcoded account).        #
# Fail fast with a clear message if it is unset or not 12 digits.             #
# --------------------------------------------------------------------------- #
if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then
  echo "ERROR: AWS_ACCOUNT_ID is not set." >&2
  echo "       Set it to your 12-digit AWS account ID, e.g.:" >&2
  echo "         export AWS_ACCOUNT_ID=123456789012" >&2
  echo "       (used to build the ECR image URI and SageMaker role ARN)." >&2
  exit 2
fi
if [[ ! "${AWS_ACCOUNT_ID}" =~ ^[0-9]{12}$ ]]; then
  echo "ERROR: AWS_ACCOUNT_ID='${AWS_ACCOUNT_ID}' is not a valid 12-digit AWS account ID." >&2
  exit 2
fi

# --------------------------------------------------------------------------- #
# Defaults (account comes from AWS_ACCOUNT_ID; region/profile overridable)     #
# --------------------------------------------------------------------------- #
REGION="us-east-1"
ACCOUNT="${AWS_ACCOUNT_ID}"
AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-default}"  # AWS CLI profile for ACCOUNT
IMAGE_URI=""                                 # empty -> newest build tag in ECR (see below)
ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/GeneformerSageMakerExecution"
PYENV_BIN="${PYENV_BIN:-}"                   # optional extra bin dir to put first on PATH

# Task defaults
PRESET="V2-104M"
EPOCHS=10
LR="5e-5"
BATCH_SIZE=""                                # empty -> per-preset default below
NUM_CROSSVAL_SPLITS=1                        # >=1 -> held-out eval; 0 -> train-only
QUANTIZE=0                                   # 1 -> QLoRA (4-bit base + LoRA adapters) -- UNSUPPORTED, see guard below
FORCE_QUANTIZE=0                             # 1 -> bypass the unsupported-QLoRA guard (expert/experimental)
LABEL_COLUMN="cell_type"
GROUP_COLUMN=""                              # obs column naming each cell's donor; empty = split by cell
PRIVACY_AUDIT=0                              # 1 -> membership-inference audit after training
CLASSIFIER_KIND="cell"
FREEZE_LAYERS=0
DATA_SOURCE="pbmc3k"                         # pbmc3k self-fetches; set --data-s3 to use your own
DATA_S3=""
MANIFEST_SHA256=""                           # SHA-256 of the --data-s3 SHA256SUMS (scfm-ft checksums)
INSTANCE_TYPE=""                             # empty -> per-preset default below
NAME=""                                      # empty -> derived from preset
JOB_NAME=""                                  # empty -> timestamped
USE_SPOT=0
DRY_RUN=0
STRIP_COLOR=1                                # 1 -> strip ANSI color from output

# --------------------------------------------------------------------------- #
# Repo + launcher-venv locations
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER_VENV="${REPO_ROOT}/.sm-launcher"
# The launcher venv is installed from this hash-locked file (see requirements-launcher.in
# for how to regenerate it); nothing in it is fetched unhashed.
LAUNCHER_REQS="${SCRIPT_DIR}/requirements-launcher.txt"

usage() {
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<EOF

Options:
  --preset NAME             Model preset: V1-10M | V2-104M | V2-316M  (default: ${PRESET})
  --epochs N                Training epochs                           (default: ${EPOCHS})
  --learning-rate LR        Learning rate                             (default: ${LR})
  --batch-size N            Per-device batch size   (default: per-preset; V2-104M=12)
  --num-crossval-splits N   0=train-only, 1=single split, 5=5-fold CV (default: ${NUM_CROSSVAL_SPLITS})
  --quantize                QLoRA (4-bit + LoRA). UNSUPPORTED on this Geneformer/transformers
                            pin -- rejected before launch (see note below). Default: off.
  --force-quantize          Bypass the QLoRA guard (expert/experimental; implies --quantize)
  --label-column COL        obs column to classify on                 (default: ${LABEL_COLUMN})
  --group-column COL        obs column naming each cell's donor: split and CV by donor, so no
                            donor is in both train and held-out data  (default: split by cell)
  --privacy-audit           Membership-inference audit after training (writes
                            <name>_privacy_audit.json; on with -- --regulated-data)
  --classifier-kind KIND    cell | gene                               (default: ${CLASSIFIER_KIND})
  --freeze-layers N         Transformer layers to freeze              (default: ${FREEZE_LAYERS})
  --data-s3 S3URI           Use your own data at this S3 prefix (sets data-source=channel)
  --manifest-sha256 HASH    SHA-256 of that prefix's SHA256SUMS, recorded outside the bucket
                            (`scfm-ft checksums DIR` prints it). Required with --data-s3
                            unless you pass -- --allow-unpinned-manifest
  --data-source SRC         pbmc3k | channel                          (default: ${DATA_SOURCE})
  --instance TYPE           SageMaker instance type (default: per-preset)
  --role ARN                SageMaker execution role ARN              (default: ${ROLE_ARN})
  --image-uri URI           ECR training image                        (default: newest b<N> build tag)
  --profile NAME            AWS CLI profile for the training account  (default: ${AWS_PROFILE_NAME})
  --region REGION           AWS region                                (default: ${REGION})
  --name NAME               Run name hyperparameter        (default: derived from preset)
  --job-name NAME           SageMaker job name             (default: timestamped)
  --use-spot                Use managed spot instances (cheaper, can be interrupted)
  --no-strip-color          Keep the raw colored SDK output (default: strip ANSI color)
  --dry-run                 Print the launch command and exit without submitting
  -- ARGS...                Pass everything after -- to sagemaker/launch.py unchanged
                            (e.g. -- --max-run 7200)
  -h, --help                Show this help
EOF
}

# --------------------------------------------------------------------------- #
# Parse arguments
# --------------------------------------------------------------------------- #
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)              PRESET="$2"; shift 2 ;;
    --epochs)              EPOCHS="$2"; shift 2 ;;
    --learning-rate)       LR="$2"; shift 2 ;;
    --batch-size)          BATCH_SIZE="$2"; shift 2 ;;
    --num-crossval-splits) NUM_CROSSVAL_SPLITS="$2"; shift 2 ;;
    --quantize)            QUANTIZE=1; shift ;;
    --force-quantize)      QUANTIZE=1; FORCE_QUANTIZE=1; shift ;;
    --label-column)        LABEL_COLUMN="$2"; shift 2 ;;
    --group-column)        GROUP_COLUMN="$2"; shift 2 ;;
    --privacy-audit)       PRIVACY_AUDIT=1; shift ;;
    --classifier-kind)     CLASSIFIER_KIND="$2"; shift 2 ;;
    --freeze-layers)       FREEZE_LAYERS="$2"; shift 2 ;;
    --data-s3)             DATA_S3="$2"; DATA_SOURCE="channel"; shift 2 ;;
    --data-source)         DATA_SOURCE="$2"; shift 2 ;;
    --manifest-sha256)     MANIFEST_SHA256="$2"; shift 2 ;;
    --instance)            INSTANCE_TYPE="$2"; shift 2 ;;
    --role)                ROLE_ARN="$2"; shift 2 ;;
    --image-uri)           IMAGE_URI="$2"; shift 2 ;;
    --profile)             AWS_PROFILE_NAME="$2"; shift 2 ;;
    --region)              REGION="$2"; shift 2 ;;
    --name)                NAME="$2"; shift 2 ;;
    --job-name)            JOB_NAME="$2"; shift 2 ;;
    --use-spot)            USE_SPOT=1; shift ;;
    --no-strip-color)      STRIP_COLOR=0; shift ;;
    --dry-run)             DRY_RUN=1; shift ;;
    -h|--help)             usage; exit 0 ;;
    --)                    shift; PASSTHROUGH=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

# --------------------------------------------------------------------------- #
# Default image: the newest b<N> build tag. The repository has immutable tags, so
# `latest` no longer moves and would silently pin an old build. launch.py then resolves
# the tag to its digest. If the lookup fails, fall back to `latest` and say so.
# --------------------------------------------------------------------------- #
if [[ -z "${IMAGE_URI}" ]]; then
  # shellcheck source=scripts/image_tag.sh
  source "${SCRIPT_DIR}/image_tag.sh"
  IMAGE_REPO_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/geneformer-ft"
  if IMAGE_TAG="$(newest_build_tag geneformer-ft "${REGION}" "${AWS_PROFILE_NAME}")"; then
    IMAGE_URI="${IMAGE_REPO_URI}:${IMAGE_TAG}"
  else
    echo "WARNING: could not find a b<N> build tag in geneformer-ft; using :latest (which does not" >&2
    echo "         move once the repository is immutable). Pass --image-uri to choose an image." >&2
    IMAGE_URI="${IMAGE_REPO_URI}:latest"
  fi
fi

# --------------------------------------------------------------------------- #
# Per-preset instance + batch defaults (only when not overridden)
#   ml.g4dn.xlarge  (T4  16GB) -> V1-10M
#   ml.g5.xlarge    (A10G 24GB) -> V2-104M  (batch 24 OOMs; use 12)
#   ml.p4d.24xlarge (A100 80GB) -> V2-316M
# --------------------------------------------------------------------------- #
case "${PRESET}" in
  V1-10M)
    [[ -z "${INSTANCE_TYPE}" ]] && INSTANCE_TYPE="ml.g4dn.xlarge"
    [[ -z "${BATCH_SIZE}" ]] && BATCH_SIZE=12
    ;;
  V2-104M)
    [[ -z "${INSTANCE_TYPE}" ]] && INSTANCE_TYPE="ml.g5.xlarge"
    [[ -z "${BATCH_SIZE}" ]] && BATCH_SIZE=12       # 24 OOMs on A10G 24GB
    ;;
  V2-316M)
    [[ -z "${INSTANCE_TYPE}" ]] && INSTANCE_TYPE="ml.p4d.24xlarge"
    [[ -z "${BATCH_SIZE}" ]] && BATCH_SIZE=12
    ;;
  *)
    echo "ERROR: unknown preset '${PRESET}' (expected V1-10M | V2-104M | V2-316M)" >&2
    exit 2
    ;;
esac

[[ -z "${NAME}" ]] && NAME="$(echo "${PRESET}" | tr 'A-Z-' 'a-z_')_run"
if [[ -z "${JOB_NAME}" ]]; then
  JOB_NAME="geneformer-$(echo "${PRESET}" | tr 'A-Z' 'a-z')-$(date -u +%Y%m%d-%H%M%S)"
fi

# --------------------------------------------------------------------------- #
# QLoRA guard: --quantize is EXPERIMENTAL for the classifier fine-tune path. It
# was validated on transformers 4.46 with a workaround for a head-add crash on the
# 4-bit Byte weights; transformers 5 fixed that crash upstream, but the 4-bit path
# has not been re-run on a GPU since the upgrade. Reject BEFORE launching (saves a
# wasted job + GPU spin-up). --force-quantize bypasses and forwards the entry-point
# escape hatch.
# --------------------------------------------------------------------------- #
if [[ "${QUANTIZE}" -eq 1 && "${FORCE_QUANTIZE}" -ne 1 ]]; then
  cat >&2 <<'EOF'
ERROR: --quantize (QLoRA) is experimental in this Geneformer build.

  It was validated on transformers 4.46 and has not been re-run on a GPU since the
  move to transformers 5.

  Use full-precision fine-tuning (drop --quantize). It fits every supported preset:
    V2-104M  -> ml.g5.xlarge    (batch 12)
    V2-316M  -> ml.p4d.24xlarge

  To run it anyway, rerun with --force-quantize.
EOF
  exit 2
fi

# Optional: PYENV_BIN puts a specific Python first on PATH for the non-interactive shell
[[ -n "${PYENV_BIN}" ]] && export PATH="${PYENV_BIN}:${PATH}"


# --------------------------------------------------------------------------- #
# Bootstrap the launcher venv (idempotent) -- isolated from the project .venv.   #
# It runs with your AWS credentials, so it is installed only from the hash-locked #
# requirements-launcher.txt (--require-hashes): a changed or malicious release on #
# the package index is refused instead of installed. A venv built from an older   #
# or different lock is recreated.                                                 #
# --------------------------------------------------------------------------- #
_sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
if [[ ! -f "${LAUNCHER_REQS}" ]]; then
  echo "ERROR: ${LAUNCHER_REQS} is missing; cannot build the launcher venv." >&2
  exit 2
fi
REQS_SHA="$(_sha256_of "${LAUNCHER_REQS}")"
STAMP="${LAUNCHER_VENV}/.requirements.sha256"
if [[ -x "${LAUNCHER_VENV}/bin/python" && "$(cat "${STAMP}" 2>/dev/null || true)" != "${REQS_SHA}" ]]; then
  echo ">> Launcher venv was not built from the current requirements-launcher.txt; recreating it"
  # Only ever delete the venv this script owns.
  if [[ "${LAUNCHER_VENV}" == "${REPO_ROOT}/.sm-launcher" && -f "${LAUNCHER_VENV}/pyvenv.cfg" ]]; then
    rm -rf "${LAUNCHER_VENV}"
  else
    echo "ERROR: refusing to delete ${LAUNCHER_VENV}; remove it yourself and rerun." >&2
    exit 2
  fi
fi
if [[ ! -x "${LAUNCHER_VENV}/bin/python" ]]; then
  echo ">> Creating launcher venv at ${LAUNCHER_VENV} from ${LAUNCHER_REQS##*/} (hashes required)"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 "${LAUNCHER_VENV}"
    VIRTUAL_ENV="${LAUNCHER_VENV}" uv pip install --require-hashes -r "${LAUNCHER_REQS}"
  else
    # sagemaker 2.x imports distutils, which Python 3.12 removed, so this needs 3.11.
    if ! command -v python3.11 >/dev/null 2>&1; then
      echo "ERROR: need uv or python3.11 to build the launcher venv." >&2
      exit 2
    fi
    python3.11 -m venv "${LAUNCHER_VENV}"
    # No `pip install --upgrade pip`: that would be an unhashed download.
    "${LAUNCHER_VENV}/bin/pip" install --quiet --require-hashes -r "${LAUNCHER_REQS}"
  fi
  printf '%s\n' "${REQS_SHA}" > "${STAMP}"
fi
LAUNCHER_PY="${LAUNCHER_VENV}/bin/python"

# --------------------------------------------------------------------------- #
# Build the launch.py argument vector
# --------------------------------------------------------------------------- #
ARGS=(
  "${REPO_ROOT}/sagemaker/launch.py"
  --image-uri "${IMAGE_URI}"
  --role "${ROLE_ARN}"
  --instance-type "${INSTANCE_TYPE}"
  --preset "${PRESET}"
  --epochs "${EPOCHS}"
  --learning-rate "${LR}"
  --batch-size "${BATCH_SIZE}"
  --freeze-layers "${FREEZE_LAYERS}"
  --num-crossval-splits "${NUM_CROSSVAL_SPLITS}"
  --label-column "${LABEL_COLUMN}"
  --classifier-kind "${CLASSIFIER_KIND}"
  --data-source "${DATA_SOURCE}"
  --name "${NAME}"
  --job-name "${JOB_NAME}"
)
[[ -n "${GROUP_COLUMN}" ]] && ARGS+=( --group-column "${GROUP_COLUMN}" )
[[ "${PRIVACY_AUDIT}" -eq 1 ]] && ARGS+=( --privacy-audit )
# Extra launch.py flags after `--` (security options, see docs/security/aws-baseline.md).
if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
  ARGS+=("${PASSTHROUGH[@]}")
fi
if [[ "${DATA_SOURCE}" == "channel" ]]; then
  if [[ -z "${DATA_S3}" ]]; then
    echo "ERROR: --data-source channel requires --data-s3 S3URI" >&2
    exit 2
  fi
  ARGS+=( --train-s3 "${DATA_S3}" )
  if [[ -n "${MANIFEST_SHA256}" ]]; then
    ARGS+=( --manifest-sha256 "train=${MANIFEST_SHA256}" )
  fi
fi
[[ "${USE_SPOT}" -eq 1 ]] && ARGS+=( --use-spot )
# QUANTIZE=1 only reaches here when --force-quantize bypassed the guard above;
# forward --force-quantize so launch.py lifts the entry-point guard too.
[[ "${QUANTIZE}" -eq 1 ]] && ARGS+=( --force-quantize )

# --------------------------------------------------------------------------- #
# Human-readable GPU for the summary, keyed on the resolved instance type
# (covers common SageMaker GPU families; falls back to "unknown" otherwise).
# --------------------------------------------------------------------------- #
case "${INSTANCE_TYPE}" in
  ml.g4dn.*)        GPU_DESC="NVIDIA T4 16GB" ;;
  ml.g5.*)          GPU_DESC="NVIDIA A10G 24GB" ;;
  ml.g6.*)          GPU_DESC="NVIDIA L4 24GB" ;;
  ml.g6e.*)         GPU_DESC="NVIDIA L40S 48GB" ;;
  ml.p3.*)          GPU_DESC="NVIDIA V100 16GB" ;;
  ml.p4d.*|ml.p4de.*) GPU_DESC="NVIDIA A100 40/80GB" ;;
  ml.p5.*)          GPU_DESC="NVIDIA H100 80GB" ;;
  *)                GPU_DESC="unknown (non-GPU or unrecognized instance)" ;;
esac

# --------------------------------------------------------------------------- #
# Summary + launch
# --------------------------------------------------------------------------- #
cat <<EOF
==========================================================================
 Geneformer fine-tuning job
--------------------------------------------------------------------------
  Account / region : ${ACCOUNT} / ${REGION}   (profile: ${AWS_PROFILE_NAME})
  Preset           : ${PRESET}
  Instance         : ${INSTANCE_TYPE}
  GPU              : ${GPU_DESC}
  Batch size       : ${BATCH_SIZE}
  Epochs           : ${EPOCHS}   LR: ${LR}   freeze: ${FREEZE_LAYERS}
  Eval             : num-crossval-splits=${NUM_CROSSVAL_SPLITS} $( [[ "${NUM_CROSSVAL_SPLITS}" -ge 1 ]] && echo "(held-out eval)" || echo "(train-only)" )
  Fine-tune        : $( [[ "${QUANTIZE}" -eq 1 ]] && echo "QLoRA (4-bit base + LoRA adapters) [FORCED -- unsupported/experimental]" || echo "full precision (all weights)" )
  Data             : ${DATA_SOURCE}${DATA_S3:+  (${DATA_S3})}
  Image            : ${IMAGE_URI}
  Role             : ${ROLE_ARN}
  Job name         : ${JOB_NAME}
==========================================================================
EOF

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] would run:"
  echo "  AWS_PROFILE=${AWS_PROFILE_NAME} ${LAUNCHER_PY} ${ARGS[*]}"
  exit 0
fi

# The SageMaker SDK / boto3 loggers colorize their output (hard to read on some
# terminals). Stream the launch through a sed filter that strips ANSI escape
# sequences so the output is plain text. 2>&1 folds stderr in first so the
# colored log lines (emitted on stderr) are filtered too. PIPESTATUS preserves
# the launcher's real exit code past the pipe.
# Override with --no-strip-color to keep the raw colored stream.
if [[ "${STRIP_COLOR:-1}" -eq 1 ]]; then
  AWS_DEFAULT_REGION="${REGION}" AWS_PROFILE="${AWS_PROFILE_NAME}" \
    "${LAUNCHER_PY}" "${ARGS[@]}" 2>&1 \
    | sed -E 's/\x1b\[[0-9;]*[mGKHF]//g'
  exit "${PIPESTATUS[0]}"
else
  AWS_DEFAULT_REGION="${REGION}" AWS_PROFILE="${AWS_PROFILE_NAME}" \
    "${LAUNCHER_PY}" "${ARGS[@]}"
fi
