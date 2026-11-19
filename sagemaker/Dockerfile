# Geneformer-ft SageMaker training image.
#
# Bakes in: CUDA runtime, git-lfs-materialized Geneformer repo + weights at
# /Geneformer, and the scfm_ft package. SageMaker runs sm_entry.py (set via
# SAGEMAKER_PROGRAM). CodeBuild normally builds it and tags b<N> only after the ECR scan
# gate; a manual push skips the gate and needs a new tag each time (immutable repos):
#   docker build -f sagemaker/Dockerfile -t geneformer-ft:manual-<stamp> .
#   aws ecr get-login-password --region <r> | docker login --username AWS --password-stdin <acct>.dkr.ecr.<r>.amazonaws.com
#   docker tag geneformer-ft:manual-<stamp> <acct>.dkr.ecr.<r>.amazonaws.com/geneformer-ft:manual-<stamp>
#   docker push <acct>.dkr.ecr.<r>.amazonaws.com/geneformer-ft:manual-<stamp>

# Runs as root on purpose: AWS documents that SageMaker expects training containers to run
# as root, and /opt/ml (input data, model output, failure file) is mounted root-owned, so a
# non-root USER would likely fail writing the model artifacts (not tested here). SageMaker
# remaps container uid 0 to an unprivileged uid on the host instance. The remaining risk is
# AR-4 in docs/security/accepted-risks.md; jobs on S3 data run under network isolation.
# checkov:skip=CKV_DOCKER_3: SageMaker training requires root in-container (see above)

# CUDA 12.1 runtime on Ubuntu 22.04 (matches the torch cu121 wheels). Pulled from
# NVIDIA's registry (nvcr.io), not Docker Hub: CodeBuild's shared IPs exhaust Docker
# Hub's anonymous pull limit (429 at this line). It is the same image -- this digest is
# what docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 also serves -- and the
# digest pin makes every build use exactly it.
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:f4d8e1264366940438f0353da6f289c7bef069d993d111f8106086ccd18c4a30

# A training job is a run-to-completion process, not a service: SageMaker tracks it by
# exit code and never runs Docker health checks. Declared explicitly so it's a choice.
HEALTHCHECK NONE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # SageMaker toolkit entry: the training script to run.
    SAGEMAKER_PROGRAM=sm_entry.py \
    GENEFORMER_REPO=/Geneformer

# Pick up Ubuntu security fixes that landed after the base image was built (the
# scanner's critical findings are openssl, krb5, gnutls, glibc). The NVIDIA packages are
# held so the CUDA, cuDNN and NCCL libraries stay exactly as the base image ships them.
RUN apt-get update \
    && dpkg-query -W -f='${Package}\n' | grep -E '^(cuda-|libcublas|libcufft|libcufile|libcurand|libcusolver|libcusparse|libcudnn|libnccl|libnpp|libnvj|libnvinfer|nvidia-)' | xargs -r apt-mark hold \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-dev python3-pip git git-lfs build-essential \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Clone Geneformer at a pinned commit and materialize its weights (git-lfs) once, at
# build time, so the training job never downloads at runtime and every build gets the
# same code and weights. Keep in sync with scripts/fetch_weights.py and setup.sh.
ARG GENEFORMER_REVISION=1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ctheodoris/Geneformer /Geneformer \
    && cd /Geneformer && git checkout --detach "${GENEFORMER_REVISION}" && git lfs pull

# scfm-ft patch: QLoRA (4-bit) classifier loads crash with
#   NotImplementedError: "normal_kernel_cpu" not implemented for 'Byte'
# because load_model() calls from_pretrained(quantization_config=...) WITHOUT a
# device_map, so the quantized model lands on CPU and the freshly-added
# classifier head cannot be .normal_()-initialized there. Inject device_map={"":0}
# whenever a quantization_config is set, so the load (and head init) run on GPU.
# Applied against the upstream clone because the image re-clones Geneformer rather
# than using the repo's vendor/ copy; see finetune._build_quantize_config.
COPY sagemaker/patches/geneformer_quantize_device_map.py /tmp/patch_geneformer.py
RUN python3 /tmp/patch_geneformer.py /Geneformer/geneformer/perturber_utils.py

# scfm-ft patch: run the pinned Geneformer on transformers 5. Geneformer pins
# transformers==4.46, which carries CVE-2026-5241 and most of the RCE advisories once
# accepted as AR-1; uv.lock now resolves transformers 5.x. This rewrites the four imports 5.x
# removed (SpecialTokensMixin and three TF helpers) to a small compat module, and maps
# the removed TrainingArguments flag group_by_length to train_sampling_strategy. It
# fails the build if Geneformer's code no longer matches what it expects.
COPY sagemaker/patches/geneformer_transformers5.py sagemaker/patches/geneformer_transformers5_compat.py /tmp/
RUN python3 /tmp/geneformer_transformers5.py /Geneformer/geneformer

# Python environment: a pinned uv provides a managed Python 3.11 in /opt/venv and
# installs the dependency set exactly as locked in uv.lock, with hash checking. Python
# 3.11 (not Ubuntu's 3.10) because current pillow and filelock releases need it; the
# lock resolves older, vulnerable versions for 3.10. /opt/venv/bin comes first on PATH,
# so `python3` and SageMaker's `train` entry point both use this environment.
ENV VIRTUAL_ENV=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH="/opt/venv/bin:${PATH}"
COPY sagemaker/requirements-bootstrap.txt /tmp/requirements-bootstrap.txt
RUN pip3 install --require-hashes --no-deps -r /tmp/requirements-bootstrap.txt && uv venv --python 3.11 /opt/venv

# Geneformer itself (code only; its dependencies come from the lock below).
RUN uv pip install --no-deps /Geneformer

# Install the scfm_ft package and its locked dependency stack.
WORKDIR /opt/ml/code
COPY pyproject.toml README.md uv.lock ./
COPY src ./src
RUN uv export --frozen --no-emit-project --format requirements-txt -o /tmp/requirements.txt \
    && uv pip install --require-hashes -r /tmp/requirements.txt \
    && uv pip install --no-deps . \
    && rm /tmp/requirements.txt
# sagemaker-training provides the SageMaker entry-point runtime. It is not in uv.lock (it
# pulls a web stack the project does not use), so it has its own hash-locked files, held to
# the uv.lock versions for anything they share. inotify-simple and sagemaker-training are
# source-only, so they build with --no-build-isolation against the hashed build tools
# instead of fetching an unhashed build environment.
COPY sagemaker/requirements-toolkit-build.txt sagemaker/requirements-toolkit.txt /tmp/locks/
RUN uv pip install --require-hashes --no-deps -r /tmp/locks/requirements-toolkit-build.txt \
    && uv pip install --require-hashes --no-build-isolation -r /tmp/locks/requirements-toolkit.txt

# Entry script (SAGEMAKER_PROGRAM points the toolkit at this).
COPY sagemaker/sm_entry.py /opt/ml/code/sm_entry.py

ENV PATH="/opt/ml/code:${PATH}"
