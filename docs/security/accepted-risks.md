# Accepted security risks

This file records known vulnerabilities that are accepted rather than fixed, why each
one is accepted, and the conditions that must stay true for that to hold. If one of
those conditions stops being true, the risk is no longer accepted and needs fixing.
Closed entries stay here as a record.

| ID | Subject | Status |
|---|---|---|
| AR-1 | transformers 4.46.3, fsspec 2024.9.0, datasets 3.1.0 (28 CVEs) | **Closed 2026-11-11**: fixed by upgrading |
| AR-2 | Geneformer `pickle.load` of its dictionaries | Open |
| AR-3 | CodeBuild `privilegedMode=true` | Open |

AR-2 and AR-3 are not dependency CVEs, so no scanner ignore file is needed for them. If one is
needed again, add an entry here first, then an `osv-scanner.toml` next to `uv.lock`
with one `[[IgnoredVulns]]` per CVE ID, each with a reason naming the entry and an
`effectiveUntil` on its review date. `tests/test_security_hardening.py` enforces that
pairing whenever the file exists.

## Not covered by the `uv.lock` scan

The scGPT image (`sagemaker/Dockerfile.scgpt`) installs from its own hash-locked
`sagemaker/requirements-scgpt*.txt` files, not `uv.lock`, so the `uv.lock` dependency
scan doesn't see it. Scan those files with OSV-Scanner too, and read the ECR scan report
that the build logs.

Geneformer `pickle.load`s its gene token/median/name dictionaries (`tokenizer.py`,
`classifier.py`, `emb_extractor.py`, ...) and the `id_class_dict` its classifier writes
during a run. No CVE covers this, but it's the same class of risk as the AR-1
deserialization advisories. It's acceptable because the dictionaries come from the
pinned Geneformer commit baked into the image, and `id_class_dict` is written by the
same job that reads it. Don't point a config at Geneformer dictionaries or a model
directory from a source you don't trust.

## AR-1: transformers 4.46.3, fsspec 2024.9.0, datasets 3.1.0 (closed)

| | |
|---|---|
| Owner | Repository maintainers |
| Accepted | 2026-11-10 (dependency scan) |
| Closed | 2026-11-11 |

**Findings:** 1 Critical, 16 High, 10 Medium, 1 Low, all from these three packages in
`uv.lock`.

### How it was closed

`uv.lock` now resolves transformers 5.22.2, datasets 5.2.0, fsspec 2026.9.0 and
huggingface-hub 2.4.0 (floors: transformers ≥5.10, datasets ≥5.0.1, fsspec ≥2026.6.0,
enforced by `tests/test_security_hardening.py`). OSV-Scanner v2.7.1 reports all 28
AR-1 CVEs on the old lock and none on the new one, including the ten transformers
advisories that the original scan listed without a fixed version.

The pinned Geneformer commit still declares `transformers==4.46`, so the image patches
it at build time (`sagemaker/patches/geneformer_transformers5.py`, also run by
`scripts/setup.sh`):

- `collator_for_classification.py` and `pretrainer.py` import `SpecialTokensMixin`
  and three TensorFlow helpers that transformers 5 removed. They now import them from
  `geneformer_transformers5_compat.py`, a port of the 4.46.3 behaviour the two
  pre-collators use.
- Geneformer's default training arguments pass `group_by_length=False`, which
  transformers 5 rejects; it becomes the equivalent `train_sampling_strategy="random"`.

Our own 4.46-only QLoRA workaround (`finetune._skip_quantized_init`) was removed:
transformers 5 initialises only keys missing from the checkpoint and skips every
loaded tensor, which fixes the head-add crash it worked around.

**Verified on CPU** (aarch64, real Geneformer V1-10M weights at the pinned commit,
synthetic 400-cell / 3-class data, 5-fold CV × 3 epochs, three runs per stack): every
Geneformer module imports; tokenizer output is byte-identical to 4.46.3; the same
encoder weights load and the same four head parameters (`bert.pooler.dense`,
`classifier`) are freshly initialised; folds 2–5 reproduce 4.46.3's eval loss to four
decimals at every epoch in all runs; fold 1 varies run to run by the same amount on
both stacks; `scfm-ft run` and `scfm-ft evaluate` complete.

**Verified on GPU** (image `geneformer-ft@sha256:072f0996…`, build 63442c16): the
image builds with the patch applied, and PBMC3k 5-fold matches the 4.46.3 results on
the same folds: V1-10M on a T4, 0.944 ± 0.011 accuracy / 0.910 ± 0.023 macro-F1 vs
0.943 ± 0.008 / 0.905 ± 0.027; V2-104M on an A10G, 0.951 ± 0.010 / 0.935 ± 0.013 vs
0.950 ± 0.008 / 0.933 ± 0.012. See the README's "transformers 5 validation" section.

**Not yet verified:** the experimental QLoRA
path (still behind `allow_quantize`).

### Original assessment (2026-11-10)

Kept for the record; it no longer binds anything.

Geneformer at the pinned commit (`1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5`) requires
`transformers==4.46` and subclasses `transformers.SpecialTokensMixin`, which
transformers 5 removed. datasets was held at `<3.2` to match, and datasets 3.1 caps
`fsspec[http]<=2024.9.0`. The exposure was judged low because almost every CVE needs an
attacker-supplied model repository, `config.json`, checkpoint, dataset folder or
crafted string to reach a specific code path, and this project reaches none of them
with untrusted input:

| CVE(s) | Trigger | Why it didn't apply |
|---|---|---|
| CVE-2026-5241 (Critical), CVE-2026-4372 | Malicious `config.json` loaded with `from_pretrained()`; overrides `trust_remote_code` | Every `from_pretrained()` call loads a local directory (baked weights or our own output); `trust_remote_code` is never set. |
| CVE-2024-11392, CVE-2024-11393, CVE-2024-11394, CVE-2025-14920, CVE-2025-14921, CVE-2025-14924, CVE-2025-14926, CVE-2025-14927, CVE-2025-14928, CVE-2025-14929, CVE-2025-14930 | Deserialization or code injection in model-specific loaders and conversion scripts | Only BERT is used; none of the affected classes or scripts run. |
| CVE-2026-1839 | `Trainer._load_rng_state()` `torch.load` on resume | Training never resumes from a checkpoint. |
| CVE-2026-9856 | `save_pretrained` path traversal via chat-template names | No chat templates. |
| CVE-2025-2099, CVE-2024-12720, CVE-2025-1194, CVE-2025-3263, CVE-2025-3264, CVE-2025-3933, CVE-2025-5197, CVE-2025-6051, CVE-2025-6638, CVE-2025-6921 | ReDoS | Availability only; no untrusted free text reaches transformers. |
| CVE-2025-3777 | Username injection in image utilities | No image processing. |
| CVE-2026-104851 (fsspec) | Template injection in `ReferenceFileSystem` | No `reference://` URLs. |
| CVE-2026-66007 (datasets) | Path traversal in folder-based `load_dataset` builders | `load_dataset` is never called. |

The code-level guards behind that table (no `trust_remote_code=True`,
`resume_from_checkpoint`, `load_dataset(` or `reference://` in our code) are still
tested as general hygiene.

## AR-2: Geneformer pickle dictionaries (open)

| | |
|---|---|
| Owner | Repository maintainers |
| Threat | T003 |

Geneformer loads its token, median and gene-name dictionaries with `pickle.load`, which
runs code embedded in the file. We can't remove that without patching Geneformer.

**Conditions that must stay true:** the dictionaries come from the Geneformer commit
pinned in `sagemaker/Dockerfile` (git-lfs checks the SHA-256 of each large file), no
config points at dictionaries or model directories from another source, and job input
channels carry data only. Input hash checking (on by default) rejects any input file that is not in
the channel's `SHA256SUMS`, and `--network-isolation` limits what a malicious pickle
could send out. Review when the Geneformer pin changes.

## AR-3: CodeBuild privileged mode (open)

| | |
|---|---|
| Owner | Repository maintainers |
| Threat | T008 |

`docker build` inside CodeBuild needs `privilegedMode=true`, which gives the build
container root on its host.

**Conditions that must stay true:** the project builds only the one source zip in the
private source bucket, nobody else can write to that zip, and the service role is no
wider than `docs/iam/codebuild-service-role-policy.json`. To remove the risk, move the
project to the CodeBuild Docker server capability (see `aws-baseline.md`, section 4)
and test that both images still build.

## AR-4: Training containers run as root (open)

| | |
|---|---|
| Owner | Repository maintainers |
| Threat | T006 |

Both images run as root. AWS documents that SageMaker expects containers to run as root
and mounts `/opt/ml` (input data, model output, failure file) root-owned, so a non-root
`USER` is likely to fail writing the model artifacts. This was not tested here: a
non-root image was not built or run.

**Conditions that must stay true:** jobs that read S3 data run under network isolation
(the default; `--regulated-data` enforces it) so code running as root has no way to send
data out, the execution role stays limited to the prefixes in
`docs/iam/sagemaker-execution-policy.json`, and no code is uploaded to the job. Review
if SageMaker documents or tests a supported non-root training setup.
