# Threat mitigations

Status of each threat from the threat model, what changed in this repository, and what
is left for you to do in the AWS account. "Verified" means exercised for real, not only
by unit tests. Account settings are in [`aws-baseline.md`](aws-baseline.md).

The verification records below come from the account this sample was developed in. Its
account ID, KMS key IDs, build IDs and some role names are replaced with placeholders
(for example `111122223333`, the AWS documentation example account).

| ID | Threat | In this repository | Verified | Needs action in your account |
|---|---|---|---|---|
| T001 | scGPT image supply chain | `Dockerfile.scgpt` installs from hash-locked files (180 pinned packages, flash-attn sdist hash, build tools); the uv bootstrap in the Geneformer image is hashed too; build logs an ECR scan report | Lock resolves with hashes (`uv pip install --dry-run`); unit tests on the files | Restage the source zip and rebuild; the new scGPT image has **not** been built |
| T003 | Malicious pickle or checkpoint | `SHA256SUMS` check of every S3 input channel (mandatory by default; `--allow-unverified-inputs` opts out and `--regulated-data` refuses it), `--network-isolation`; Geneformer pickles remain (AR-2) | Unit tests; real `Estimator` accepts the options | Create manifests for your data; run a short job under isolation |
| T004 | Malformed input files | Header-only size, cell-count and gene-count limits before any data is read (defaults 4 GiB, 1,000,000 cells, 100,000 genes; `input:` in the config) | Unit tests with real HDF5 files | None |
| T005 | Tampered S3 objects or ECR image | Launcher pins the image digest; unique immutable build tags; build manifest and scan report in the build log; `SHA256SUMS` pinned by `--manifest-sha256`, which the launcher requires for every S3 input channel (`--allow-unpinned-manifest` opts out and `--regulated-data` refuses it); the wrapper defaults to the newest `b<N>` tag | Digest resolution against the live ECR repo. `geneformer-ft` and `scgpt-ft` are `IMMUTABLE` with scan on push (applied 2026-11-17, read back) | S3 versioning (T007). `latest` is frozen at build b40 |
| T006 | Exfiltration from the job | network isolation on by default for S3 data (`--allow-network` opts out, `--regulated-data` refuses it), `--subnets`, `--security-group-ids`, `--encrypt-inter-container-traffic`; the container still runs as root (AR-4) | Real SDK `Estimator` accepts all of them; **no job has run under isolation or in a VPC** | VPC, subnets, endpoints, role policy |
| T007 | Unprotected S3 data | `--volume-kms-key` (EBS instance types only; local-NVMe types such as g5 are encrypted in hardware and refuse it), `--output-kms-key`; baseline doc; TLS-only bucket policy template; KMS role policy | Real `Estimator` accepts the keys | Create the key; apply the bucket settings |
| T008 | CodeBuild privileged mode | Least-privilege role and trust policy; privileged mode accepted as AR-3 with conditions; Docker server option documented | Applied 2026-11-17: the old wildcard inline policy is deleted, trust is limited to this account and the `geneformer-ft-build` project, and builds b38 and b39 passed on the new policy alone (b39 also with the trust change) | Privileged mode is still on. Not "fixed" |
| T010 | No audit trail | Baseline for CloudTrail data events and S3 access logs; `scripts/check_audit_trail.sh` checks a trail read-only; `docs/security/audit-alerts.yaml` emails on protection changes and on data access outside the job | `scripts/check_audit_trail.sh` passed all 10 checks against trail `scfm-audit` on 2026-11-18 (logging, multi-region, validation, delivery within 24 h with no error, management write events from every source, S3 data events on four buckets) | Alerts deployed as stack `scfm-audit-alerts` (2026-11-18); all eight rules emailed for a live event (the build-project rule after a fix to match the project ARN). Log bucket policy applied 2026-11-18; management events added to the trail 2026-11-18. Object Lock on the log bucket (governance mode, 90 days, 2026-11-18), existing logs locked, bypass denied (checked live). The data-access rule also covers callers with no ARN (deployed and pattern-tested 2026-11-18) and `HeadObject`, `ListObjects` and `ListObjectVersions` (live listings emailed) |
| T012 | Unverified PBMC3k download | Both files downloaded to a temp directory and checked against a pinned size and SHA-256 (the processed file from a commit-pinned URL) before scanpy reads them | Real download and load: 2,638 cells, 32,738 genes | None |

## Verified in the account (2026-11-17)

- **Isolation.** A V1-10M job with `--network-isolation --require-input-hashes` completed:
  `EnableNetworkIsolation=true`, offline environment variables set, `SHA256SUMS` checked
  before the data was read. A copy of the data whose manifest had a wrong hash failed with
  `SHA-256 does not match SHA256SUMS` before tokenizing.
- **CodeBuild role.** Builds ran with only `geneformer-codebuild-least-privilege`, and then
  with the tightened trust policy. The ECR scan report works (`IMAGE_SCAN status=COMPLETE`).
- **OS packages.** `apt-get upgrade` in the Geneformer image took the ECR basic scan from 35
  critical / 185 high to 0 critical / 6 high (`python3-pip` and `git-lfs`, build-time tools).
  V1-10M on PBMC3k, 5-fold on the same folds: 0.940 ± 0.013 accuracy, 0.897 ± 0.022 macro-F1.
- **scGPT image.** Built as `scgpt-ft` tag `b41` (digest `8c028775…`) with every package
  hash-checked. The build's final assertion passed in the image: torch 2.0.1+cu117,
  torchtext 0.15.2, flash-attn imports. ECR scan: 0 critical, 6 high (pip, git-lfs). Building
  it found and fixed a setuptools 82 incompatibility. No GPU job has run on it.
- **ECR.** `geneformer-ft` and `scgpt-ft` are `IMMUTABLE` with scan on push (read back).
  `SCAN_FAIL_ON=CRITICAL` is set on the CodeBuild project (read back); no build has run
  with it yet.
- **S3 baseline** (read back from an admin session): `sagemaker-us-east-1-111122223333`,
  `geneformer-cb-source-111122223333`, `scgpt-weights-111122223333` and the new log bucket
  `scfm-cloudtrail-111122223333` each have versioning enabled, all four Block Public Access
  options on, and a TLS-only bucket policy. All use SSE-S3 default encryption (read on the
  first three before the change; SSE-KMS is not applied to them).
- **Data bucket** (read back in CloudShell, 2026-11-18): `scfm-data-111122223333` has
  versioning enabled, Block Public Access on (BlockPublicAcls, BlockPublicPolicy,
  RestrictPublicBuckets read), a TLS-only bucket policy, and SSE-KMS default encryption with
  the customer-managed key `arn:aws:kms:us-east-1:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab`.
  Not read: that key's policy, whether rotation is on, and whether the SageMaker execution
  role can use the key.
- **Audit trail** (read in CloudShell, 2026-11-17 ~23:25 UTC, with `aws cloudtrail` calls run
  directly). Trail `scfm-audit`: logging is on, the last log delivery was 23:21:48 UTC with no
  delivery error, the digest (log file validation) delivery was 23:13:26 UTC, and it is
  multi-region with log file validation on. Its data-event selectors name four buckets:
  `sagemaker-us-east-1-111122223333`, `geneformer-cb-source-111122223333`,
  `scgpt-weights-111122223333` and `scfm-data-111122223333`. The account's existing
  organization trail was left alone. `scripts/check_audit_trail.sh` (the version
  from `fix/audit-check-no-profile`, run with `-` as the profile) then passed all 9 checks
  with exit code 0: logging, multi-region, log file validation, no delivery error, delivery in
  the last 24 hours, and S3 data events for each of the four buckets. That fix is now on
  `main`. Not covered: alarms, and who can read the log bucket.
- **Execution role without `AmazonSageMakerFullAccess`** (2026-11-18). The role
  `GeneformerSageMakerExecution` had the managed policy attached alongside its inline
  policies, which gave it, among other things, `iam:PassRole` to SageMaker on every role in the
  account and full object access to every bucket named `*sagemaker*`. It was detached after
  adding an inline `scfm-execution-core` (ECR pull for the two repositories, CloudWatch Logs,
  `cloudwatch:PutMetricData`, and output write in `scfm-ft/` of the default bucket). The role now
  has five inline policies and no managed ones (read back). Job
  `geneformer-v1-10m-20261118-011657` (V1-10M, 1 epoch, `geneformer-ft:b43`) then completed on
  that role alone: it pulled the image, read `scfm-data-111122223333/train/smoke/` under SSE-KMS,
  passed `--manifest-sha256` (the first run with a pin) under network isolation, logged to
  CloudWatch, and wrote `model.tar.gz` to `scfm-data-111122223333/output/`.
- **Model output key.** That `model.tar.gz` was encrypted with `alias/aws/s3`
  (`0d40fe5e…`, AWS-managed), not the bucket's customer-managed key, because SageMaker ignores
  the bucket default when no output key is given. The launcher now passes the bucket's default
  key. Job `geneformer-v1-10m-20261118-014304`, launched from `main` at `d0dbe5b` with no
  `--output-kms-key`, was submitted with `OutputDataConfig.KmsKeyId` set to the bucket's key
  `1234abcd…`, and its `model.tar.gz` reads back as `aws:kms` with that key. This needed
  `kms:Encrypt` on the execution role's key grant and `s3:GetEncryptionConfiguration` on the
  data bucket for the launch role, both applied on 2026-11-18.
- **Audit alerts** (2026-11-18). `docs/security/audit-alerts.yaml` was deployed in us-east-1 as stack
  `scfm-audit-alerts` with an email subscription. Two test actions from an admin session each
  produced an alert email: a `GetObject` on `scfm-data-111122223333/train/smoke/SHA256SUMS`
  (rule `scfm-alert-data-accessed-outside-job`) and a `PutBucketVersioning` on
  `scgpt-weights-111122223333` (rule `scfm-alert-bucket-protection-changed`). The trail, data-key,
  ECR and build-project rules are checked only by
  `tests/test_audit_alerts.py`, not by a live event.
- **Role-change alert** (2026-11-18). A `PutRolePolicy` on `GeneformerSageMakerExecution` that wrote
  `ScgptWeightsRead` back unchanged produced an email from rule `scfm-alert-role-permissions-changed`.
  Three of the seven alert rules have now fired on live events.
- **`--regulated-data` run** (2026-11-18). Stack `scfm-training-vpc`
  (`docs/security/training-vpc.yaml`) added two private subnets with no internet route, an S3
  gateway endpoint limited to the data and weights buckets, and an HTTPS-to-S3-only security group to
  the account's default VPC (the account was at its VPC quota). Job
  `geneformer-v1-10m-20261118-030707` (V1-10M, 1 epoch, `ml.g4dn.xlarge`, `geneformer-ft@sha256:72cef917…`)
  then completed with `--regulated-data`. Before submitting, the launcher read versioning, Block Public
  Access, default encryption and the TLS-only policy on `scfm-data-111122223333` (a first attempt was
  refused while the launch role could not read three of them). The job definition shows
  `EnableNetworkIsolation=true`, `EnableInterContainerTrafficEncryption=true`, the two subnets and the
  security group, and `OutputDataConfig.KmsKeyId` set to the customer-managed key; the pinned manifest
  matched before tokenizing; input was read from the SSE-KMS data bucket and `model.tar.gz` written to
  `output/` there. Not read back: the artifact's own key (the launch role cannot read objects there).
- **Manifest pin required by default** (2026-11-18, branch `security/manifest-pin-default`). Through
  `scripts/do-fine-tuning-job.sh` with `--data-s3` on the `train/smoke/` prefix of
  `scfm-data-111122223333` and no `--manifest-sha256`, the launcher stopped with "No
  --manifest-sha256 for the train channel" and exit
  code 1 before submitting; SageMaker has no job under that name. The same command with
  `--manifest-sha256 7a33e1f9…` (no `--regulated-data`) ran job `geneformer-v1-10m-20261118-120555`
  (V1-10M, 1 epoch, `ml.g4dn.xlarge`, `geneformer-ft@sha256:72cef917…`, 300 billable seconds): the
  wrapper forwarded the pin as `train=<hash>`, the launcher printed no unpinned warning, the job ran under
  network isolation with input hashes required and the data key as `OutputDataConfig.KmsKeyId`, and
  logged "1 file(s) match SHA256SUMS" before training. Train loss 1.803, train-only (no held-out split).
- **Wrong manifest pin** (2026-11-18). The same command with `--manifest-sha256` set to 64 zeros ran
  job `geneformer-v1-10m-20261118-121953`, which failed after "[integrity] verifying the train
  channel" with `IntegrityError: SHA256SUMS in /opt/ml/input/data/train does not match the pinned
  hash` (`integrity.py:135`). It did not tokenize or train; SageMaker marked it Failed (exit code 1,
  194 billable seconds, mostly start-up and download), and the wrapper exited 1.
- **Who can write to `train/`** (2026-11-18, reviewed by the account admin in CloudShell). The data
  bucket policy grants nothing (its only statement denies non-TLS requests), and the data key has the
  default key policy (account root `kms:*`, so IAM decides). The IAM policy simulator, run over every
  role and user in 111122223333 for `s3:PutObject` and `s3:DeleteObject` on `train/` and
  `kms:GenerateDataKey` on the data key, found six roles:
  - `Admin`, `SecondAdminRole`, `CloudFormationExecutionRole` and
    `cdk-hnb659fds-cfn-exec-role-…`: put, delete and the data key. Admin-level; covered by the
    data-access and bucket-protection alerts, not by policy.
  - `cdk-hnb659fds-deploy-role-…`: put and delete, not the data key.
  - `geneformer-sm-exec`: put and delete, not the data key. An execution role created 2026-10-14 with
    `AmazonS3FullAccess` and `AmazonSageMakerFullAccess`, trusted by `sagemaker.amazonaws.com` with
    no source conditions, last used 2026-10-14 by job `geneformer-pbmc3k-20261014-170512` and by no
    notebook, domain, pipeline or model. The role-change alert did not watch it. **Deleted 2026-11-18.**

  `GeneformerSageMakerExecution`, `GeneformerLauncher` and the CodeBuild role cannot write
  to `train/`. Still open: a role with `s3:PutObject` but no access to the data key can still upload
  by asking for SSE-S3 (`x-amz-server-side-encryption: AES256`), because the bucket policy does not
  require `aws:kms`; this applies to the CDK deploy role. The simulator does not fully model SCPs or
  session policies, and the trust policies of the remaining roles were not reviewed. The template
  [`s3-bucket-policy.json`](s3-bucket-policy.json) now closes that gap (it also refuses `aws:kms`
  uploads without a key ID, which S3 would put under `aws/s3`).
- **KMS-only uploads on the data bucket** (2026-11-18). The account admin applied that template to
  `scfm-data-111122223333` (TLS rule plus the three upload rules, read back with the launcher's
  role; `bucket_baseline_problems` still reports nothing for `--regulated-data`). The bucket-protection
  alert emailed the `PutBucketPolicy` event. Probe uploads as `Admin`: `AES256` and `aws:kms` without
  a key ID were refused with "explicit deny in a resource-based policy"; an upload with no header and
  one naming the key's ARN were both stored under the data key with a bucket key. The probe versions
  were deleted. So writing to the bucket now needs `kms:GenerateDataKey` on the data key, which only
  the four admin-level roles have among the principals that can put objects; the CDK deploy role's
  SSE-S3 route is closed. Not probed live: an upload naming another key (unit-tested).
- **Who can write to the weights bucket** (2026-11-18, same simulator run against
  `scgpt-weights-111122223333/scgpt/`). The bucket policy only denies non-TLS requests, encryption is
  SSE-S3 (SSE-C blocked) and object ownership is enforced. Five roles can put and delete objects:
  the four admin-level roles and `cdk-hnb659fds-deploy-role-…`; the execution, launch and CodeBuild
  roles cannot. Writes there were logged by the trail but did not alert, so `audit-alerts.yaml` now has
  a `WeightsChanged` rule (`WeightsBucket` parameter) that emails on any write or delete in the
  weights bucket, by any principal. The account admin deployed it the same day
  (`WeightsBucket=scgpt-weights-111122223333`, rule `scfm-alert-weights-changed` `ENABLED`), and a
  probe `PutObject` and `DeleteObject` on `probe/alert-check` each produced an email. Four of the
  eight alert rules have now fired on live events. The pinned `weights` manifest the launcher now
  requires still stops a swapped file from being loaded.
- **Weights bucket under its own KMS key** (2026-11-18, account admin in CloudShell, following
  `aws-baseline.md` "A key for the weights bucket"). Stack `scfm-weights-key`
  (`docs/security/weights-key.yaml`) created key `0987dcba-09fe-87dc-65ba-ab0987654321` (rotation on,
  retained on delete). The execution role got inline policy `ScgptWeightsKms` (`kms:Decrypt` and
  `DescribeKey` on that key, via S3 only), so it now has six inline policies. The bucket's default
  encryption is SSE-KMS with that key and a bucket key (SSE-C still blocked); its four objects were
  copied onto themselves under the key, and `head-object` shows `aws:kms` with the key for each; their
  earlier SSE-S3 versions remain. The KMS-only bucket policy is applied, `scfm-audit-alerts` watches the
  key (`WeightsKeyArn`), and the launch role's `scfm-read-bucket-encryption` now covers this bucket.
  Read back with the launch role, the bucket passes `bucket_baseline_problems`, so `--regulated-data`
  accepts it as a `--scgpt-weights-s3` source. Writing weights now needs the weights key, which the
  CDK deploy role lacks. Found on the way: the bucket holds `scGPT_human/` at its root, not under
  `scgpt/` as the README and `sagemaker-execution-policy.json` assume (the live `ScgptWeightsRead`
  allows reads on the whole bucket), and it has no `SHA256SUMS`, so a `weights` channel would be
  refused until one is written and pinned. No job has read the re-encrypted weights yet.
- **Weights manifest and pin** (2026-11-18, account admin in CloudShell). Each of the four objects
  was streamed out of `scgpt-weights-111122223333` (decrypted with the weights key) through
  `sha256sum`, and `SHA256SUMS` was written at the bucket root in the format `write_sha256sums`
  produces, then uploaded with no encryption header (stored under the weights key; one
  weights-changed email). The stored copy hashes to the pin below, and the repository's parser reads
  all four entries. This file is the record outside the bucket: launch scGPT with
  `--scgpt-weights-s3` pointing at the bucket root and
  `--manifest-sha256 weights=f7e4966dd6057e12ef3143f0cbe4be5975b024561fce9acaf43e47a90a9f1e7c`.

  ```text
  c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60  scGPT_human/args.json
  6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922  scGPT_human/best_model.pt
  6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922  scGPT_human/model.pt
  acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f  scGPT_human/vocab.json
  ```

  The hashes certify the files as they were on 2026-11-18, not that they match the upstream scGPT
  release (no upstream hashes were found to compare). `model.pt` is a copy of `best_model.pt`, as
  the README's upload steps make it. Adding or changing any file in the bucket now needs a new
  manifest and a new pin.
- **Pinned scGPT smoke job** (2026-11-18). Job `scgpt-scgpt-human-20261118-135255` (`--backend scgpt
  --preset scgpt-human`, 1 epoch, `ml.g4dn.xlarge`, `scgpt-ft@sha256:8c028775…` = b41, 235 billable
  seconds), launched with `launch.py` from this branch on the `train/smoke/` data and the weights
  bucket root, each pinned with `--manifest-sha256` (train `7a33e1f9…`, weights `f7e4966d…`). It ran
  under network isolation with input hashes required and the data key as the output key, and logged
  "train: 1 file(s) match SHA256SUMS" and "weights: 4 file(s) match SHA256SUMS" before loading
  anything, then "loaded 159/169 pretrained tensors; 10 left at init (cls_decoder x10)" (the new
  classifier head) and "FlashAttention enabled". It completed. This is the first GPU job on the scGPT
  image, the first live pinned `weights` channel, and the first read of the weights under the weights
  key. Train-only (no held-out split) and the backend logged no loss line, so it shows the pipeline
  runs, not how well it trains; no VPC (not `--regulated-data`). Pins used, recorded here outside the
  buckets: `train/smoke/` in the data bucket
  `train=7a33e1f91f0aed7464aac335fa2c4325c4c5b967d321aeda625be7842faac10c`; the weights bucket root
  `weights=f7e4966dd6057e12ef3143f0cbe4be5975b024561fce9acaf43e47a90a9f1e7c`.
- **scGPT image through the scan gate** (2026-11-18). CodeBuild build 44
  (`geneformer-ft-build:a1b2c3d4-5678-90ab-cdef-EXAMPLE11111`) ran with `BACKEND=scgpt` and
  `IMAGE_REPO_NAME=scgpt-ft` on the staged source zip, whose image inputs (`src/`, `sm_entry.py`,
  the requirement and lock files, `buildspec.yml`, `post_build.py`) are identical to `main`; the two
  Dockerfiles differ only in comments. The log showed `Backend=scgpt ... Repo=scgpt-ft Tag=b44`. It
  pushed `scgpt-ft:gate-b44`, the ECR scan completed with HIGH 6, MEDIUM 2, UNDEFINED 2 and no
  CRITICAL, and only then tagged `b44` (`scgpt-ft@sha256:bdf07814e08975d6858b4a0265d9c4316029c919e4814eced38022b9d3a7a8fd`).
  `geneformer-ft`'s newest image is still `b43`. The project sets `IMAGE_REPO_NAME=geneformer-ft`, so
  the README's earlier command (only `BACKEND=scgpt`) would have pushed this image into
  `geneformer-ft`; the README now sets both.
- **Pinned scGPT smoke job on b44** (2026-11-18). Job `scgpt-scgpt-human-20261118-143359`, launched
  from `main` (`b622374`) with the same settings as the b41 run (scgpt-human, 1 epoch,
  `ml.g4dn.xlarge`, `train/smoke/` and the weights bucket root, both pinned), resolved `scgpt-ft:b44`
  to `scgpt-ft@sha256:bdf07814…` and completed in 225 billable seconds under network isolation, with
  input hashes required and the data key as the output key. Its log matched the b41 run: "train: 1
  file(s)" and "weights: 4 file(s) match SHA256SUMS" before loading, "FlashAttention enabled",
  "loaded 159/169 pretrained tensors; 10 left at init (cls_decoder x10)", then training success.
  Train-only and no loss line, so it shows the gated image runs the pipeline, not how well it trains;
  no VPC (not `--regulated-data`).
- **scGPT with `--regulated-data`** (2026-11-18). Job `scgpt-scgpt-human-20261118-144833`, launched
  from `main` (`7eea150`) with `--regulated-data`, the `scfm-training-vpc` subnets and security group,
  `--output-kms-key` set to the data key, and both channels pinned (train `7a33e1f9…`, weights
  `f7e4966d…`). The launcher's preflight passed for the data and weights buckets (the weights bucket's
  first baseline check under `--regulated-data`) before it submitted. The job definition shows
  `scgpt-ft@sha256:bdf07814…` (b44), `EnableNetworkIsolation=true`,
  `EnableInterContainerTrafficEncryption=true`, the two subnets and the security group, and
  `OutputDataConfig.KmsKeyId` set to the data key. The log showed both manifests matching before
  loading, "loaded 159/169 pretrained tensors; 10 left at init (cls_decoder x10)" and training
  success; the weights came through the S3 gateway endpoint and were decrypted with the weights key.
  227 billable seconds on `ml.g4dn.xlarge`, 1 epoch, train-only (no held-out split, no loss line).
  Not read back: the artifact's own key (the launch role cannot read objects in `output/`).
- **The other four alert rules, and a fix** (2026-11-18). The account admin made a no-op change for
  each: `UpdateTrail` on `scfm-audit` (log file validation, already on), `PutKeyPolicy` writing the
  data key's policy back unchanged, `PutImageScanningConfiguration` on `scgpt-ft` (scan on push,
  already on) and `UpdateProject` on `geneformer-ft-build` (timeout 60, already 60); read-backs
  showed nothing changed. The first three emailed. The fourth did not: `events test-event-pattern`
  against the real CloudTrail event returned False, because CloudTrail recorded
  `requestParameters.name` as the project ARN while `scfm-alert-build-project-changed` listed only
  the name, so a change to the build project (for example removing `SCAN_FAIL_ON`) would not have
  alerted. The rule now matches the name or the ARN (tests use the live event's shape); after
  redeploying, the same `UpdateProject` emailed. All eight alert rules have now fired on live events.
- **Who can read the log bucket** (2026-11-18, account admin in CloudShell, IAM policy simulator
  over every role and user against a log object in `scfm-cloudtrail-111122223333`). The bucket
  policy allows only CloudTrail to write, for trail `scfm-audit` (`aws:SourceArn`), plus TLS-only;
  encryption is SSE-S3 and the trail has no KMS key, so S3 permissions alone decide reads;
  versioning and Block Public Access are on, Object Lock and lifecycle are not, and log file
  validation is on. `Admin`, `SecondAdminRole`, `CloudFormationExecutionRole` and
  the CDK cfn-exec and deploy roles can read, delete and overwrite log files; the CDK lookup role can
  read and list them. None of the project's roles can. Nothing alerts on deleting or overwriting a log
  file (the bucket-protection rule covers bucket settings only). [`log-bucket-policy.json`](log-bucket-policy.json)
  adds denies on deletes, on writes by anything but the trail, and on reads by anyone but the
  listed roles.
- **Log bucket policy applied** (2026-11-18, account admin in CloudShell, readers `Admin` and
  `SecondAdminRole`; the previous policy is kept for rollback). As `Admin`,
  `get-object` on a log file succeeded (1893 bytes); `put-object` and `delete-object` failed with
  AccessDenied, "explicit deny in a resource-based policy". The trail kept delivering: a digest at
  16:13:29 with no delivery error, and after a data-bucket read at 16:48:35 a new log file arrived
  at 16:53:31. Object Lock is still off, so an admin who changes the policy can still delete logs
  (the bucket-protection alert reports the policy change).
- **The trail records no management events** (found during the check above). `scfm-audit` has one
  advanced selector, "scfm S3 object access" (S3 object data events on the SageMaker, code-source,
  weights and data buckets), and no management-event selector. IAM, KMS, bucket, ECR and CodeBuild
  changes are therefore kept only in the 90-day Event history, not in the log bucket. The alerts
  still fire, because EventBridge receives management events without a trail selector.
  `scripts/check_audit_trail.sh` does not check for management events, so it passed 9/9 regardless.
  Fix: add a management-event selector alongside the data selector (an account change).
- **Management events turned on** (2026-11-18, account admin in CloudShell). The current advanced
  selectors were saved to `sel-before.json` for rollback, a selector with `eventCategory` =
  `Management` was appended, and `put-event-selectors` returned both selectors: "scfm S3 object
  access" unchanged and "Management events" (reads and writes, every source). The check script now
  fails a trail without one (`management write events are recorded from every source`). Run from
  `main` in CloudShell after the change, it passed all 10 checks against `scfm-audit` for the four
  buckets, including the management-event check. The check reads the selectors, not the delivered
  logs; see the read-back below.
- **Object Lock plan for the log bucket** (2026-11-18). Decision: governance mode with
  a 90-day default retention, plus a `DenyGovernanceBypass` statement in
  [`log-bucket-policy.json`](log-bucket-policy.json) so that bypassing a lock first needs a policy
  change, which alerts. `audit-alerts.yaml` now alerts on `PutBucketObjectLockConfiguration`; before
  this, an Object Lock change on a protected bucket did not alert. Compliance mode was not chosen
  for now because it cannot be reversed. Steps, checks and rollback are in
  [`log-bucket-object-lock.md`](log-bucket-object-lock.md); turning Object Lock on cannot be undone.
- **Object Lock turned on, and an alert fix** (2026-11-18, account admin in CloudShell). The alert
  stack was redeployed, `DenyGovernanceBypass` was added to the live log bucket policy (read back:
  seven statements; the `PutBucketPolicy` alert emailed), and `put-object-lock-configuration` set
  `GOVERNANCE` with 90 days on `scfm-cloudtrail-111122223333` (read back). That call did not email:
  CloudTrail recorded it as `PutBucketObjectLockConfiguration`, while the rule listed the API name
  `PutObjectLockConfiguration`, so an Object Lock change on a protected bucket would not have
  alerted. The rule now uses the recorded name; after redeploying, the live rule's pattern includes
  it, and writing the same settings back at 17:39:17 delivered a bucket-protection email for
  `PutBucketObjectLockConfiguration` on the log bucket.
- **Object Lock checked** (2026-11-18, CloudShell). All 547 log versions then in the bucket were
  locked with `put-object-retention` (governance, until 2027-02-16 17:44 UTC), with no failures; the
  oldest (2026-11-17 19:56) and a middle version (2026-11-18 08:14) read back with that lock. The
  newest log file, delivered at 17:50:56 after Object Lock went on, read back with `GOVERNANCE`
  until 2027-02-16 17:50 from the 90-day default, so CloudTrail delivers into the locked bucket;
  `get-trail-status` showed a delivery at 17:53:09 with no error. As `Admin`,
  `put-object-retention --bypass-governance-retention` to shorten that lock to one day failed with
  AccessDenied, explicit deny in a resource-based policy (`DenyGovernanceBypass`).
  `check_audit_trail.sh` still passed 10/10. Residual: an admin can still remove
  `DenyGovernanceBypass` and then shorten locks or delete versions; the policy change alerts.
  Compliance mode would close that and cannot be reversed.
- **Management events read back from the log bucket** (2026-11-18, `Admin` in CloudShell). The
  us-east-1 log files delivered after 17:30 UTC, downloaded and filtered on
  `eventCategory` = `Management`, held 457 management events, among them the account admin's
  `PutBucketPolicy` (17:31:19) and both `PutBucketObjectLockConfiguration` calls (17:32:36 and
  17:39:17) on the log bucket. The 17:39:17 event's `eventID` (`0bb6b066-…`) matches the alert
  email, so the event that was emailed is also kept in the locked log bucket.
- **Data-bucket requests with no caller ARN** (2026-11-18). `scfm-alert-data-accessed-outside-job`
  matched `userIdentity.arn` with `anything-but` the execution role, which never matches a missing
  field, and CloudTrail records no `userIdentity.arn` for AWS services, other accounts or anonymous
  callers, so none of those could alert. A baseline over every log file since the trail started
  (2026-11-17 19:56 to 2026-11-18 ~18:00 UTC; the files skipped were digests) found no such request
  on `scfm-data-111122223333`: all of them came from the execution role, `Admin` or
  `GeneformerLauncher`. The rule now has a second `$or` branch on
  `userIdentity.arn` `exists: false`; tests cover an AWS service, another account and an anonymous
  caller, and fail against the old rule. Deployed the same day; `events test-event-pattern` against
  the live rule returned True for an AWS service, an anonymous caller and `Admin`, and False for the
  execution role. No real no-ARN request has reached the bucket, so no such email has been seen.
  `HeadObject` and `ListObjects` were added to the rule afterwards (deployed 2026-11-18; the live rule's pattern lists both), so reading an
  object's metadata or listing the bucket by anyone but the job also alerts; in the logs since
  2026-11-17 that would have meant 4 `HeadObject` emails for `Admin` and none for listings outside
  the job. After the deploy, a `list-objects-v2` on the data bucket as `Admin` from CloudShell delivered a
  data-access email for `ListObjects` (18:43:46 UTC). `ListObjectVersions` was added next (`Admin`
  made one such call in the logs since 2026-11-17); after deploying, the live pattern lists it and a
  `list-object-versions` on the data bucket as `Admin` delivered a data-access email. Still not watched:
  `UploadPart` (its upload's `CompleteMultipartUpload` alerts).
- **Still unverified:** no job has used a
  volume KMS key (g4dn and g5 storage is hardware-encrypted local NVMe).

## Testing the unverified parts

- **Isolation.** A short job with isolation on a
  channel with a `SHA256SUMS` file covers the code-channel behaviour, the offline
  environment and the hash check in one run.
- **scGPT image.** Restage the zip, run the build, and read the `BUILD_MANIFEST` line.
  The build's final assertion checks torch 2.0.1, torchtext 0.15.2 and flash-attn.
- **CodeBuild role.** Apply the policy, run a build, and add any action that is denied.

## T009: launcher permissions and spend

Not part of the original table. [`../iam/sagemaker-launcher-policy.json`](../iam/sagemaker-launcher-policy.json)
scopes `iam:PassRole` to the one execution role and `sagemaker.amazonaws.com`, and denies
unapproved instance types and runs over 12 hours. The budget guidance is in section 6 of
[`aws-baseline.md`](aws-baseline.md). The deny conditions have not been tested against a real
launch, and nothing is applied in the account. Reading SageMaker SDK 2.219.0 and boto3 (2026-11-18)
showed the policy alone could not launch: `default_bucket()`, which `launch.py` calls on every
launch, makes a `HeadBucket` call on the default bucket, which needs `s3:ListBucket` (a denied
`ListBuckets` is ignored by boto3). The policy now allows it. The rollout runbook is [`launcher-and-budget-rollout.md`](launcher-and-budget-rollout.md), with
safe deny tests, and the budget is the CloudFormation template [`budget.yaml`](budget.yaml), which
takes the alert email and monthly limit as parameters so it can be deployed in any account. In this
account the launching role is `GeneformerLauncher`. The launcher policy is not applied.
**What the launching role has** (2026-11-18, read-only, account admin): no managed policies; in
the inline policy `GeneformerLauncherPolicy`, `sagemaker:*` on `*` (Sid `SageMakerLaunch`)
and `iam:PassRole` on `GeneformerSageMakerExecution` and `GeneformerCodeBuildService` with no
`iam:PassedToService` condition; one other PassRole, for `NlDeployServiceRole` to CodeBuild only.
The simulator showed PassRole `allowed` for the execution role and `implicitDeny` for `Admin`, so
PassRole was already narrow. `sagemaker:*` was not: with PassRole on the execution role it allows
processing jobs, batch transforms, endpoints and notebooks as that role, on any instance type and
with no runtime cap, which the launcher policy's training-job denies do not cover. In the 90 days of
Event history, this role's SageMaker calls were `DescribeTrainingJob` (6,289), `ListTrainingJobs`
(181), `CreateTrainingJob` (68), `StopTrainingJob` (5) and one each of `ListPipelines`,
`ListNotebookInstances`, `ListModels` and `ListDomains`, so the runbook now removes `sagemaker:*`
and the unconditioned PassRole on the execution role after attaching the launcher policy, and checks
the result with the policy simulator before any launch. The launcher policy also allowed
`ListTrainingJobs` only on `training-job/*`, which that action does not support, so every
`ListTrainingJobs` call would have been denied; it is now its own statement on `*`.
**Launcher policy applied** (2026-11-18). The account admin attached `ScfmLauncher` (the template
with this account's values) to `GeneformerLauncher`, then removed `SageMakerLaunch`
(`sagemaker:*`) and the execution role from `PassExecAndServiceRoles` in
`GeneformerLauncherPolicy` (diff checked first; `GeneformerCodeBuildService` kept; the
previous document saved for rollback). The policy simulator then gave: `CreateTrainingJob` allowed on
`ml.g4dn.xlarge` for 1 hour, explicitDeny on `ml.p4d.24xlarge` and for 43,260 s;
`CreateProcessingJob`, `CreateTransformJob`, `CreateEndpoint` and `CreateNotebookInstance`
implicitDeny; `ListTrainingJobs`, `DescribeTrainingJob` and `StopTrainingJob` allowed; PassRole on
the execution role allowed to SageMaker and implicitDeny to EC2. Live launches through
`do-fine-tuning-job.sh` with the launching role's profile: `--instance ml.m5.large` and
`--max-run 43260 --allow-long-run` were each refused at `CreateTrainingJob` with AccessDenied,
explicit deny in an identity-based policy, and no job was created; a pinned V1-10M smoke job on
`train/smoke/` (`geneformer-v1-10m-20261118-launcher-ok`, `ml.g4dn.xlarge`, 1 epoch, max run
3,600 s) matched its manifest, ran under network isolation and completed in 335 billable seconds.
Its held-out eval accuracy (0.42 after one epoch on the smoke set, no baseline) only shows the job
ran. Residual: admin roles are not limited by this policy.
**Training-instance quotas** (2026-11-18, read-only, `list-service-quotas`, us-east-1). The three
allowed types, `ml.g4dn.xlarge`, `ml.g5.xlarge` and `ml.g5.2xlarge`, each have a quota of 1
on-demand and 1 spot instance, so the launching role can run at most six jobs at once (three
on-demand, three spot), each capped at 12 hours. Every other GPU training type in the account
(g4dn, g5, g6, g6e, g7, g7e sizes) is also at 1, and there is no p-family quota, so no principal
can start a large multi-GPU job without a quota increase. CPU types have quotas up to 30, which
the launcher policy's instance-type deny blocks for the launching role but not for admins.
**Budget deployed** (2026-11-18, account admin in CloudShell): stack `scfm-budget` from
`budget.yaml` with `MonthlyLimitUsd=500` and the admin's address as `AlertEmail`; read back, the
budget `scfm-sagemaker-monthly` has the three alerts (actual 80%, actual 100%, forecast 100%). No
alert has fired yet. `describe-budget` read back the limit (500.0 USD) and the cost filter
(`Service: Amazon SageMaker`).

## Threats with conditions or account steps

| Threat | In the repository | Still needs you |
|---|---|---|
| T5, T16 (Mitigated 2026-11-18) | The launcher refuses an S3 input channel without `--manifest-sha256`, so by default each channel's `SHA256SUMS` is pinned to a hash recorded outside the bucket and the job refuses a manifest that does not match. `--allow-unpinned-manifest` and `--allow-unverified-inputs` are explicit opt-outs with a launch warning; `--regulated-data` refuses both. The wrapper takes `--manifest-sha256 <hash>` | Record the hash `scfm-ft checksums` prints outside the bucket and pass it at every launch |
| T3, T6 (Accepted 2026-11-18) | `--regulated-data` makes isolation, hash checks, KMS, VPC and pins mandatory; a Geneformer job and an scGPT job (with its weights channel pinned) each ran with all of them | Use `--regulated-data` for human data |
| T1 (Accepted) | scGPT image b41 (digest 8c028775…): hashes verified, torch 2.0.1, torchtext 0.15.2 and flash-attn checked in the image. b44 (digest bdf07814…) was built through the scan gate (no CRITICAL, 6 high, 2 medium), and a pinned GPU smoke job ran on each (2026-11-18). `geneformer-ft` b45 (digest be24cb7d…, 2026-11-18) was built from commit 9d3a194 (the staged S3 zip's object version, whose `commit` metadata the build was pinned to) through the gate: pushed as `gate-b45`, scan complete with no CRITICAL (HIGH 6, MEDIUM 2, UNDEFINED 2), then tagged `b45`. Pinned smoke job `geneformer-v1-10m-20261118-b45-smoke` (`ml.g4dn.xlarge`, 1 epoch) ran on that digest under network isolation, matched its manifest and completed in 295 billable seconds. `scgpt-ft` b46 (digest 2ff6fb8b…, 2026-11-18) was built from the same zip version (commit 9d3a194) with `BACKEND=scgpt` and `IMAGE_REPO_NAME=scgpt-ft` through the gate: the final build step checked torch 2.0.1 and flash-attn 1.0.9, the scan completed with no CRITICAL (HIGH 6, MEDIUM 2, UNDEFINED 2), then it was tagged `b46`. Pinned scGPT smoke job `scgpt-scgpt-human-20261118-214651` (scgpt-human, 1 epoch, train-only, `ml.g4dn.xlarge`, `train/smoke/` and the weights bucket root both pinned, the data key as output key) ran on that digest under network isolation: both channels matched their manifests, FlashAttention was enabled, 159/169 pretrained tensors loaded (10 `cls_decoder` at init; 4 checkpoint tensors unused), and it completed in 230 billable seconds. `geneformer-ft` b47 (digest 059ba7f1…) and `scgpt-ft` b48 (digest 54cdc10c…) were built on 2026-11-18 from commit 4e95b5e (the staged zip's object version, `commit` metadata checked) through the gate, each with no CRITICAL (HIGH 6, MEDIUM 2, UNDEFINED 2); b48's final step checked torch 2.0.1 and flash-attn. Both ran a smoke job with `--privacy-audit` (see T13). No failed-gate images in either repository | Delete any image `scripts/unpromoted_images.sh` lists after a failed build |
| T7 (Accepted 2026-11-18) | Versioning, Block Public Access and TLS-only on all five buckets; SSE-KMS with a customer-managed key on the data bucket, used for input and model output by the `--regulated-data` run; alerts on bucket and data-access changes | Keep human data and its models in the data bucket only |
| T20 (Mitigated 2026-11-18) | Broad managed policy on the execution role: found attached and detached; docs say not to attach it; `audit-alerts.yaml` alerts on any policy change to the project roles, and a live `PutRolePolicy` produced the alert email | Optional: an SCP or permissions boundary to make it preventive |
| T13 (steps 1 and 2 of the plan, 2026-11-18) | Splits were by cell, so one donor's cells landed in both training and held-out data: held-out scores were optimistic for multi-donor data, and a membership-inference test against that held-out set would understate what a model leaks about its donors. `--group-column <donor column>` (`scfm_ft/splits.py`) now assigns whole donors to train, valid or test, for both backends; Geneformer gets the donor lists through its `split_id_dict` (its own `attr_to_split` asserts one label per donor, which cell-type data never has, and its cell 5-fold slices by position, so grouped 5-fold is refused there). Each job writes `<name>_group_split.json`. `--regulated-data` requires a group column unless `--allow-cell-level-split` (warned). Built into `geneformer-ft:b45` (see T1); a smoke job on b45 logged `[split] by cell (no group column)`, which only the new entry point prints. `scgpt-ft:b46` has it too (its smoke job logged the same line). No job has run with a group column yet. Step 2, the membership-inference audit (`--privacy-audit`, `scfm_ft/privacy.py`), is built: after training it scores training and held-out cells with the model and reports the loss-threshold attack's AUC (bootstrap 95% interval), TPR at 1% and 0.1% FPR, advantage, the accuracy and loss gap, and per-donor results with a group split; `--regulated-data` turns it on. Tested on synthetic losses, in the real scGPT fold loop with a stand-in model, and with a fake Geneformer predictor; built into `geneformer-ft:b47` and `scgpt-ft:b48` (see T1) and has run once on each. scGPT job `scgpt-scgpt-human-20261118-b48-audit2` (scgpt-human, 1 epoch, single split by cell, `train/smoke/` and weights pinned, network isolation, data key as output key) scored 2,110 training and 264 held-out cells: AUC 0.519 (95% interval 0.481-0.559), TPR 0.024 at 1% FPR and 0.001 at 0.1% FPR, advantage 0.061, accuracy 0.913 vs 0.886, mean loss 0.327 vs 0.515; the interval covers 0.5, so this attack does not separate members after one epoch, and the single-donor smoke data gives no donor-level result. Geneformer job `geneformer-v1-10m-20261118-b47-audit` (V1-10M, 1 epoch) logged `[privacy-audit] skipped: the Geneformer backend needs classifier.group_column`, as designed. A first launch of the scGPT job, started while the Geneformer job held the account's one `ml.g4dn.xlarge` training instance, was refused with ResourceLimitExceeded (the quota recorded under T9). A fully trained model on multi-donor data has not been audited. A release rule is now agreed (2026-11-18, `aws-baseline.md` section 7): donor-level AUC at most 0.60 and at most 5% of training cells caught at 1% FPR, with a donor split; `scfm-ft privacy-check` applies it to an audit file (exit 1 on failure) and the job log prints the verdict. It is a documented gate, not an automatic block on sharing | Pass `--group-column` for human multi-donor data; run `scfm-ft privacy-check` before sharing a model trained on donor data |

## T017, T018, T019: launcher venv and the code channel

| Threat | In the repository | Verified | Still needs you |
|---|---|---|---|
| T017 launcher venv supply chain | `scripts/do-fine-tuning-job.sh` installs the launcher venv only from `scripts/requirements-launcher.txt` with `--require-hashes` (no `pip install --upgrade pip`, no bare `boto3`). A venv built from a different lock is recreated. Regenerate the lock with the command in `requirements-launcher.in`. | Fresh and stale-venv bootstrap run on a copy of the repo; `launch.py` imports from the locked venv; unit tests on the lock and the script | The first run recreates your existing `.sm-launcher`. Use short-lived credentials for the launcher session |
| T018 code tarball runs as root | `launch.py` passes no `entry_point`, `source_dir`, `dependencies` or `git_config`, so the SDK uploads no code and the job runs the `sm_entry.py` baked into the image (`SAGEMAKER_PROGRAM`). The execution role no longer has read access to a code prefix. | The real SDK 2.219.0 stages nothing and sets no `sagemaker_submit_directory` or `sagemaker_program` (test; run where the SDK is importable). The training toolkit source (5.1.4) reads `SAGEMAKER_PROGRAM` from the environment and defaults the code directory to `/opt/ml/code`. A V1-10M job ran with this change and completed (below). | Delete the old `scfm-ft/code/` objects |
| T019 over-broad upload | Nothing under `sagemaker/` is uploaded any more | Same test | None |

The tarball was already ignored by toolkit 5.1.4 whenever `/opt/ml/code` is not empty (it
is, in both images), but the Geneformer image used to install `sagemaker-training` unpinned
and without hashes, so another toolkit version could behave differently. Not uploading removes
the dependency on that behaviour, and the toolkit is now installed from
`sagemaker/requirements-toolkit.txt` and `requirements-toolkit-build.txt` (hashes required,
versions held to `uv.lock`, source-only packages built with `--no-build-isolation`). Verified:
the combined resolve against `uv.lock` has no conflicts, and `inotify-simple` and
`sagemaker-training` build from their hashed sdists with only the hashed build tools. **The
x86-64 image had not been rebuilt when this was written; see "Verified in CodeBuild and
SageMaker (2026-11-17)" below.**

## M022: the scan gate guards the default image

`sagemaker/buildspec.yml` used to push the `b<N>` tag and then run the scan gate, so a build
that failed the gate (or whose scan never arrived, as with b42) still left a `b<N>` image and
the launcher's default, the highest `b<N>` tag, would pick it. Now the image is pushed as
`gate-b<N>`, scanned, and tagged `b<N>` (and the moving tag) only after `post_build.py`
exits 0; a failed gate stops the phase before any `b<N>` push. `newest_build_tag` ignores
`gate-` tags. Verified with a shell test that runs the post-build commands under `set -e`
with the gate exiting 0, 3 and 4, and in CodeBuild build 43 (below). Images tagged
`b<N>` up to b42 were pushed before the gate and stay as they are. Failed builds leave a
`gate-b<N>` image; `scripts/unpromoted_images.sh` lists them. Do not clean them up with an ECR
lifecycle rule on `gate-` (see `aws-baseline.md`): promoted images carry that tag too.

## Verified in CodeBuild and SageMaker (2026-11-17)

Build 43 of `geneformer-ft-build` (source: `main` at `bac416a`, `SCAN_FAIL_ON=CRITICAL`) and
a SageMaker job on its image, both read back from the account.

- **Build 43 succeeded** (pre-build 15 s, build 258 s, post-build 468 s). The log shows the
  order of M022: `gate-b43` pushed, `post_build.py` run, `IMAGE_SCAN status=COMPLETE` with
  6 high, 2 medium and 2 undefined findings and no critical, then `b43` tagged. ECR holds
  `gate-b43` and `b43` on the same digest (`sha256:72cef917…`). The moving `latest` tag was
  skipped because the repository is immutable.
- **Toolkit locks.** The image build ran `uv pip install --require-hashes
  --no-build-isolation` on `requirements-toolkit.txt` and built `inotify-simple` 1.2.1 and
  `sagemaker-training` 5.1.4 from their hashed sdists. `BUILD_MANIFEST` lists both new lock
  files.
- **A V1-10M job on `b43`** (1 epoch, PBMC3k self-fetch, `ml.g4dn.xlarge`, `max_run` 1800 s,
  launched with the locked launcher venv) completed in 314 s and wrote its model artifact. The
  job definition has no input channels, no `sagemaker_submit_directory` or `sagemaker_program`
  hyperparameter, and the image is pinned by digest. In the job log the toolkit reports
  `module_dir /opt/ml/code`, `user_entry_point sm_entry.py` and "`/opt/ml/code` is not empty,
  abandoning unzipping sourcedir.tar.gz", then runs `sm_entry.py`. So T018 and T019 hold on a
  real job, and the toolkit works from the locked install. Accuracy 0.60 and macro-F1 0.22
  after one epoch are not a quality result.
- **Still not done:** the gate has not failed in CodeBuild (that needs a threshold below the
  image's 6 high findings, for example `SCAN_FAIL_ON=HIGH`); the scGPT image was not built
  and no GPU job ran on it; no job has run in a VPC, with KMS keys, with `--regulated-data` or
  with `--manifest-sha256`; failed-gate images are not cleaned up (none exist yet).
