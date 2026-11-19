# AWS security baseline

These are settings in your AWS account, not in this repository, so nothing here is
applied for you. Each section gives the setting, the command or policy to apply it, and
what in this repository relies on it. Replace the `<placeholders>`.

If the data is PHI or otherwise regulated, treat S3 encryption with your own KMS key,
versioning, TLS-only access and the audit logging below as required, not optional.

Treat a fine-tuned model as sensitive as the data it was trained on. A model can memorise
training records, so a `model.tar.gz` from human donor data needs the same bucket protections,
the same KMS key and the same audience as the data. Do not publish one, or share it outside the
group allowed to see the data, unless it passes the release rule in
[Sharing a model](#7-sharing-a-model-trained-on-human-donor-data).

## 1. S3 buckets (data, weights, code, output)

| Setting | Why | Command |
|---|---|---|
| Block Public Access, all four options | Stops a bad ACL or policy from exposing data | `aws s3api put-public-access-block --bucket <bucket> --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true` |
| Versioning | Lets you recover from, and detect, an overwritten object | `aws s3api put-bucket-versioning --bucket <bucket> --versioning-configuration Status=Enabled` |
| Default encryption with a customer-managed KMS key (SSE-KMS, bucket key on) | Key policy and CloudTrail show who decrypted what | `aws s3api put-bucket-encryption --bucket <bucket> --server-side-encryption-configuration file://sse-kms.json` |
| TLS-only bucket policy | Rejects plain-HTTP requests | `aws s3api put-bucket-policy --bucket <bucket> --policy file://policy.json` |
| Object Ownership = bucket owner enforced | Disables ACLs | `aws s3api put-bucket-ownership-controls --bucket <bucket> --ownership-controls 'Rules=[{ObjectOwnership=BucketOwnerEnforced}]'` |

`sse-kms.json`:

```json
{"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "aws:kms", "KMSMasterKeyID": "arn:aws:kms:<region>:<account-id>:key/<key-id>"}, "BucketKeyEnabled": true}]}
```

[`s3-bucket-policy.json`](s3-bucket-policy.json) is a template. Replace `BUCKET_NAME`,
`REGION`, `ACCOUNT_ID` and `KEY_ID` (use the key's full ARN). Its statements deny:

- any request without TLS;
- uploads that ask for anything other than `aws:kms`, such as `AES256` (SSE-S3);
- uploads that name a different KMS key;
- uploads that ask for `aws:kms` without naming a key. S3 would encrypt those with the
  account's AWS-managed `aws/s3` key, which every principal in the account can use through S3.

An upload that sends no encryption header is allowed and gets the bucket's default key, so
plain `aws s3 cp` and `aws s3 sync` keep working. Each of the three upload rules is its own
statement because conditions inside one statement must all match. Together they mean that
anything written to the bucket is encrypted under your key, so only principals allowed to use
that key (`kms:GenerateDataKey`) can write, whatever their S3 permissions. A request that
names the key must give its full ARN; a key ID or alias does not match and is denied. A
training job whose model output goes to this bucket must pass this key
(`--output-kms-key`, or the bucket default the launcher fills in); otherwise SageMaker uses
`aws/s3` and the upload at the end of the job is refused.

### KMS key and the SageMaker role

The training job reads and writes S3 as the execution role, so that role must be allowed
to use the key. Attach
[`../iam/sagemaker-execution-kms-policy.json`](../iam/sagemaker-execution-kms-policy.json)
(one key ARN, `kms:Decrypt`, `Encrypt`, `GenerateDataKey`, `DescribeKey`, `CreateGrant`; AWS
lists `kms:Encrypt` as needed when a key is given for model output) and allow the role in
the key policy as well, unless the key policy delegates to IAM. Then launch with:

```bash
python sagemaker/launch.py ... \
  --volume-kms-key arn:aws:kms:<region>:<account-id>:key/<key-id> \
  --output-kms-key arn:aws:kms:<region>:<account-id>:key/<key-id>
```

`--volume-kms-key` encrypts the training instance's storage volume. It works only on instance
types with EBS storage (for example `ml.p3` or `ml.m5`). Types with local NVMe storage (`ml.g4dn`,
`ml.g5`, `ml.g6`, `ml.p4d` and others) encrypt it in hardware, SageMaker rejects a volume key for
them, and the launcher refuses the flag there. `--output-kms-key`
encrypts the model artifacts written to S3. SageMaker does not use the output bucket's default
encryption for them: without `--output-kms-key` it encrypts `model.tar.gz` with the account's
AWS-managed `aws/s3` key, whose policy you cannot change. The launcher therefore reads the output
bucket's default encryption and, if it names a customer-managed key, passes that key. If it
cannot read the bucket (the role you launch with needs `s3:GetEncryptionConfiguration` on it) it
prints a note and the output falls back to `aws/s3`, so pass `--output-kms-key` explicitly in
that case.

Do not attach `AmazonSageMakerFullAccess` to the execution role. Among other things it allows
`iam:PassRole` on every role in the account to SageMaker and read, write and delete on every
bucket whose name contains `sagemaker`, which undoes the scoping above.

### A key for the weights bucket

The scGPT weights bucket needs the same protection if anyone other than the admins could
otherwise replace weight files, and `--regulated-data --scgpt-weights-s3` refuses a weights
bucket without SSE-KMS. [`weights-key.yaml`](weights-key.yaml) creates a rotated
customer-managed key (alias `alias/scfm-weights`) that is kept if the stack is deleted. Then, in
order:

1. Deploy the key and note `WeightsKeyArn` from the outputs.
2. Let the execution role decrypt with it: put
   [`../iam/sagemaker-execution-weights-kms-policy.json`](../iam/sagemaker-execution-weights-kms-policy.json)
   on the role as an inline policy (`kms:Decrypt` and `DescribeKey` on that key, only through S3).
3. Set the bucket's default encryption to the key, with a bucket key:
   `aws s3api put-bucket-encryption --bucket <weights-bucket> --server-side-encryption-configuration file://sse-kms.json`
   (`sse-kms.json` as above, with the weights key).
4. Re-encrypt what is already there by copying each object onto itself under the key:
   `aws s3 cp --recursive s3://<weights-bucket>/ s3://<weights-bucket>/ --sse aws:kms --sse-kms-key-id <weights-key-arn>`.
   The content does not change, so `SHA256SUMS` and the manifest pin stay valid. Older versions
   stay under SSE-S3 until you delete them.
5. Apply [`s3-bucket-policy.json`](s3-bucket-policy.json) to the weights bucket with the weights key.
6. Pass `WeightsKeyArn` to the `scfm-audit-alerts` stack so changes to the key alert like changes to
   the data key, and give the role you launch with the `CheckBucketBaselineForRegulatedData`
   reads on the weights bucket.

Do step 2 before step 4: once an object is under the new key, a job whose role cannot use the key
cannot read it.

### Verify data integrity

Stage data with a `SHA256SUMS` manifest and let the job check it:

```bash
scfm-ft checksums ./my-data                  # writes ./my-data/SHA256SUMS
# upload ./my-data (including SHA256SUMS) to s3://<bucket>/<prefix>/ with your usual tool
python sagemaker/launch.py ...                          # refuses unlisted or changed files (the default)
```

`scfm-ft checksums ./my-data --verify --strict` re-checks a local copy.

A manifest in the same bucket as the data protects against corruption and partial
overwrites, but not against someone who can write the whole prefix: they can replace a
file and its manifest together. Close that by pinning the manifest. `scfm-ft checksums`
prints the manifest's own SHA-256 when it writes it. Record that value outside the bucket
(your ticket, a repository, a separate bucket) and pass it at launch:

```bash
python sagemaker/launch.py ... --manifest-sha256 train=<64 hex> [weights=<64 hex>]
```

The job refuses to start if a pinned channel's `SHA256SUMS` hashes to anything else, is
missing, or does not list every file. Both are required by default: the launcher refuses an
S3 input channel without a pin, and the job refuses a channel without a manifest.
`--allow-unpinned-manifest` drops the pin (the job then trusts the manifest in the bucket)
and `--allow-unverified-inputs` drops both; the launcher prints a warning for each.

### One flag for regulated data

`--regulated-data` turns on network isolation, refuses `--allow-unverified-inputs` and
`--allow-unpinned-manifest`, and turns on encrypted
inter-container traffic, and refuses to launch unless you also give KMS keys for the volume
and output, a VPC (`--subnets`, `--security-group-ids`), a `--manifest-sha256` for each
input channel (`train`, plus `weights` with `--scgpt-weights-s3`) and a digest-pinnable
image. It lists everything still missing in one message. It does not create the keys, VPC
or endpoints, so those must exist first. One job has run with it (2026-11-18, see
[`threat-mitigations.md`](threat-mitigations.md)).

It also checks every bucket the job reads or writes (the `--train-s3` and
`--scgpt-weights-s3` buckets and the output bucket) against this document before it submits
anything: versioning on, all four Block Public Access settings on, default encryption with
KMS, and a bucket policy that denies requests made without TLS. A check it cannot run (for
example `AccessDenied`) counts as a failure. The launching identity needs the read-only
`CheckBucketBaselineForRegulatedData` statement in
[`../iam/sagemaker-launcher-policy.json`](../iam/sagemaker-launcher-policy.json). The role
used while building this repository was denied all four reads, so it cannot pass the check as
it stands. `--skip-bucket-check` launches anyway and weakens `--regulated-data`; use it only
when you have checked the buckets another way.

## 2. Network isolation and VPC jobs

`launch.py` runs a job that reads S3 data (`--data-source channel`) under network isolation
by default, which blocks all outbound network access from the training container.
`--allow-network` turns that off and `--regulated-data` refuses it. Isolation also sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE`
so libraries fail fast instead of retrying. It needs data from an S3 channel
(`--data-source channel`); `--data-source pbmc3k` downloads at job start, so it runs with
network access and `--network-isolation` is refused for it.
Weights are delivered by SageMaker as an input channel, so they still work. (The launcher
uploads no code archive.) A Geneformer job under isolation completed (see `threat-mitigations.md`);
an scGPT job has not been run under it.

To place the job in your VPC:

```bash
python sagemaker/launch.py ... \
  --subnets subnet-aaaa,subnet-bbbb --security-group-ids sg-cccc \
  --encrypt-inter-container-traffic
```

Both `--subnets` and `--security-group-ids` are required together. The execution role
needs the network-interface permissions in
[`../iam/sagemaker-execution-vpc-policy.json`](../iam/sagemaker-execution-vpc-policy.json).
Those EC2 actions cannot be restricted to a resource, so that file uses `Resource: "*"`,
named `...AcceptedRisk...` like the other wildcard statements.

[`training-vpc.yaml`](training-vpc.yaml) is a CloudFormation template for this: two private
subnets whose route table has no internet or NAT route, a security group that allows HTTPS out
only to the S3 prefix list plus traffic between instances of the same job, and one S3 gateway
endpoint whose policy allows only the data and weights buckets. It creates its own VPC, or, with
`ExistingVpcId` and two free subnet CIDRs, adds the subnets to an existing VPC (useful when the
account is at its VPC quota). In an existing VPC its other subnets stay reachable over the local
route; the security group and network isolation are what keep the job from reaching them. That is all a training job needs: SageMaker pulls the
image from ECR and collects CloudWatch logs and metrics itself, outside your VPC, and S3 decrypts
SSE-KMS objects on the server side ([AWS: training jobs in a VPC](https://docs.aws.amazon.com/sagemaker/latest/dg/train-vpc.html)).
The VPC and a gateway endpoint have no hourly charge. Interface endpoints (ECR, Logs, KMS) are
only needed if code in the container calls those services, which it cannot under network
isolation.

### A `--regulated-data` run

1. Deploy the VPC (us-east-1, admin):

   ```bash
   PL=$(aws ec2 describe-managed-prefix-lists --filters Name=prefix-list-name,Values=com.amazonaws.<region>.s3 --query 'PrefixLists[0].PrefixListId' --output text)
   aws cloudformation deploy --stack-name scfm-training-vpc --template-file docs/security/training-vpc.yaml \
     --parameter-overrides DataBucket=<data-bucket> WeightsBucket=<weights-bucket> S3PrefixListId=$PL
   # or, in an existing VPC: add ExistingVpcId=<vpc-...> SubnetACidr=<free-cidr> SubnetBCidr=<free-cidr>
   aws cloudformation describe-stacks --stack-name scfm-training-vpc --query "Stacks[0].Outputs[?OutputKey=='LauncherArgs'].OutputValue" --output text
   ```

2. Let the role you launch with read the data bucket's settings, which `--regulated-data` checks
   before it launches (the `CheckBucketBaselineForRegulatedData` statement in
   [`../iam/sagemaker-launcher-policy.json`](../iam/sagemaker-launcher-policy.json)):
   `s3:GetBucketVersioning`, `s3:GetBucketPublicAccessBlock`, `s3:GetEncryptionConfiguration` and
   `s3:GetBucketPolicy` on the data bucket (and the weights bucket for scGPT).

3. Launch with the data in the data bucket, its manifest pinned, the output in the same bucket and
   its KMS key given explicitly (`--regulated-data` checks for `--output-kms-key` before the
   launcher reads the bucket's default key):

   ```bash
   python sagemaker/launch.py --image-uri <image> --role <execution-role-arn> --instance-type ml.g4dn.xlarge \
     --train-s3 s3://<data-bucket>/train/<prefix>/ --manifest-sha256 train=<hash> \
     --output-s3 s3://<data-bucket>/output/ --output-kms-key <key-arn> \
     --subnets <SubnetA> <SubnetB> --security-group-ids <SecurityGroup> --regulated-data ...
   ```

   g4dn and g5 instances use local NVMe storage, which is hardware-encrypted, so no
   `--volume-kms-key` is needed there.

## 3. Container images (ECR)

`sagemaker/buildspec.yml` creates new repositories as `IMMUTABLE` with scan on push. The
two existing repositories (`geneformer-ft`, `scgpt-ft`) were created earlier with
mutable tags and no scanning. Change them (existing images are untouched):

```bash
for repo in geneformer-ft scgpt-ft; do
  aws ecr put-image-tag-mutability --repository-name $repo --image-tag-mutability IMMUTABLE
  aws ecr put-image-scanning-configuration --repository-name $repo --image-scanning-configuration scanOnPush=true
done
```

Once immutable, a tag can be pushed only once, so builds push a unique `b<build number>`
tag and no longer move `:latest`. `launch.py` resolves any tagged `--image-uri` to its
`@sha256:` digest before submitting the job and prints it as `Image:`, so a job runs the
exact image you saw even if a tag later moves. Use `--allow-mutable-image` only to skip
that on purpose.

Every build logs one `BUILD_MANIFEST {...}` line (repository, tag, digest, base image,
SHA-256 of each lock file, scan counts). It lives only in the CodeBuild log (the project
has no artifacts), so keep the log group's retention long enough to be useful, or add an
S3 artifact to the project. Scan results are reported but do not fail the build unless
you set `SCAN_FAIL_ON` (for example `CRITICAL`) as a CodeBuild environment variable.
With `SCAN_FAIL_ON` set, a build whose scan has not finished within `SCAN_TIMEOUT` seconds
(default 1200) also fails, so the gate never passes an image it has not seen. If ECR is
slow, re-run the build, or raise `SCAN_TIMEOUT`.

The image is pushed under a staging tag `gate-b<N>` first, so ECR can scan it, and gets its
`b<N>` tag only after the scan step succeeds. The launcher's default image is the highest
`b<N>` tag, so a build that fails the gate leaves a `gate-b<N>` image that nothing picks by
default. Those images pile up. List them with `scripts/unpromoted_images.sh REPO REGION
PROFILE` (read-only; ignore any pushed in the last half hour, whose build may still be
running) and delete the ones you no longer need with `aws ecr batch-delete-image`. Do not
write an ECR lifecycle rule on the `gate-` prefix: a promoted image carries both `gate-b<N>`
and `b<N>`, and ECR expires an image when any one of its tags matches, so that rule would delete
the good images too. If you want automatic cleanup, use a count rule on all images (for example
keep the newest 30), accepting that old images age out. Tags `b<N>` that were pushed before this
change (up to b42) were not gated; pass `--image-uri` with a digest you have checked rather
than relying on the default if one of those is the newest.
The CUDA and torch versions pinned for Geneformer and scGPT will likely always have some
findings with no fix available, so choose the threshold after reading a first report.

## 4. CodeBuild

[`../iam/codebuild-service-role-policy.json`](../iam/codebuild-service-role-policy.json)
and [`codebuild-trust-policy.json`](../iam/codebuild-trust-policy.json) are the target
permissions for the project's service role: logs for this project only, the one source
zip, push to the two repositories, and the ECR login token (the only whole-resource
action, because AWS does not allow scoping it). They were written from what
`buildspec.yml` and `post_build.py` call, **not** compared against the role that exists
today, which could not be read. Apply them, run a build, and add any action the build
reports as denied.

The project builds with `privilegedMode=true` because `docker build` needs a Docker
daemon. That gives the build container root on its host, which is acceptable only
because the build runs code from one private source zip, in one project, with the role
above. See AR-3 in [`accepted-risks.md`](accepted-risks.md). AWS offers a way to build
without privileged mode (the CodeBuild Docker server capability, which needs the project
in a VPC and a security group allowing port 9876 from itself, and builds with
`docker buildx build --load`). It has not been tried here. Rootless builders were not
evaluated.

## 5. Audit logging

Nothing below exists in this repository; set it up in the account.

- **CloudTrail management events** (on by default for 90 days in Event history; create a
  trail to keep them): covers `CreateTrainingJob`, `PutImage`, IAM and KMS calls. With
  advanced event selectors, a trail records management events only if one selector has
  `eventCategory` = `Management`; adding a data-event selector alone drops them. Include both
  selectors in the same `put-event-selectors` call, and check with
  `aws cloudtrail get-event-selectors`; `check_audit_trail.sh` fails a trail without it.
- **CloudTrail data events for S3** on the data, weights, code and output buckets, so
  object reads and writes by the execution role are recorded:

  ```bash
  aws cloudtrail put-event-selectors --trail-name <trail> --advanced-event-selectors '[{"Name":"Management events","FieldSelectors":[{"Field":"eventCategory","Equals":["Management"]}]},{"Name":"scfm S3 objects","FieldSelectors":[{"Field":"eventCategory","Equals":["Data"]},{"Field":"resources.type","Equals":["AWS::S3::Object"]},{"Field":"resources.ARN","StartsWith":["arn:aws:s3:::<bucket>/"]}]}]'
  ```

  Data events are billed per event; scope them to the buckets that hold sensitive data.
- **S3 server access logging** to a separate log bucket (not the data bucket):
  `aws s3api put-bucket-logging --bucket <bucket> --bucket-logging-status file://logging.json`.
- **Check it.** `scripts/check_audit_trail.sh TRAIL REGION PROFILE BUCKET...` (PROFILE `-` uses the current credentials, as in CloudShell) is read-only and
  prints PASS or FAIL for: logging on, multi-region, log file validation, a delivery in the
  last 24 hours with no error, management write events from every source, and S3 read and write
  data events for each bucket you name.
  It does not check the alerts, and it needs `cloudtrail:GetTrailStatus`, `GetEventSelectors` and
  `DescribeTrails`.
- **Alerts.** [`audit-alerts.yaml`](audit-alerts.yaml) is a CloudFormation template for
  EventBridge rules that email an SNS topic when any of these happen: a protected bucket's policy,
  encryption, versioning, public access, ACL, lifecycle, replication or Object Lock settings change or the bucket is
  deleted; an object in the data bucket is read, written, deleted or has its metadata read
  (`HeadObject`), or the bucket's objects or versions are listed (`ListObjects`,
  `ListObjectVersions`), by anyone other than the execution role
  (browsing the bucket in the S3 console as an admin therefore emails); an object in the weights bucket is written or deleted by anyone (pass
  `WeightsBucket`); a CloudTrail trail is stopped, deleted or reconfigured; the data key is
  disabled, scheduled for deletion, has rotation turned off or its policy changes; a watched IAM
  role's policies or trust policy change; an image repository's immutability, scanning, policy or
  images change; or the build project changes. Deploy it in us-east-1 (IAM events arrive only
  there) with the command in the template header, then confirm the subscription email. The
  data-access and weights rules need the trail's S3 data events on those buckets. The data-access
  rule also alerts on requests CloudTrail records without a `userIdentity.arn` (AWS services,
  other accounts and anonymous callers), which `anything-but` alone would miss. The topic is not
  KMS-encrypted (see the template header).
- **CloudTrail log file validation and a log bucket with versioning** (and S3 Object Lock
  if you must show the logs weren't altered). [`log-bucket-policy.json`](log-bucket-policy.json)
  keeps the standard CloudTrail grants and TLS-only rule, and adds four denies: no one deletes a
  log file or version; no one bypasses a governance-mode Object Lock (`DenyGovernanceBypass`);
  no one but the trail (`aws:SourceArn`) writes there, so a log cannot be replaced; and only the
  roles you list in `DenyLogReadsExceptAdmins` read log files. Replace
  `LOG_BUCKET`, `REGION`, `ACCOUNT_ID`, `TRAIL_NAME` and the reader role(s). An admin can still
  change the policy, which the bucket-protection alert reports. Check delivery after applying
  (`aws cloudtrail get-trail-status`: `LatestDeliveryTime` advances, `LatestDeliveryError` empty).
  [`log-bucket-object-lock.md`](log-bucket-object-lock.md) is the runbook for turning on Object
  Lock in governance mode with 90-day retention, which keeps logs even if that policy is changed.
  It cannot be undone.
- **KMS**: key use appears in CloudTrail automatically, which gives a second record of
  every object decrypted.
- **SageMaker**: training job logs go to CloudWatch Logs (`/aws/sagemaker/TrainingJobs`);
  set a retention period on that log group.
- **ECR**: enable scan on push (section 3) and EventBridge rules on `ECR Image Scan` if
  you want alerts.

## 6. Who can launch jobs, and what they can spend

A principal that can call `sagemaker:CreateTrainingJob` and `iam:PassRole` can run any
container as the execution role, so it can read everything that role can read and bill
GPU hours. `launch.py` caps `--max-run` at 12 hours (the same limit the launcher policy
below enforces; `--allow-long-run` lifts it for people who have not applied that policy),
but nothing in this repository limits who may launch.

**Restrict `iam:PassRole`.** Give the people (or CI role) that run `launch.py` the
policy in [`../iam/sagemaker-launcher-policy.json`](../iam/sagemaker-launcher-policy.json).
It allows `iam:PassRole` for the one execution role only, and only to
`sagemaker.amazonaws.com` (`iam:PassedToService`). Without that scoping, `PassRole` on
`*` lets the holder run a job as any role that SageMaker can assume, including more
privileged ones. The policy also:

- limits training-job actions to this account and region, and log reads to the SageMaker
  training log group;
- allows `ecr:DescribeImages` on the two repositories, which `launch.py` needs to
  resolve a tag to a digest, and `s3:ListBucket` on the SageMaker default bucket, which the SDK's
  `default_bucket()` needs for its `HeadBucket` call on every launch;
- denies jobs on instance types other than `ml.g4dn.xlarge`, `ml.g5.xlarge` and
  `ml.g5.2xlarge` (`sagemaker:InstanceTypes`), and jobs with a maximum runtime over
  12 hours (`sagemaker:MaxRuntimeInSeconds`). Edit both lists to match your presets.

The two denies use SageMaker condition keys that AWS lists for `CreateTrainingJob`. They
have not been tested against a real request here. Try one allowed and one denied launch
(for example `--instance-type ml.p4d.24xlarge`) after attaching the policy.

To make the isolation, encryption and VPC options mandatory instead of opt-in, add
deny statements for the missing option. They block any job that does not use it, so add
them only when every run can meet them (`--data-source pbmc3k` cannot run under
isolation):

```json
{
  "Sid": "DenyJobsWithoutNetworkIsolation",
  "Effect": "Deny",
  "Action": "sagemaker:CreateTrainingJob",
  "Resource": "*",
  "Condition": {"Bool": {"sagemaker:NetworkIsolation": "false"}}
}
```

`sagemaker:NetworkIsolation`, `sagemaker:VolumeKmsKey`, `sagemaker:OutputKmsKey`,
`sagemaker:InterContainerTrafficEncryption`, `sagemaker:VpcSubnets` and
`sagemaker:VpcSecurityGroupIds` are the matching keys. When a request leaves the
parameter out, the key may be absent rather than `false`, so pair the `Bool` test with a
`Null` test (`{"Null": {"sagemaker:NetworkIsolation": "true"}}`) in a second statement.

[`launcher-and-budget-rollout.md`](launcher-and-budget-rollout.md) is the runbook for attaching
the policy, testing the denies safely, and creating the budget from [`budget.yaml`](budget.yaml).

**Budget.** Training cost is the main spend. Set these in the account (they need billing
access, which `launch.py` does not):

- An AWS Budget on the SageMaker service. [`budget.yaml`](budget.yaml) is a CloudFormation
  template for one, filtered to `Service = Amazon SageMaker`, that emails at 80% and 100% of
  actual spend and at 100% of forecast. The address and the limit are parameters:
  `aws cloudformation deploy --stack-name scfm-budget --template-file docs/security/budget.yaml
  --parameter-overrides AlertEmail=<you@example.com> MonthlyLimitUsd=<limit>`.
- A Cost Anomaly Detection monitor on the same service, to catch a job that runs far
  longer than expected.
- A SageMaker service quota for the GPU instance types you use. Keep it low (for example
  one instance) so a mistaken or malicious launch cannot start many. The role used
  while building this repository could not read quotas, so check them from an admin role.
- A shorter `--max-run` for routine jobs (for example 4 hours). The 12-hour default is
  the ceiling, not an estimate.

## 7. Sharing a model trained on human donor data

A model trained on human donor data stays with the people allowed to see that data unless its
membership-inference audit passes this rule (agreed 2026-11-18):

1. **Train with a donor split and the audit on**: `--group-column <donor column>` and
   `--privacy-audit`. `--regulated-data` turns the audit on and requires the group column.
   Without a donor split, held-out cells can share donors with training cells and the audit
   understates what the model leaks about its donors, so the rule fails.
2. **Check the audit**: download `<name>_privacy_audit.json` from the job's output and run
   `uv run scfm-ft privacy-check <name>_privacy_audit.json`. It exits 1 unless:
   - the **donor-level AUC is at most 0.60** (0.5 means the attack cannot tell training donors
     from held-out donors), and
   - **at most 5% of training cells are caught at a 1% false-positive rate**.

   Under cross-validation every fold must pass. The job log shows the same verdict as
   `[privacy-audit] release rule: PASS` or `FAIL`. With fewer than 5 held-out donors the
   donor-level AUC moves in large steps; the check warns, and a pass on so few donors is weak.
3. **Share predictions rather than weights where you can.** Predictions on new data carry less
   than the weights do.
4. **Record the decision** with the audit file and the job name.

The audit is a lower bound: it runs the simple loss-threshold attack, and stronger attacks (for
example LiRA, which trains shadow models) can do better. Passing it is necessary, not
sufficient; it does not replace your organisation's privacy or data-use review. The limits are
in `src/scfm_ft/privacy.py` (`RELEASE_GROUP_AUC_MAX`, `RELEASE_CELL_TPR_AT_1PCT_FPR_MAX`).
