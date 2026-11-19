# Object Lock on the CloudTrail log bucket

[`log-bucket-policy.json`](log-bucket-policy.json) already denies deletes and overwrites of log
files, but an admin can change that policy and then delete logs (the bucket-protection alert
reports the policy change). S3 Object Lock adds a lock on each log version that a policy change
alone does not remove.

This runbook uses **governance mode with a 90-day default retention**, plus the policy's
`DenyGovernanceBypass` statement:

- In governance mode, a principal with `s3:BypassGovernanceRetention` can shorten a lock or
  delete a locked version by sending the bypass header. `DenyGovernanceBypass` denies that
  permission to everyone, so bypassing a lock first needs a bucket policy change, which alerts.
- Compliance mode would stop admins and root as well, but a compliance lock cannot be shortened
  or removed by anyone, and the bucket cannot be emptied or deleted until every lock expires.
  Consider it once governance mode has run without delivery problems.

**Read this before step 3.** Turning on Object Lock cannot be undone: the bucket keeps Object Lock
and versioning for good. The default retention applies only to log files written after step 3;
step 4 locks the ones already there. Object Lock adds no storage cost here: there is no lifecycle
rule, so logs are already kept, and the policy blocks overwrites, so no extra versions build up.

Run these as an admin in AWS CloudShell, in us-east-1. Set the log bucket's name first:

```bash
LOG_BUCKET=<your-cloudtrail-log-bucket>
```

Replace `<you@example.com>` and the stack
parameters with the values the stack was deployed with.

## 1. Alert on Object Lock changes

`audit-alerts.yaml` now includes `PutBucketObjectLockConfiguration` (the name CloudTrail records for
the `PutObjectLockConfiguration` API) in
`scfm-alert-bucket-protection-changed`. Check that the stack watches the log bucket, then
redeploy it with the template from `main`:

```bash
aws cloudformation describe-stacks --stack-name scfm-audit-alerts \
  --query "Stacks[0].Parameters[?ParameterKey=='ProtectedBuckets'].ParameterValue" --output text
```

The output must include your log bucket. If it does not, add it to
`ProtectedBuckets` when you redeploy (the command is in the template header).

## 2. Apply the updated bucket policy

Keep the current policy for rollback, then add the template's new `DenyGovernanceBypass`
statement to it, so nothing else in the live policy changes:

```bash
aws s3api get-bucket-policy --bucket "$LOG_BUCKET" --query Policy --output text > log-policy-before-lock.json
jq '.Statement += [{"Sid":"DenyGovernanceBypass","Effect":"Deny","Principal":"*","Action":"s3:BypassGovernanceRetention","Resource":"arn:aws:s3:::'"$LOG_BUCKET"'/*"}]' \
  log-policy-before-lock.json > log-policy-lock.json
aws s3api put-bucket-policy --bucket "$LOG_BUCKET" --policy file://log-policy-lock.json
```

This sends one bucket-protection email.

## 3. Turn on Object Lock (cannot be undone)

```bash
aws s3api get-bucket-versioning --bucket "$LOG_BUCKET"   # must show "Status": "Enabled"
aws s3api put-object-lock-configuration --bucket "$LOG_BUCKET" \
  --object-lock-configuration '{"ObjectLockEnabled":"Enabled","Rule":{"DefaultRetention":{"Mode":"GOVERNANCE","Days":90}}}'
aws s3api get-object-lock-configuration --bucket "$LOG_BUCKET"
```

This should send a bucket-protection email for `PutBucketObjectLockConfiguration`, which also tests the
new alert.

## 4. Lock the log files already in the bucket (optional)

Count the versions first; the loop makes one call per version:

```bash
aws s3api list-object-versions --bucket "$LOG_BUCKET" --query 'length(Versions)'
until=$(date -u -d '+90 days' +%Y-%m-%dT%H:%M:%SZ)
aws s3api list-object-versions --bucket "$LOG_BUCKET" \
  --query 'Versions[].[Key,VersionId]' --output text |
while read -r key ver; do
  aws s3api put-object-retention --bucket "$LOG_BUCKET" \
    --key "$key" --version-id "$ver" --retention "Mode=GOVERNANCE,RetainUntilDate=$until"
done
```

## 5. Check

- The two alert emails from steps 2 and 3 arrived.
- After the next delivery (about 5 to 15 minutes), `aws cloudtrail get-trail-status --name scfm-audit`
  shows a newer `LatestDeliveryTime` and no `LatestDeliveryError`.
- `aws s3api head-object` on that new log file (key from `aws s3 ls --recursive`, newest first)
  shows `ObjectLockMode: GOVERNANCE` and an `ObjectLockRetainUntilDate` about 90 days ahead.
- Shortening the lock with the bypass header fails with AccessDenied, explicit deny in a
  resource-based policy:

  ```bash
  aws s3api put-object-retention --bucket "$LOG_BUCKET" --key <that-key> \
    --retention "Mode=GOVERNANCE,RetainUntilDate=$(date -u -d '+1 day' +%Y-%m-%dT%H:%M:%SZ)" \
    --bypass-governance-retention
  ```

- If delivery shows an error instead, remove the default rule (see Rollback) and check again.
- `bash scripts/check_audit_trail.sh scfm-audit us-east-1 - <the four buckets>` still passes 10/10.

## Rollback

Object Lock itself stays on. To stop locking new log files, remove the default rule:

```bash
aws s3api put-object-lock-configuration --bucket "$LOG_BUCKET" \
  --object-lock-configuration '{"ObjectLockEnabled":"Enabled"}'
```

Versions already locked keep their retain-until date. To shorten those, first restore
`log-policy-before-lock.json` (which has no `DenyGovernanceBypass`) and then use
`put-object-retention --bypass-governance-retention`.
