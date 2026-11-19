# Rolling out the launcher policy and a budget (T009, T011)

[`../iam/sagemaker-launcher-policy.json`](../iam/sagemaker-launcher-policy.json) limits who can
launch training jobs and what they can launch; [`budget.yaml`](budget.yaml) is a CloudFormation
template for a monthly budget that emails before SageMaker spend passes a limit. Run the account
steps as an admin in AWS CloudShell, from the repository root.

## 0. Set your values

Every command below uses these. Set them for your own account:

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
LAUNCH_ROLE=<role-that-runs-launch.py>       # the role your launches use
EXEC_ROLE=GeneformerSageMakerExecution       # the SageMaker execution role
DATA_BUCKET=<data-bucket>
ALERT_EMAIL=<you@example.com>                # receives the budget alerts
MONTHLY_LIMIT_USD=<limit>                    # monthly SageMaker spend to alert against
```

## What the policy does

- `iam:PassRole` only for the execution role, and only to `sagemaker.amazonaws.com`.
- Training-job actions in this account and region, log reads on the SageMaker training log group,
  `ecr:DescribeImages` on the two repositories, the bucket reads `--regulated-data` makes, and
  `s3:ListBucket` on `sagemaker-$REGION-$ACCOUNT`. The SageMaker SDK's `default_bucket()`, which
  `launch.py` calls on every launch, makes a `HeadBucket` call that needs it.
- Denies `CreateTrainingJob` on any instance type other than `ml.g4dn.xlarge`, `ml.g5.xlarge` and
  `ml.g5.2xlarge`, and any job with a maximum runtime over 12 hours.

The Allow statements add to whatever the principal already has; only the Deny statements remove
anything. So the PassRole limit only holds if the principal has no broader `iam:PassRole`
elsewhere. Step 1 finds out. Admin roles are not limited by this policy.

## 1. See what the launching role has now (read-only)

```bash
aws iam list-attached-role-policies --role-name "$LAUNCH_ROLE" --query 'AttachedPolicies[].PolicyName' --output text
for p in $(aws iam list-role-policies --role-name "$LAUNCH_ROLE" --query PolicyNames --output text); do
  echo "== $p"
  aws iam get-role-policy --role-name "$LAUNCH_ROLE" --policy-name "$p" --query PolicyDocument --output json |
    jq -c '.Statement[] | select((.Action|tostring)|test("sagemaker|PassRole|iam:\\*|^\\*$|\"\\*\""))'
done
aws iam simulate-principal-policy --policy-source-arn "arn:aws:iam::$ACCOUNT:role/$LAUNCH_ROLE" \
  --action-names iam:PassRole \
  --resource-arns "arn:aws:iam::$ACCOUNT:role/$EXEC_ROLE" "arn:aws:iam::$ACCOUNT:role/Admin" \
  --context-entries ContextKeyName=iam:PassedToService,ContextKeyValues=sagemaker.amazonaws.com,ContextKeyType=string \
  --query 'EvaluationResults[].ResourceSpecificResults[].[EvalResourceName,EvalResourceDecision]' --output text
# Which SageMaker calls the role has made in the last 90 days (CloudTrail Event history):
aws cloudtrail lookup-events --lookup-attributes AttributeKey=EventSource,AttributeValue=sagemaker.amazonaws.com \
  --start-time "$(date -u -d '-90 days' +%Y-%m-%dT%H:%M:%SZ)" --query 'Events[].CloudTrailEvent' --output json |
  jq -r --arg r "$LAUNCH_ROLE" '.[] | fromjson | select(.userIdentity.sessionContext.sessionIssuer.userName==$r) | .eventName' |
  sort | uniq -c | sort -rn
```

Look for two things:

- **Broader PassRole.** Replace `Admin` with any other privileged role in your account. If PassRole
  on it comes back `allowed`, a broader grant exists.
- **Broad SageMaker access** such as `sagemaker:*`. The launcher policy only denies
  `CreateTrainingJob` on other instance types and long runs, so `sagemaker:*` plus PassRole on the
  execution role still lets the holder start processing jobs, batch transforms, endpoints or
  notebooks as the execution role, on any instance type and for any length of time. If the 90-day
  usage shows only training-job calls (`CreateTrainingJob`, `DescribeTrainingJob`,
  `ListTrainingJobs`, `StopTrainingJob`), the broad grant can go. Keep anything else the usage
  shows another user of the role relies on.

## 2. Attach the policy

```bash
sed -e "s/ACCOUNT_ID/$ACCOUNT/g" -e "s/REGION/$REGION/g" \
    -e "s/EXECUTION_ROLE_NAME/$EXEC_ROLE/g" -e "s/DATA_BUCKET/$DATA_BUCKET/g" \
    docs/iam/sagemaker-launcher-policy.json > launcher-policy.json
grep -c 'ACCOUNT_ID\|REGION\|EXECUTION_ROLE_NAME\|DATA_BUCKET' launcher-policy.json   # must print 0
aws iam put-role-policy --role-name "$LAUNCH_ROLE" --policy-name ScfmLauncher \
  --policy-document file://launcher-policy.json
```

Attach it before step 3, so launching keeps working while the broad grants are removed. If the
launching role is in the alert stack's `WatchedRoleNames`, this sends a `RolePermissionsChanged`
email.

## 3. Remove broader grants (only if step 1 found any)

Save the inline policy that holds them, then edit a copy. For a statement with Sid `<broad-sid>`
granting `sagemaker:*`, and the execution role listed in a PassRole statement `<passrole-sid>`
without an `iam:PassedToService` condition (the launcher policy now passes it, with the condition):

```bash
POLICY=<inline-policy-name>
aws iam get-role-policy --role-name "$LAUNCH_ROLE" --policy-name "$POLICY" --query PolicyDocument --output json > "$POLICY-before.json"
jq --arg exec "arn:aws:iam::$ACCOUNT:role/$EXEC_ROLE" '
  del(.Statement[] | select(.Sid=="<broad-sid>"))
  | (.Statement[] | select(.Sid=="<passrole-sid>") | .Resource) |= ([.] | flatten | map(select(. != $exec)))
  | del(.Statement[] | select(.Sid=="<passrole-sid>" and (.Resource | length) == 0))' \
  "$POLICY-before.json" > "$POLICY-after.json"
diff <(jq -S . "$POLICY-before.json") <(jq -S . "$POLICY-after.json")
aws iam put-role-policy --role-name "$LAUNCH_ROLE" --policy-name "$POLICY" --policy-document "file://$POLICY-after.json"
```

Check the `diff` shows only those two changes before the `put-role-policy`.

## 4. Check with the policy simulator, then test with real launches

The simulator evaluates the role's policies without starting anything:

```bash
sim() { aws iam simulate-principal-policy --policy-source-arn "arn:aws:iam::$ACCOUNT:role/$LAUNCH_ROLE" "$@" \
  --query 'EvaluationResults[].[EvalActionName,EvalDecision]' --output text; }
job="arn:aws:sagemaker:$REGION:$ACCOUNT:training-job/sim-test"
sim --action-names sagemaker:CreateTrainingJob --resource-arns "$job" --context-entries \
  ContextKeyName=sagemaker:InstanceTypes,ContextKeyValues=ml.g4dn.xlarge,ContextKeyType=stringList \
  ContextKeyName=sagemaker:MaxRuntimeInSeconds,ContextKeyValues=3600,ContextKeyType=numeric    # allowed
sim --action-names sagemaker:CreateTrainingJob --resource-arns "$job" --context-entries \
  ContextKeyName=sagemaker:InstanceTypes,ContextKeyValues=ml.p4d.24xlarge,ContextKeyType=stringList \
  ContextKeyName=sagemaker:MaxRuntimeInSeconds,ContextKeyValues=3600,ContextKeyType=numeric    # explicitDeny
sim --action-names sagemaker:CreateTrainingJob --resource-arns "$job" --context-entries \
  ContextKeyName=sagemaker:InstanceTypes,ContextKeyValues=ml.g4dn.xlarge,ContextKeyType=stringList \
  ContextKeyName=sagemaker:MaxRuntimeInSeconds,ContextKeyValues=43260,ContextKeyType=numeric   # explicitDeny
sim --action-names sagemaker:CreateProcessingJob sagemaker:CreateTransformJob sagemaker:CreateEndpoint \
  sagemaker:CreateNotebookInstance                                                              # implicitDeny
sim --action-names sagemaker:ListTrainingJobs                                                   # allowed
sim --action-names iam:PassRole --resource-arns "arn:aws:iam::$ACCOUNT:role/$EXEC_ROLE" --context-entries \
  ContextKeyName=iam:PassedToService,ContextKeyValues=ec2.amazonaws.com,ContextKeyType=string  # implicitDeny
```

Then, as the launching role:

- **Allowed:** a pinned smoke job completes:
  `scripts/do-fine-tuning-job.sh --preset V1-10M --data-s3 s3://<data-bucket>/<smoke-prefix>/ --manifest-sha256 <pin> -- --max-run 1800`
- **Instance type denied:** the same with `--instance ml.m5.large` fails at `CreateTrainingJob`
  with AccessDenied, explicit deny in an identity-based policy. Use a cheap CPU type for this
  test: if the deny did not work, the job that starts costs cents and fails fast. Do not test with
  a large GPU type.
- **Runtime denied:** the same with `-- --max-run 43260 --allow-long-run` fails with AccessDenied.
  If a job is created instead, stop it at once with `aws sagemaker stop-training-job`.

## 5. Budget

```bash
aws cloudformation deploy --stack-name scfm-budget --template-file docs/security/budget.yaml \
  --region "$REGION" --parameter-overrides AlertEmail="$ALERT_EMAIL" MonthlyLimitUsd="$MONTHLY_LIMIT_USD"
aws budgets describe-budget --account-id "$ACCOUNT" --budget-name scfm-sagemaker-monthly \
  --query 'Budget.[BudgetLimit,CostFilters,CalculatedSpend.ActualSpend]'
aws budgets describe-notifications-for-budget --account-id "$ACCOUNT" --budget-name scfm-sagemaker-monthly \
  --query 'Notifications[].[NotificationType,Threshold]' --output text
```

It emails `ALERT_EMAIL` at 80% and 100% of actual spend and when the month's forecast passes
100%; there is no subscription to confirm. To change the address or the limit later, deploy again
with new values. Budgets update a few times a day, so they catch a runaway month, not a single
job. The 12-hour runtime cap is what bounds one job. If a budget with the same name already
exists (for example one made by hand), delete it first or pass `BudgetName=<other>`.

## 6. Instance quotas (read-only)

```bash
aws service-quotas list-service-quotas --service-code sagemaker \
  --query "Quotas[?contains(QuotaName,'for training job usage') && Value>\`0\`].[QuotaName,Value]" --output text
```

This lists the training instance types the account can run, and how many at once. You can request
increases but cannot lower a default yourself, so this shows the worst case: how many jobs a
launcher could start in parallel within the allowed types.

## Rollback

`aws iam delete-role-policy --role-name "$LAUNCH_ROLE" --policy-name ScfmLauncher` removes the
policy. Restore anything removed in step 3 with `put-role-policy` from `$POLICY-before.json`.
`aws cloudformation delete-stack --stack-name scfm-budget` removes the budget.
