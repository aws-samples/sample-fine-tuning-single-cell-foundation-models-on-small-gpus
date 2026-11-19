# Running on SageMaker (launch-ready GPU job)

The core logic in `src/scfm_ft/` is deployment-agnostic — it reads a `Config`
and writes to local paths. The `sagemaker/` directory wraps it as a managed
**SageMaker training job** on a GPU instance:

| File | Role |
|---|---|
| `sagemaker/sm_entry.py` | Training entry point — maps SageMaker's env/hyperparameters onto a `Config` and runs tokenize → finetune. |
| `sagemaker/launch.py` | Submits the job via a SageMaker `Estimator`. |
| `sagemaker/Dockerfile` | Training image: CUDA + Geneformer weights (baked at build time) + the `scfm_ft` package. |

## 1. Build and push the training image

The image bakes in Geneformer and its weights so the job never downloads at runtime.
The usual path is the CodeBuild project (README, "Deploy — build & push the training image"), which pushes each build
as `gate-b<N>` and tags it `b<N>` only after the ECR scan passes. A manual push like the one
below skips that gate. The repositories have immutable tags, so give each push a new tag; the
wrapper's default (the newest `b<N>`) ignores it, so pass it with `--image-uri`.

```bash
ACCT=<your-account>; REGION=<your-region>; TAG=manual-$(date -u +%Y%m%d%H%M)
docker build -f sagemaker/Dockerfile -t geneformer-ft:$TAG .
aws ecr create-repository --repository-name geneformer-ft --region "$REGION" \
  --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true || true
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCT.dkr.ecr.$REGION.amazonaws.com"
docker tag geneformer-ft:$TAG "$ACCT.dkr.ecr.$REGION.amazonaws.com/geneformer-ft:$TAG"
docker push "$ACCT.dkr.ecr.$REGION.amazonaws.com/geneformer-ft:$TAG"
```

## 2. Stage the data in S3

Ship either a raw `.loom`/`.h5ad` or a pre-tokenized `.dataset` directory. Tokenizing
once on CPU and uploading the `.dataset` keeps GPU time spent only on fine-tuning.

The prefix must also carry a `SHA256SUMS` listing every file in it. Write it locally,
record the manifest hash that `scfm-ft checksums` prints somewhere outside the bucket,
and upload the directory as it is:

```bash
# Example with the real PBMC3k loader:
uv run scfm-ft load-pbmc3k --out stage/pbmc3k/data.loom

# Or tokenize first, then ship the .dataset:
uv run scfm-ft tokenize --config configs/pbmc3k.yaml
cp -r outputs/tokenized/pbmc3k.dataset stage/pbmc3k/

uv run scfm-ft checksums stage/pbmc3k     # prints "Manifest SHA-256: <hash>"
aws s3 sync stage/pbmc3k/ s3://<bucket>/pbmc3k/
```

## 3. Launch

```bash
python sagemaker/launch.py \
  --image-uri $ACCT.dkr.ecr.$REGION.amazonaws.com/geneformer-ft:<tag> \
  --role arn:aws:iam::$ACCT:role/<SageMakerExecutionRole> \
  --train-s3 s3://<bucket>/pbmc3k/ --manifest-sha256 train=<hash> \
  --instance-type ml.g5.xlarge \
  --preset V2-104M --epochs 10 --batch-size 24 \
  --label-column cell_type --num-crossval-splits 1
```

The launcher refuses an S3 channel without `--manifest-sha256` (pass
`--allow-unpinned-manifest` to accept the manifest in the bucket as it is). `launch.py`
forwards all hyperparameters to `sm_entry.py`. The fine-tuned model is
written to `SM_MODEL_DIR` and SageMaker uploads it to S3 (`estimator.model_data`).

## Instance sizing

| Model | Params | Input size | Instance | GPU |
|---|---|---|---|---|
| V1-10M | 10M | 2048 | `ml.g4dn.xlarge` | T4 16GB |
| V2-104M | 104M | 4096 | `ml.g5.xlarge` | A10G 24GB |
| V2-316M | 316M | 4096 | `ml.p4d.24xlarge` | A100 80GB |

Add `--use-spot` for managed spot instances (cheaper; the launcher sets `max_wait`).

## Notes

- `sm_entry.py` auto-detects the input: a `.dataset` directory in the `train` channel
  is used as-is (tokenization skipped); otherwise the first `.loom`/`.h5ad` is tokenized.
- Unlike the CPU/MPS local path, Geneformer's eval forward pass works here (GPU present),
  so `--num-crossval-splits 1` yields held-out metrics + a confusion matrix.
- A reference MLflow-tracked SageMaker workflow exists at
  `../fine-tune-single-cell-rna-seq-fm/Finetune_Geneformer_SM_Jobs_MLFlow.ipynb`.
