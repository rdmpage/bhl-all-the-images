# AWS half — embed all of BHL, export only the vectors

Transient, spot-priced compute in **us-east-2** (same region as
`s3://bhl-open-data`, so image reads are free). Output is ~65–130 GB of vectors
written to your own S3 bucket — that's the *only* thing that leaves AWS.

## Source: read the webp derivatives, not the JP2 (`--source webp`)

The bucket has a `web/` folder of webp derivatives
(`web/<barcode>/<barcode>_<seq>_<size>.webp`, sizes thumb/small/medium/large/
full) parallel to the archival `images/` JP2. **Embedding is decode-bound**, and
webp decodes ~20–40× faster than JPEG-2000, so `embed_s3.py --source webp`
(default `--webp-size medium`) is the cost path — it turns a ~$470 full-corpus
run into roughly $25–50 on Spot. `_medium` (~465px) measured ~JP2-equivalent
retrieval (p@5 0.60 vs 0.63); `_small` is lossy (0.53), avoid it. Missing
derivatives transparently fall back to the JP2. CLIP downsamples to 224px, so
the JP2's extra resolution is wasted regardless.

## 0. Confirm the bucket key layout (one-time)

```bash
aws s3 ls --no-sign-request s3://bhl-open-data/
python manifest_s3.py --sample 40        # eyeball real keys
```

If the keys don't match the `KEY_RE` in `manifest_s3.py`, adjust `parse_key()`.
This is the only place the layout is assumed.

## 1. Build + stage the manifest

```bash
pip install boto3
python manifest_s3.py --shards 1000 --out manifest/   # ~63k LIST calls, a few cents
aws s3 sync manifest/ s3://MY-BUCKET/manifest/
```

Each shard is ~63k pages. 1000 shards → ~7h of work split 1000 ways.

## 2. Build + push the worker image

```bash
docker build -t bhl-embed aws/
# tag + push to ECR, then point the Batch job definition at it
```

## 3. Run as a Batch array job

Job definition: the `bhl-embed` image, ~2 vCPU / 4 GB per task, command
overridden to read from S3. Submit one task per shard:

```bash
aws batch submit-job \
  --job-name bhl-embed \
  --job-queue  MY-SPOT-QUEUE \
  --job-definition bhl-embed \
  --array-properties size=1000 \
  --container-overrides 'command=["--manifest-dir","s3://MY-BUCKET/manifest","--out","s3://MY-BUCKET/out"]'
```

Each task resolves its shard from `$AWS_BATCH_JOB_ARRAY_INDEX`, writes
`out/shard-NNNNN.parquet`, and **skips itself if that output already exists** —
so spot reclaim just means the task reruns cleanly.

Use a **Spot** compute environment. The work is decode-bound, so cheap CPU
instances (c7i/c6i family) are the sweet spot; no GPU needed for ViT-B/32.

## 4. Hand off to the Hetzner half

```bash
aws s3 sync s3://MY-BUCKET/out/ ./out/    # ~65–130 GB, within the free egress tier
```

Then load `./out/` with `../hetzner/load_parquet.py`.

## Knobs (env vars)

| var | default | meaning |
|---|---|---|
| `BHL_CLIP_MODEL` / `BHL_CLIP_PRETRAINED` | `ViT-B-32` / `laion2b_s34b_b79k` | model (must match search side) |
| `BHL_EMBED_DIM` | `512` | vector dim for the Parquet schema |
| `BHL_TARGET_PX` | `224` | reduced-decode target; raise to ~336 for ViT-L |
| `BHL_S3_BUCKET` / `BHL_S3_REGION` | `bhl-open-data` / `us-east-2` | source bucket |
