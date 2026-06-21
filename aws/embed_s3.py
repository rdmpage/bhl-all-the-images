"""
AWS-side worker: BHL JP2 page images -> OpenCLIP vectors -> Parquet.

Runs as one task of a Batch array job, in us-east-2, next to the public
`bhl-open-data` bucket so the image reads are in-region (free). For its shard
of the manifest it streams each JP2 from S3, decodes it at a reduced wavelet
level (a ~thumbnail is nearly free to pull from a JP2), embeds it with the same
OpenCLIP model as the rest of the pipeline, and writes one Parquet file of
(barcode, seq, embedding). CPU-only by default; uses CUDA automatically if a
GPU is present.

    # local test against a hand-made shard:
    python embed_s3.py --shard manifest/shard-00042.tsv --out out/

    # Batch array task (shard chosen from the array index):
    python embed_s3.py --manifest-dir s3://MY-BUCKET/manifest \
                       --out          s3://MY-BUCKET/out

Idempotent per shard: if the output Parquet already exists the task exits at
once, so the array job is freely retryable on spot reclaim.

Reads of bhl-open-data are unsigned (public). Reads/writes of your own
manifest/out bucket use the task's IAM role (default credential chain).
"""
import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are baked into the image

import numpy as np
import torch
import open_clip
from PIL import Image
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
import pyarrow as pa
import pyarrow.parquet as pq

BUCKET = os.environ.get("BHL_S3_BUCKET", "bhl-open-data")
REGION = os.environ.get("BHL_S3_REGION", "us-east-2")
MODEL_NAME = os.environ.get("BHL_CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("BHL_CLIP_PRETRAINED", "laion2b_s34b_b79k")
EMBED_DIM = int(os.environ.get("BHL_EMBED_DIM", "512"))
TARGET_PX = int(os.environ.get("BHL_TARGET_PX", "224"))

# Unsigned client for the public source bucket; signed client for your own
# manifest/output bucket (works off the Batch task role / default creds).
# Bounded timeouts + adaptive retries so transient read timeouts (common over a
# slow/contended link, e.g. a Tier-0 run from a laptop) recover instead of
# silently dropping pages.
SRC_S3 = boto3.client("s3", region_name=REGION,
                      config=Config(signature_version=UNSIGNED,
                                    max_pool_connections=64,
                                    connect_timeout=10, read_timeout=30,
                                    retries={"max_attempts": 5, "mode": "adaptive"}))
IO_S3 = boto3.client("s3", region_name=REGION)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def s3_split(uri):
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def reduced_jp2(data, target=TARGET_PX):
    """Decode a JP2 at the lowest wavelet level whose short side still >= target.

    JPEG-2000 is wavelet-coded, so Pillow can return a 1/2**rlevel-scaled image
    without the full inverse transform -- the cheap path, and all CLIP needs
    since it resizes to 224 anyway. But an image may carry fewer resolution
    levels than we ask for, which makes OpenJPEG raise "broken data stream"; so
    fall back to successively shallower levels, down to a full decode, rather
    than dropping the page. The decode is forced here (convert -> load) so the
    error surfaces inside the retry loop, not later in preprocess.
    """
    short = min(Image.open(io.BytesIO(data)).size)
    rlevel = 0
    while (short >> (rlevel + 1)) >= target and rlevel < 5:
        rlevel += 1
    for r in range(rlevel, -1, -1):
        try:
            im = Image.open(io.BytesIO(data))
            if r:
                im.reduce = r  # Pillow Jpeg2K reduced-resolution decode
            return im.convert("RGB")
        except Exception:
            if r == 0:
                raise  # genuinely undecodable, even at full resolution


def fetch_bytes(key, tries=4):
    """Read an S3 object's bytes; None if genuinely absent or still unreachable.
    A 404 (missing key) returns at once -- only transient errors back off + retry,
    so a missing webp derivative falls through to the JP2 cheaply."""
    for attempt in range(tries):
        try:
            return SRC_S3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            if attempt == tries - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            if attempt == tries - 1:
                return None
            time.sleep(0.5 * (attempt + 1))


def webp_key(jp2_key, size):
    """Map an images/ JP2 key to its web/ webp derivative, preserving the exact
    zero-padding: images/<bc>/<bc>_0007.jp2 -> web/<bc>/<bc>_0007_<size>.webp"""
    base = jp2_key[:-4] if jp2_key.endswith(".jp2") else jp2_key
    if base.startswith("images/"):
        base = "web/" + base[len("images/"):]
    return f"{base}_{size}.webp"


def read_shard(path):
    items = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                key, barcode, seq = parts
                items.append((key, barcode, int(seq)))
    return items


def resolve_shard(args):
    """Return (shard_local_path, output_uri, name) from args / the array index."""
    if args.shard:
        shard_uri = args.shard
    else:
        idx = args.shard_index
        if idx is None:
            idx = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", "0"))
        shard_uri = f"{args.manifest_dir.rstrip('/')}/shard-{idx:05d}.tsv"

    name = os.path.splitext(os.path.basename(shard_uri))[0]
    out_uri = f"{args.out.rstrip('/')}/{name}.parquet"

    if shard_uri.startswith("s3://"):
        local = os.path.join("/tmp", os.path.basename(shard_uri))
        b, k = s3_split(shard_uri)
        IO_S3.download_file(b, k, local)
        shard_local = local
    else:
        shard_local = shard_uri
    return shard_local, out_uri, name


def output_exists(out_uri):
    if out_uri.startswith("s3://"):
        b, k = s3_split(out_uri)
        try:
            IO_S3.head_object(Bucket=b, Key=k)
            return True
        except Exception:
            return False
    return os.path.exists(out_uri)


def write_output(table, out_uri, name):
    if out_uri.startswith("s3://"):
        local = os.path.join("/tmp", f"{name}.parquet")
        pq.write_table(table, local, compression="zstd")
        b, k = s3_split(out_uri)
        IO_S3.upload_file(local, b, k)
        os.unlink(local)
    else:
        os.makedirs(os.path.dirname(out_uri) or ".", exist_ok=True)
        pq.write_table(table, out_uri, compression="zstd")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_argument_group("shard selection (one of)")
    src.add_argument("--shard", help="explicit shard path or s3:// uri")
    src.add_argument("--manifest-dir", default="manifest",
                     help="dir/uri of shard-NNNNN.tsv; index from --shard-index")
    src.add_argument("--shard-index", type=int, default=None,
                     help="defaults to $AWS_BATCH_JOB_ARRAY_INDEX")
    ap.add_argument("--out", default="out", help="output dir or s3:// uri")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--fetch-workers", type=int, default=16)
    ap.add_argument("--min-std", type=float, default=0.0,
                    help="skip near-blank pages: drop if grayscale stddev < this "
                         "(0 = keep everything; ~10 cuts blank/cream leaves)")
    ap.add_argument("--source", choices=["jp2", "webp"], default="jp2",
                    help="read archival JP2 (images/) or the webp derivative "
                         "(web/); webp decodes ~38x faster, falls back to JP2 "
                         "if a derivative is missing")
    ap.add_argument("--webp-size", default="medium",
                    choices=["thumb", "small", "medium", "large", "full"],
                    help="webp derivative size; 'medium' (~465px) gives ~JP2-"
                         "equivalent retrieval (measured p@5 0.60 vs 0.63); "
                         "'small' is faster but lossy (0.53), 'thumb' too small")
    args = ap.parse_args()

    shard_local, out_uri, name = resolve_shard(args)
    if output_exists(out_uri):
        print(f"skip: {out_uri} exists", file=sys.stderr)
        return

    items = read_shard(shard_local)
    src_desc = args.source + (f" (_{args.webp_size})" if args.source == "webp" else "")
    print(f"{len(items):,} images in {name}  [source: {src_desc}]", file=sys.stderr)
    if not items:
        return

    dev = device()
    print(f"device: {dev}", file=sys.stderr)
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model.eval().to(dev)

    def fetch(item):
        key, barcode, seq = item
        status = "ok"
        im = None
        if args.source == "webp":
            data = fetch_bytes(webp_key(key, args.webp_size))
            if data is not None:
                try:
                    im = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    im = None
            if im is None:  # missing/undecodable webp -> fall back to the JP2
                data = fetch_bytes(key)
                if data is None:
                    return barcode, seq, None, "fail"
                try:
                    im = reduced_jp2(data)
                except Exception:
                    return barcode, seq, None, "fail"
                status = "fallback"
        else:
            data = fetch_bytes(key)
            if data is None:
                return barcode, seq, None, "fail"
            try:
                im = reduced_jp2(data)
            except Exception:
                return barcode, seq, None, "fail"
        if args.min_std > 0 and np.asarray(im.convert("L")).std() < args.min_std:
            return barcode, seq, None, "blank"  # near-empty scan -> skip
        return barcode, seq, preprocess(im), status

    barcodes, seqs, vecs = [], [], []
    batch_keys, batch_t = [], []
    done, failed, blank, fellback, t0 = 0, 0, 0, 0, time.time()

    def flush():
        nonlocal done
        if not batch_t:
            return
        with torch.no_grad():
            feats = model.encode_image(torch.stack(batch_t).to(dev))
            feats = torch.nn.functional.normalize(feats, dim=-1)
        arr = feats.cpu().numpy().astype(np.float32)
        for i, (b, s) in enumerate(batch_keys):
            barcodes.append(b)
            seqs.append(s)
            vecs.append(arr[i])
        done += len(batch_keys)
        batch_keys.clear()
        batch_t.clear()

    # Fetch+decode in a thread pool (S3/JP2 bound) while the model encodes.
    with ThreadPoolExecutor(max_workers=args.fetch_workers) as ex:
        for b, s, t, status in ex.map(fetch, items):
            if status == "fail":
                failed += 1
                continue
            if status == "blank":
                blank += 1
                continue
            if status == "fallback":
                fellback += 1
            batch_keys.append((b, s))
            batch_t.append(t)
            if len(batch_t) >= args.batch_size:
                flush()
                rate = done / max(1e-9, time.time() - t0)
                print(f"{done}/{len(items)}  {rate:.0f} img/s", file=sys.stderr)
    flush()

    table = pa.table({
        "barcode": pa.array(barcodes, pa.string()),
        "seq": pa.array(seqs, pa.int32()),
        "embedding": pa.array([v.tolist() for v in vecs],
                              pa.list_(pa.float32(), EMBED_DIM)),
    })
    write_output(table, out_uri, name)
    extra = f", {fellback:,} jp2-fallback" if args.source == "webp" else ""
    print(f"wrote {out_uri}: {len(barcodes):,} vectors "
          f"({failed:,} dropped after retries, {blank:,} blank-skipped{extra}) in "
          f"{(time.time() - t0) / 60:.1f}m", file=sys.stderr)


if __name__ == "__main__":
    main()
