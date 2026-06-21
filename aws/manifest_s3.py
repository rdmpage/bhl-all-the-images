"""
Build the sharded work manifest for the BHL AWS Open Data image dump.

Lists the public bucket s3://bhl-open-data (us-east-2, unsigned reads), maps
each page-image key to the pipeline's (barcode, seq) page key, and writes N
shard files of TAB-separated `key<TAB>barcode<TAB>seq` lines. The Batch array
job (embed_s3.py) then processes one shard per task.

    python manifest_s3.py --sample 40            # print first keys, confirm layout
    python manifest_s3.py --shards 1000 --out manifest/
    aws s3 sync manifest/ s3://MY-BUCKET/manifest/   # stage for Batch

Key layout (confirmed against the bucket README, 2026-06):
    images/[BarCode]/[BarCode]_[####].jp2
The BarCode is one path segment but may contain '.', '-' and mixed case
(e.g. 00921238.85096.emory.edu); the sequence is zero-padded and starts at 1.
parse_key() matches this exactly. Everything downstream is layout-agnostic --
the (barcode, seq) mapping is resolved here, once, and carried in the manifest.

Faster alternative for a full run: rather than ~63k LIST calls over the whole
bucket, read data/item.txt.gz (every BarCode) + data/page.txt.gz (page count
per item) from the same bucket and synthesise keys (pages never skip numbers,
first image is always _0001.jp2). The S3 listing here is the no-extra-inputs
path.

Note: writing N shard files keeps N file handles open. For --shards > ~1000
raise the limit first, e.g.  ulimit -n 4096.
"""
import argparse
import os
import sys

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = os.environ.get("BHL_S3_BUCKET", "bhl-open-data")
REGION = os.environ.get("BHL_S3_REGION", "us-east-2")


def parse_key(key):
    """(barcode, seq) for a page-image key, or None if it isn't one.

    Exact match for  images/<barcode>/<barcode>_<seq>.jp2  -- the barcode dir
    and the filename prefix must agree, which also rejects stray keys.
    """
    if not key.startswith("images/") or not key.lower().endswith(".jp2"):
        return None
    parts = key.split("/")
    if len(parts) != 3:
        return None
    _, barcode, fname = parts
    prefix = barcode + "_"
    stem = fname[:-4]  # drop ".jp2"
    if not stem.startswith(prefix):
        return None
    seq = stem[len(prefix):]
    if not seq.isdigit():
        return None
    return barcode, int(seq)


def iter_keys(s3, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manifest")
    ap.add_argument("--shards", type=int, default=1000)
    ap.add_argument("--prefix", default=os.environ.get("BHL_S3_PREFIX", "images/"))
    ap.add_argument("--sample", type=int, default=0,
                    help="just print the first N keys and exit (layout check)")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=REGION,
                      config=Config(signature_version=UNSIGNED))

    if args.sample:
        for i, key in enumerate(iter_keys(s3, args.prefix)):
            print(key)
            if i + 1 >= args.sample:
                break
        return

    os.makedirs(args.out, exist_ok=True)
    handles = [open(os.path.join(args.out, f"shard-{i:05d}.tsv"), "w")
               for i in range(args.shards)]

    total = skipped = 0
    try:
        for key in iter_keys(s3, args.prefix):
            pk = parse_key(key)
            if pk is None:
                skipped += 1
                continue
            barcode, seq = pk
            handles[total % args.shards].write(f"{key}\t{barcode}\t{seq}\n")
            total += 1
            if total % 1_000_000 == 0:
                print(f"{total:,} keys...", file=sys.stderr)
    finally:
        for h in handles:
            h.close()

    print(f"done: {total:,} images across {args.shards} shards "
          f"({skipped:,} unparseable keys)", file=sys.stderr)


if __name__ == "__main__":
    main()
