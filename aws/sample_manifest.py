"""
Sample a manifest of BHL pages for a local (Tier-0) trial -- no AWS account,
no Batch. Lists item prefixes under images/ in the public bucket, picks a random
set of items, then a random set of pages within each, and writes a
`key<TAB>barcode<TAB>seq` manifest that embed_s3.py consumes directly.

    python aws/sample_manifest.py --items 80 --pages-per-item 20 --out tier0/sample.tsv

Sampling is uniform over the first --scan-pages*1000 items listed. A strictly
uniform draw over all ~290k items would mean paginating the whole prefix list;
for a trial the bounded scan is plenty and much faster. Reproducible via --seed.
"""
import argparse
import os
import random
import sys

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = os.environ.get("BHL_S3_BUCKET", "bhl-open-data")
REGION = os.environ.get("BHL_S3_REGION", "us-east-2")


def list_item_barcodes(s3, scan_pages):
    """Item barcodes from up to scan_pages listing pages of the images/ prefix."""
    barcodes = []
    token = None
    pages = 0
    while True:
        kw = dict(Bucket=BUCKET, Prefix="images/", Delimiter="/")
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for cp in resp.get("CommonPrefixes", []):
            barcodes.append(cp["Prefix"].split("/")[1])  # images/<barcode>/
        pages += 1
        token = resp.get("NextContinuationToken")
        if not token or pages >= scan_pages:
            break
    return barcodes


def item_pages(s3, barcode):
    """Sorted [(key, seq)] page images for one item."""
    out = []
    token = None
    prefix = f"images/{barcode}/"
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix)
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".jp2"):
                continue
            stem = key.rsplit("/", 1)[1][:-4]
            sp = barcode + "_"
            if stem.startswith(sp) and stem[len(sp):].isdigit():
                out.append((key, int(stem[len(sp):])))
        token = resp.get("NextContinuationToken")
        if not token:
            break
    out.sort(key=lambda t: t[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=80)
    ap.add_argument("--pages-per-item", type=int, default=20,
                    help="0 = every page in the item")
    ap.add_argument("--scan-pages", type=int, default=60,
                    help="listing pages of item prefixes to sample from (x1000 items)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="sample.tsv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    s3 = boto3.client("s3", region_name=REGION,
                      config=Config(signature_version=UNSIGNED))

    print(f"listing item prefixes (up to {args.scan_pages * 1000:,})...",
          file=sys.stderr)
    barcodes = list_item_barcodes(s3, args.scan_pages)
    print(f"{len(barcodes):,} items seen; sampling {args.items}", file=sys.stderr)
    chosen = rng.sample(barcodes, min(args.items, len(barcodes)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = 0
    with open(args.out, "w") as fh:
        for i, bc in enumerate(chosen, 1):
            pages = item_pages(s3, bc)
            if not pages:
                continue
            if args.pages_per_item and len(pages) > args.pages_per_item:
                pages = sorted(rng.sample(pages, args.pages_per_item),
                               key=lambda t: t[1])
            for key, seq in pages:
                fh.write(f"{key}\t{bc}\t{seq}\n")
                total += 1
            print(f"  [{i}/{len(chosen)}] {bc}: {len(pages)} pages",
                  file=sys.stderr)
    print(f"done: {total:,} pages from {len(chosen)} items -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
