"""
Build an embedding manifest from an explicit list of BHL item barcodes.

Given a file of barcodes (one per line -- the same strings used as the
`images/<barcode>/` folder names in the bucket, e.g. BioStor's IA identifiers),
list every page image of each item in the public bucket and write the sharded
`key<TAB>barcode<TAB>seq` manifest that embed_s3.py consumes directly. This is
the curated alternative to sample_manifest.py's random draw: you decide exactly
which items go in (e.g. 2000 recent BioStor items).

    python aws/manifest_from_ids.py --ids items.txt --out manifest/ --shards 64
    #   -> prints the total page count (your corpus size) and any missing ids

Keys are the .jp2 form (images/<bc>/<bc>_<seq>.jp2); embed_s3.py --source webp
derives the faster web/ webp derivative from each. --pages-per-item defaults to
0 = every page (right for an image demo: plates/figures are interleaved, not on
the indexed article pages). Listing is threaded; reads are unsigned (public).
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = os.environ.get("BHL_S3_BUCKET", "bhl-open-data")
REGION = os.environ.get("BHL_S3_REGION", "us-east-2")


def item_pages(s3, barcode):
    """Sorted [(key, seq)] of .jp2 page images for one item ([] if none)."""
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


def read_ids(path):
    """Barcodes from a file: one per line, blanks and #-comments ignored,
    deduplicated while preserving order."""
    seen, ids = set(), []
    with open(path) as fh:
        for line in fh:
            bc = line.strip()
            if not bc or bc.startswith("#") or bc in seen:
                continue
            seen.add(bc)
            ids.append(bc)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="file of barcodes, one per line")
    ap.add_argument("--out", default="manifest")
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--pages-per-item", type=int, default=0,
                    help="0 = every page (default); else first N pages of each item")
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent S3 listers")
    args = ap.parse_args()

    ids = read_ids(args.ids)
    print(f"{len(ids):,} unique barcodes", file=sys.stderr)

    s3 = boto3.client("s3", region_name=REGION,
                      config=Config(signature_version=UNSIGNED,
                                    max_pool_connections=args.workers * 2))

    os.makedirs(args.out, exist_ok=True)
    handles = [open(os.path.join(args.out, f"shard-{i:05d}.tsv"), "w")
               for i in range(args.shards)]

    total = missing = 0
    missing_ids = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for bc, pages in zip(ids, ex.map(lambda b: item_pages(s3, b), ids)):
                if not pages:
                    missing += 1
                    missing_ids.append(bc)
                    continue
                if args.pages_per_item:
                    pages = pages[:args.pages_per_item]
                for key, seq in pages:
                    handles[total % args.shards].write(f"{key}\t{bc}\t{seq}\n")
                    total += 1
    finally:
        for h in handles:
            h.close()

    print(f"done: {total:,} pages from {len(ids) - missing:,} items "
          f"across {args.shards} shards", file=sys.stderr)
    if missing:
        with open(os.path.join(args.out, "missing_ids.txt"), "w") as mf:
            mf.write("\n".join(missing_ids) + "\n")
        print(f"WARNING: {missing:,} barcodes had no images in the bucket "
              f"-> {args.out}/missing_ids.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
