"""
Load the AWS-produced embedding Parquet shards into Hetzner Postgres/pgvector.

Streams each Parquet file into page_embedding with COPY -- the fast bulk path.
The table should be bare (db/schema_hetzner.sql); build the PK + HNSW index
afterwards with db/index_hetzner.sql. Source is a local directory of *.parquet
or the s3://bucket/prefix the worker wrote to.

    python load_parquet.py --src ./out/                 --dsn "$DATABASE_URL"
    python load_parquet.py --src s3://MY-BUCKET/out/    --dsn "$DATABASE_URL"

The embedding column is halfvec(512); COPY sends each vector as its pgvector
text form `[v1,v2,...]`, which Postgres parses straight into halfvec.
"""
import argparse
import glob
import os
import sys
import tempfile

import pyarrow.parquet as pq
import psycopg


def iter_files(src):
    """Yield (local_path, is_temp) for every *.parquet under src."""
    if src.startswith("s3://"):
        import boto3
        s3 = boto3.client("s3")
        bucket, _, prefix = src[len("s3://"):].partition("/")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".parquet", delete=False).name
                    s3.download_file(bucket, obj["Key"], tmp)
                    yield tmp, True
    elif os.path.isfile(src):
        yield src, False
    else:
        for path in sorted(glob.glob(os.path.join(src, "*.parquet"))):
            yield path, False


def copy_file(conn, path):
    table = pq.read_table(path, columns=["barcode", "seq", "embedding"])
    barcodes = table.column("barcode").to_pylist()
    seqs = table.column("seq").to_pylist()
    embs = table.column("embedding").to_pylist()
    with conn.cursor() as cur:
        with cur.copy(
            "COPY page_embedding (barcode, seq, embedding) FROM STDIN"
        ) as cp:
            for barcode, seq, emb in zip(barcodes, seqs, embs):
                lit = "[" + ",".join(f"{x:.6g}" for x in emb) + "]"
                cp.write_row((barcode, seq, lit))
    conn.commit()
    return len(barcodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="directory or s3://bucket/prefix of *.parquet")
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL",
                                                     "postgresql:///bhl"))
    args = ap.parse_args()

    conn = psycopg.connect(args.dsn)
    total = 0
    for path, is_tmp in iter_files(args.src):
        n = copy_file(conn, path)
        total += n
        print(f"{os.path.basename(path)}: +{n:,}  (total {total:,})",
              file=sys.stderr)
        if is_tmp:
            os.unlink(path)
    conn.close()
    print(f"done: {total:,} vectors loaded", file=sys.stderr)


if __name__ == "__main__":
    main()
