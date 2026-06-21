# Hetzner half — host the vectors, serve the search

The vectors are tiny next to the images, so they live here permanently on a
cheap dedicated box while the heavy embedding ran (once) on AWS. A ~128 GB-RAM
NVMe box holds all 63M `halfvec(512)` vectors (~65 GB) plus the HNSW graph in
memory.

## 1. Postgres + pgvector >= 0.7

`halfvec` needs pgvector 0.7+. On Debian/Ubuntu with PGDG:

```bash
sudo apt install postgresql-16 postgresql-16-pgvector   # or build from source
createdb bhl
psql -d bhl -f ../db/schema_hetzner.sql
```

## 2. Pull the vectors from AWS and load

```bash
aws s3 sync s3://MY-BUCKET/out/ ./out/      # ~65–130 GB, ~free egress
pip install -r requirements.txt
python load_parquet.py --src ./out/ --dsn "postgresql:///bhl"
# (or skip the sync and stream straight from S3:)
# python load_parquet.py --src s3://MY-BUCKET/out/ --dsn "postgresql:///bhl"
```

COPY into the **bare** table first — fast. ~63M rows is a matter of minutes.

## 3. Build the index (the slow, one-time step)

```bash
psql -d bhl -f ../db/index_hetzner.sql      # PK + HNSW; budget a few hours
```

## 4. Query it

`../search.py` works unchanged except for the column type: `halfvec` needs the
query vector cast. Two small edits to `search.py`:

```python
# after computing qvec (a float32 numpy array), build its pgvector literal:
qstr = "[" + ",".join(f"{x:.6g}" for x in qvec) + "]"

# and cast both placeholders to halfvec in the query:
cur.execute(
    "SELECT barcode, seq, 1 - (embedding <=> %s::halfvec) AS score "
    "FROM page_embedding ORDER BY embedding <=> %s::halfvec LIMIT %s",
    (qstr, qstr, args.k + 1),
)
```

Optionally `SET hnsw.ef_search = 100;` per connection to trade speed for
recall. (Drop the `register_vector` call — the literal cast replaces it.)

Note: `thumb_path()` in `search.py` points at the local viewer cache, which
doesn't exist here. For a hosted UI, map `(barcode, seq)` to a BHL page-image
URL instead — the key still joins back to BHL exactly as before.

## Sizing cheat-sheet

| | float32 `vector` | float16 `halfvec` |
|---|---|---|
| 63M × 512 vectors | ~130 GB | **~65 GB** |
| fits in RAM on | 256 GB box | **128 GB box** |

halfvec is the reason a single mid-range Hetzner dedicated server can serve the
whole of BHL from memory.
