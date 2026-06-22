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

## 4. Serve it

`search_api.py` (FastAPI) is the serving process: it loads the same OpenCLIP
model, encodes a text/image query, runs the pgvector ANN, and returns JSON with
public S3 webp image URLs. Run it next to Postgres:

```bash
pip install -r requirements-api.txt          # CPU torch + fastapi + ...
export DATABASE_URL=postgresql:///bhl
uvicorn search_api:app --host 127.0.0.1 --port 8000
curl 'http://127.0.0.1:8000/search?q=a+colour+plate+of+birds&k=6'
```

Key `halfvec` details (already handled in `search_api.py`): the query vector is
sent as a pgvector text literal `[v1,v2,...]` and cast `%s::halfvec` (no
`register_vector` adapter needed); `SET hnsw.ef_search = <n>` per query trades
speed for recall — and note `SET` takes a literal, **not** a bind parameter, so
the value is interpolated, not passed as `%s`.

**For the full validated walkthrough — box, Postgres, pgvector build, load,
index, serve, and the PHP demo — follow [`dry_run.md`](dry_run.md).** It is the
tested recipe (Ubuntu 26.04 / PG 18) with every gotcha folded in.

Image URLs are reconstructed from `(barcode, seq)` as
`web/<bc>/<bc>_<seq:04d>_<size>.webp` on the public bucket — nothing is hosted
here; the key still joins back to BHL exactly as before. (`../search.py` remains
the local-CLI tool against the old `vector(512)` PoC table; it is not used here.)

## Sizing cheat-sheet

| | float32 `vector` | float16 `halfvec` |
|---|---|---|
| 63M × 512 vectors | ~130 GB | **~65 GB** |
| fits in RAM on | 256 GB box | **128 GB box** |

halfvec is the reason a single mid-range Hetzner dedicated server can serve the
whole of BHL from memory.
