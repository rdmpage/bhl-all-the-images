-- Hetzner-side pgvector store for the full-BHL run.
-- Apply with:  psql "$DATABASE_URL" -f db/schema_hetzner.sql
--
-- Requires pgvector >= 0.7 (for halfvec) on Postgres >= 14.
--
-- halfvec(512) is float16: ~65 GB for 63M rows vs ~130 GB at float32, so the
-- whole set plus the HNSW graph fits in RAM on a ~128 GB box. CLIP vectors are
-- L2-normalised and tolerate fp16 with negligible recall loss.
--
-- Load order matters: COPY into this bare table first (load_parquet.py), THEN
-- add the primary key and build HNSW (db/index_hetzner.sql). Bulk-loading into
-- an already-indexed table is far slower.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS page_embedding (
    barcode    text         NOT NULL,
    seq        integer      NOT NULL,
    embedding  halfvec(512) NOT NULL
);
