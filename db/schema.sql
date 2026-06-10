-- pgvector store for BHL page-image embeddings.
-- Apply with: psql -d bhl -f db/schema.sql
--
-- pgvector 0.4.x (Postgres.app) predates HNSW, so we rely on exact KNN by
-- default. At ~218k rows a brute-force cosine scan is ~100-300ms/query and
-- has zero recall loss. If/when it needs to scale, add the IVFFlat index at
-- the bottom (or upgrade pgvector to >=0.5 for HNSW).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS page_embedding (
    barcode    text    NOT NULL,
    seq        integer NOT NULL,
    embedding  vector(512) NOT NULL,
    PRIMARY KEY (barcode, seq)
);

-- Optional ANN index for scaling (pgvector >= 0.5 needed for HNSW instead):
--   SET maintenance_work_mem = '512MB';
--   CREATE INDEX ON page_embedding USING ivfflat (embedding vector_cosine_ops)
--       WITH (lists = 500);
--   ANALYZE page_embedding;
