-- Run AFTER load_parquet.py has COPYed every shard into page_embedding.
-- Apply with:  psql "$DATABASE_URL" -f db/index_hetzner.sql
--
-- Building HNSW over 63M rows is the slow, one-time step: give it RAM and
-- parallelism. On an N-core box set workers to ~N-1; expect a few hours.
-- (halfvec keeps the in-memory graph small enough to build without thrashing.)

SET maintenance_work_mem = '8GB';
SET max_parallel_maintenance_workers = 7;

ALTER TABLE page_embedding ADD PRIMARY KEY (barcode, seq);

CREATE INDEX IF NOT EXISTS page_embedding_hnsw
    ON page_embedding
    USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

ANALYZE page_embedding;

-- Query-time recall/speed tradeoff (per session, raise for better recall):
--   SET hnsw.ef_search = 100;
