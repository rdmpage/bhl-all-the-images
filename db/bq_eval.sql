-- Binary-quantization recall experiment on the CURRENT box (the Tier-0 / BioStor
-- 589K set). Question being answered: can a tiny bit(512) Hamming HNSW + an exact
-- halfvec re-rank match the full halfvec HNSW closely enough that the full-BHL
-- run could serve from a small (16-32 GB) box instead of a ~128 GB one?
--
-- This adds the bit column + its index next to the existing halfvec data; it
-- does NOT touch page_embedding.embedding or the existing HNSW. Idempotent.
--
-- Requires pgvector >= 0.7 (binary_quantize + bit_hamming_ops). The current box
-- built pgvector from a recent tag, so this is satisfied.
--
--   psql "$DATABASE_URL" -f db/bq_eval.sql

-- 1) Sign-bit quantization: each 512-d vector -> 512-bit string. binary_quantize
--    keeps just the sign of each dimension (>0 -> 1). 64 bytes/row vs ~1 KB for
--    halfvec. A plain column + UPDATE (rather than a GENERATED column) is
--    portable across pgvector point releases and lets us re-run cheaply.
ALTER TABLE page_embedding ADD COLUMN IF NOT EXISTS embedding_bit bit(512);

UPDATE page_embedding
   SET embedding_bit = binary_quantize(embedding::vector)::bit(512)
 WHERE embedding_bit IS NULL;

-- 2) HNSW over Hamming distance on the bit column. THIS is the structure that
--    would live in RAM on the small box. At full-BHL scale it is ~13x smaller
--    than the halfvec HNSW, which is the entire RAM argument.
SET maintenance_work_mem = '2GB';
SET max_parallel_maintenance_workers = 7;

CREATE INDEX IF NOT EXISTS page_embedding_bit_hnsw
    ON page_embedding
    USING hnsw (embedding_bit bit_hamming_ops)
    WITH (m = 16, ef_construction = 64);

ANALYZE page_embedding;

-- 3) Footprint comparison -- the headline number for the RAM decision.
--    to_regclass(...) yields NULL (not an error) if the halfvec HNSW is named
--    differently or absent, so this never fails the script.
SELECT
    pg_size_pretty(pg_relation_size(to_regclass('page_embedding_hnsw')))    AS halfvec_hnsw,
    pg_size_pretty(pg_relation_size('page_embedding_bit_hnsw'))             AS bit_hnsw,
    pg_size_pretty(pg_total_relation_size('page_embedding'))                AS table_total;
