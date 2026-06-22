# bhl-all-the-images

Experiments with page-image embeddings: embed every BHL page thumbnail with
[OpenCLIP](https://github.com/mlfoundations/open_clip), store the vectors in
[pgvector](https://github.com/pgvector/pgvector), and search the collection by
visual similarity. A query can be an **existing page** ("show me pages like this
one") or **any external image** — an unseen BHL plate, or even a photo of a real
organism — which is embedded into the same space and matched against the set.

This is one of three sibling repos that share a single page key
(`BarCode` + `seq`, plus `PageID`):

- `bhl-all-the-pages` — the Leaflet viewer + the canonical `cache/thumbs/`
  (218,567 thumbnails, 150×234 webp). **This repo reads those thumbs directly.**
- `bhl-all-the-text` — OCR text fetch + locality / taxonomic-name extraction.
- `bhl-all-the-images` — *this repo*.

## Two halves of this repo

1. **Local PoC** (this README below) — `embed.py` / `search.py` / `label.py`
   over the 218k local thumbnail cache and a local pgvector DB. Where the
   approach was first validated.
2. **BHL-wide scale-up** — embed *all* ~63M BHL pages on AWS (in-region, reading
   the public `bhl-open-data` webp derivatives), then serve the vectors from a
   single pgvector box and search them from a thin web demo:
   - `aws/` — the in-region embedding workers + the cost/eval harness.
   - `db/schema_hetzner.sql`, `hetzner/load_parquet.py`, `hetzner/search_api.py`
     — the serving half (`halfvec` + HNSW + a FastAPI CLIP search service).
   - **[`hetzner/dry_run.md`](hetzner/dry_run.md)** — the tested end-to-end
     runbook (✅ validated on a live box 2026-06-22), from bare OS to a working
     search endpoint on the Tier-0 vectors, no AWS spend.
   - `demo/index.php` — a PHP front-end (text + image-similarity search).

   See `hetzner/README.md` and `aws/README.md` for the architecture rationale
   (embed where the data is; serve the tiny vectors anywhere).

## How it works

1. **`embed.py`** walks `cache/thumbs/{BarCode}/{seq}.webp`, runs each image
   through OpenCLIP **ViT-B/32** (`laion2b_s34b_b79k`, 512-d), L2-normalises the
   vector, and upserts it into the `page_embedding` table. Image decoding is
   parallelised across CPU workers; encoding uses the best available device.
   The run is **resumable** — keys already in the table are skipped.
2. **`search.py`** embeds a query with the same model and returns the nearest
   pages by cosine distance (exact KNN), optionally writing an HTML contact
   sheet for eyeballing the results. A query can be an **image file**, an
   **in-set page** (`--page BarCode/seq`), or — because CLIP maps text into the
   *same* 512-d space — a **text phrase** (`--text "a map"`), which ranks pages
   directly with no labels to precompute.
3. **`label.py`** uses that shared space the other way round: it scores the
   already-stored page vectors against a set of candidate text labels (softmax
   across the label set) to **zero-shot classify** pages — e.g. map / portrait /
   plate / beetle — without re-embedding any images.

The key is derived straight from the thumb path, so every result joins back to
the viewer and to the text repo on `BarCode + seq`.

## Setup

Requires Python 3 and a local Postgres with the `vector` extension
(Postgres.app ships it).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

createdb bhl
psql -d bhl -f db/schema.sql
```

Configuration (`config.py`) can be overridden by environment variables:

| var | default | meaning |
|---|---|---|
| `BHL_THUMBS_DIR` | `~/Sites/bhl-all-the-pages/cache/thumbs` | thumbnail source |
| `DATABASE_URL` | `postgresql:///bhl` | pgvector database |
| `BHL_CLIP_MODEL` / `BHL_CLIP_PRETRAINED` | `ViT-B-32` / `laion2b_s34b_b79k` | model |

## Usage

Embed the whole collection (resumable; ~38 min on CPU for 218k thumbs):

```bash
.venv/bin/python embed.py
# or a quick sample: .venv/bin/python embed.py --limit 800
```

Search:

```bash
# an external image (organism photo, unseen page, anything Pillow can read):
.venv/bin/python search.py --image /path/to/query.jpg -k 20 --html results.html

# an in-set page — "images like this one":
.venv/bin/python search.py --page systemaaviumaus00math/378 -k 20 --html results.html

# a text phrase — CLIP ranks pages against the words directly:
.venv/bin/python search.py --text "a map" -k 30 --html maps.html
```

Each result line is `cosine  BarCode/seq  thumb_path`. `--html` writes a contact
sheet (query + ranked thumbnails) to the given path.

Zero-shot label pages against candidate text labels (built-in set, a `--labels`
file, or repeated `--label`); `--only`/`--min-score` filter to confident hits:

```bash
.venv/bin/python label.py --limit 200 --html labels.html
.venv/bin/python label.py --labels labels.txt --only map --min-score 0.5 --html maps.html
```

## Notes & current state

- **Status:** all 218,566 readable thumbs are embedded (one 0-byte source thumb
  was skipped). Both the in-set and external-image search paths are validated.
- **Device:** runs on **CPU** (~84 img/s). torch reports MPS *built* but *not
  available* because the host is macOS 13 and recent torch needs macOS 14+ for
  MPS. The code auto-selects MPS/CUDA if either becomes available.
- **Query latency:** exact KNN over 218k vectors is ~2 s/query — fine for CLI
  use. pgvector 0.4.1 (bundled with Postgres.app) predates HNSW; an **IVFFlat**
  index (stubbed in `db/schema.sql`) is the scaling step for interactive search.
- **Interpreting cosine scores:** the corpus is mostly scanned text pages, so
  CLIP embeddings are highly anisotropic — **two random pages average ~0.61
  cosine**, not 0. The signal is in the *ranking* and the top tail (>0.90), not
  the absolute value. Mean-centering the embeddings (subtract the corpus mean,
  renormalise) re-centers random pairs near 0 and sharpens discrimination — a
  one-step upgrade if intuitive thresholds are wanted.
- **Model choice:** ViT-B/32 was picked to validate the pipeline and whether
  150px thumbs suffice. If retrieval is weak (especially the organism-photo →
  illustration cross-domain case), re-embed with ViT-L or SigLIP and/or fetch
  the larger `_medium` page images.

## Files

| file | purpose |
|---|---|
| `config.py` | thumbs dir, DB DSN, model, device auto-detect |
| `db/schema.sql` | pgvector table (exact KNN; IVFFlat noted for scaling) |
| `embed.py` | thumbs → CLIP → pgvector, parallel decode, resumable |
| `search.py` | query by `--image`, `--page`, or `--text`; top-K, optional `--html` |
| `label.py` | zero-shot label stored vectors against candidate text labels |
| `examples/` | documented sample query images |
