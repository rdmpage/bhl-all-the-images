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

## How it works

1. **`embed.py`** walks `cache/thumbs/{BarCode}/{seq}.webp`, runs each image
   through OpenCLIP **ViT-B/32** (`laion2b_s34b_b79k`, 512-d), L2-normalises the
   vector, and upserts it into the `page_embedding` table. Image decoding is
   parallelised across CPU workers; encoding uses the best available device.
   The run is **resumable** — keys already in the table are skipped.
2. **`search.py`** embeds a query (an image file or an in-set page) with the
   same model and returns the nearest pages by cosine distance (exact KNN),
   optionally writing an HTML contact sheet for eyeballing the results.

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
```

Each result line is `cosine  BarCode/seq  thumb_path`. `--html` writes a contact
sheet (query + ranked thumbnails) to the given path.

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
| `search.py` | query by `--image` or `--page`, top-K, optional `--html` |
