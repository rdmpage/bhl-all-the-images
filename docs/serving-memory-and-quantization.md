# Serving the image-search vectors: memory requirements, and why compression doesn't rescue us

*Experiment run 2026-06-29 on the live 589K-vector box. Author: Rod Page (with Claude Code).*

## TL;DR

- To make **all of BHL (~60M page-images)** searchable with full retrieval quality, the
  serving machine needs to hold the vector index **in RAM** — that is ~**80 GB**, so a
  **~128 GB box**. This is the cost of the data structure, not over-provisioning.
- We tested whether **binary quantization** (compressing each vector ~16×) could let us serve
  the same corpus from a small (16–32 GB) box. **It can't, at acceptable quality.** It loses
  too much retrieval accuracy on real text queries (recall@12 drops from ~0.88 to ~0.74), and
  the index only shrinks ~3.8× anyway because most of an index's size is its graph, not its
  vectors.
- The only way to run on a small box is to **index fewer pages**. A few-million-page slice
  (e.g. BioStor, ~3.6M) fits a 16 GB box at *full* quality. The RAM bill is a function of
  corpus size, not of any tunable we've found.

So: **small box for a slice, big box for all of BHL.** There is no cheap middle.

---

## Background: why RAM is the constraint

The architecture (see the wider embedding plan) is a deliberate **data-gravity split**: we
embed the page-images on AWS in-region (the images are ~100+ TB and never move) and export only
the resulting vectors, which we serve from a single box running PostgreSQL + `pgvector`.

Each page-image becomes one **512-dimensional CLIP vector** (model `ViT-B-32`,
`laion2b_s34b_b79k`), stored as `halfvec` (16-bit floats) = ~1 KB per page. Search is
approximate-nearest-neighbour (ANN) over these vectors using an **HNSW index** (the standard
high-recall ANN structure in pgvector).

The catch with HNSW: a search walks a graph, **hopping to hundreds–thousands of arbitrary
nodes per query**. That random-access pattern is the worst case for spilling to disk —

| storage | random read latency |
|---|---|
| RAM | ~0.1 µs |
| NVMe SSD | ~50–100 µs (500–1000× slower) |

If even a fraction of those hops miss RAM and hit disk, a query that should take single-digit
milliseconds takes hundreds of milliseconds to seconds. **Recall is unaffected by RAM** (that
depends on the index parameters); what suffers is latency and throughput. So the practical rule
is simple: **the HNSW index must fit in RAM.** That is what sizes the box.

---

## The question we tested

The full-corpus index is ~80 GB (derivation below), which means a ~128 GB box — not cheap to
rent month after month. Before committing to that, we tested the obvious escape hatch:

> Can we **compress the vectors** so the index fits a small, cheap box, without wrecking search
> quality?

The candidate was **binary quantization**: reduce each 512-dim float vector to 512 *bits* (keep
only the sign of each dimension), search that tiny representation with Hamming distance, then
**re-rank** the top *F* candidates using the original `halfvec` vectors. This is a well-known
trick for serving large vector sets on small machines. The question is whether it holds up *on
our data*.

## Method

Run against the **existing 589,043-vector set** (the BioStor demo corpus already loaded on the
box) — no new infrastructure needed. Tooling committed to the repo:

- [`db/bq_eval.sql`](../db/bq_eval.sql) — adds the binary (`bit(512)`) column and its Hamming
  HNSW index alongside the existing `halfvec` data.
- [`hetzner/bq_recall_eval.py`](../hetzner/bq_recall_eval.py) — measures **recall@12 vs. exact
  brute-force search**, for two query types, sweeping the re-rank pool size *F*.

**recall@12** = of the 12 truly-nearest results (computed by exhaustive search, the ground
truth), how many does the method return? `1.000` = identical to brute force. We compare two
methods against that ground truth:

1. **halfvec HNSW** — the current production path (full-precision vectors in the index). This is
   what a 128 GB box gives you.
2. **bit + re-rank** — the cheap-box candidate described above, swept over candidate-pool *F*.

Two query distributions, because they behave differently:

- **corpus** — random stored vectors used as queries (tests whether quantization preserves the
  raw vector *geometry*; 300 queries).
- **text** — natural-language queries encoded by CLIP, e.g. *"a colour plate of birds"*, *"a
  portrait of a person"* (tests the queries users **actually send**; 12 queries).

---

## Result 1 — the index barely shrinks

Measured index sizes for the 589K-vector corpus:

| structure | size | per vector |
|---|---|---|
| `halfvec` HNSW (full precision) | **784 MB** | ~1.4 KB |
| `bit` HNSW (binary quantized) | **207 MB** | ~0.4 KB |

Binary quantization compresses the *vectors* ~16× (1 KB → 64 bytes), but the **index** only
shrinks **3.8×**. Why: an HNSW index stores the vectors **plus the navigation graph**, and the
graph is made of node pointers whose size doesn't depend on how the vectors are stored. Both
indexes carry ~170–180 MB of graph links; quantization only shrinks the other part.

```
halfvec index = ~603 MB vectors + ~181 MB graph links = 784 MB
bit index     =  ~38 MB vectors + ~169 MB graph links = 207 MB   <- graph dominates
```

So even in the best case, quantization is a 3.8× memory saving, not the ~16× the raw numbers
suggest.

## Result 2 — and it costs too much accuracy

**recall@12 vs. exact search** (higher is better; `F` = re-rank candidate pool):

| method | corpus queries | text queries |
|---|---:|---:|
| **halfvec HNSW** (production / big box) | **0.985** | **0.875** |
| bit + re-rank, F=12 | 0.358 | 0.139 |
| bit + re-rank, F=50 | 0.633 | 0.354 |
| bit + re-rank, F=100 | 0.740 | 0.458 |
| bit + re-rank, F=200 | 0.833 | 0.618 |
| bit + re-rank, F=500 | 0.909 | 0.736 |

Reading this:

- The full-precision path is **near-perfect on corpus queries (0.985)** and good on text (0.875).
- Binary quant **never catches up.** Even fetching and re-ranking **500 candidates** to return
  12, it reaches only 0.909 (corpus) / **0.736 (text)** — meaning roughly **a quarter of the
  truly-best results are missing** on real text queries.
- **Text is the case that matters, and it's the worst.** Text queries land in the *sparse*
  regions of the vector space (maps, portraits, photographs of people — categories that are thin
  in the corpus), which is exactly where the coarse sign-bit approximation breaks down.

The trade is bad on both axes: a modest 3.8× memory saving in exchange for a large, user-visible
drop in result quality. **Binary quantization is rejected.**

---

## The real lever: corpus size, not compression

Since compression doesn't help, the index size is essentially fixed by the number of pages. From
the measured 784 MB / 589K, the full-precision HNSW index scales at **~1.33 GB per million
vectors**, at full retrieval quality:

| what you index | vectors | index size | RAM / box | quality |
|---|---:|---:|---|---|
| BioStor articles | ~3.6M | ~4.8 GB | **16 GB** | full (0.985) |
| BioStor all-pages | ~7M | ~9 GB | 16–32 GB | full |
| **All of BHL** | **~60M** | **~80 GB** | **128 GB** | full |

(~60M = the ~63M BHL pages minus blank/near-blank pages removed by a filter.)

This is the rationale for the big box in one line: **a high-quality, in-memory ANN index over
all ~60M BHL page-images is ~80 GB, and an 80 GB index needs a ~128 GB machine to live in RAM.**
The alternative isn't a cleverer index — we tested that — it's indexing fewer pages.

## Recommendation

1. **For a demo / proof-of-value:** serve a few-million-page slice (BioStor, ~3.6M) on a **16 GB
   box** at full quality. Cheap, and already proven — the 589K set runs comfortably on an 8 GB box
   today.
2. **For full-BHL search:** budget a **~128 GB-RAM box**. This is a real recurring cost
   (≈€100–120/month at Hetzner dedicated pricing), justified by the numbers above, not padding.
3. **Do not** spend effort on quantization to avoid (2) — measured, it doesn't deliver acceptable
   quality on real queries.

### Side finding, already applied

The production search default was `hnsw.ef_search = 100`, which gave only **0.875** recall on
text queries. Raising it to **300** recovers most of the gap for a few milliseconds of extra
latency; this is now the default (`BHL_HNSW_EF_SEARCH`, overridable in `/etc/bhl-search.env`).

---

## Reproduce it

On the serving box, with the 589K corpus loaded:

```bash
export DATABASE_URL=postgresql:///bhl
psql "$DATABASE_URL" -f db/bq_eval.sql                       # build the bit index + sizes
hetzner/.venv/bin/python hetzner/bq_recall_eval.py --mode corpus --queries 300 --k 12
hetzner/.venv/bin/python hetzner/bq_recall_eval.py --mode text --k 12
```

## Caveats

- **One model.** All numbers are for OpenCLIP `ViT-B/32` (512-d). A larger model would change
  absolute recall but not the structural conclusions (graph-dominated index size; quantization
  loss worst in sparse regions).
- **Small text sample.** Only 12 text queries (a noisy estimate), but it agrees in direction with
  the 300-query corpus run, and text is consistently worse — enough to make the call.
- **Strict metric.** recall@12 vs. *exact* penalizes any swap, even when the swapped-in result is
  almost as relevant; user-perceived quality may be somewhat better than the raw numbers. This
  cuts the same way for all methods and doesn't change the ranking between them.
- **Memory, not latency, at scale.** Latency was not measured at 60M (the 589K set fits RAM
  either way, so it can't show the disk-thrash effect). The recall findings transfer directly; the
  RAM requirement is the arithmetic above.
