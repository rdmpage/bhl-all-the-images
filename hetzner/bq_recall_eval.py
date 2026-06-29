"""
Does bit(512) + re-rank match the full halfvec HNSW?  Recall experiment for the
"serve full BHL from a small box" question (see bhl-wide-embedding-plan).

Run db/bq_eval.sql first (adds page_embedding.embedding_bit + its Hamming HNSW).
Then, on the Hetzner box:

    export DATABASE_URL=postgresql:///bhl
    python hetzner/bq_recall_eval.py --queries 300 --k 12

For each of N random corpus vectors used as a probe, we compute three top-k sets
and compare them:

  * ground truth  -- exact top-k cosine over halfvec (forced seq scan, no index)
  * halfvec HNSW  -- the CURRENT production path; the recall a 128 GB box buys
  * bit + re-rank -- Hamming HNSW over bit(512) -> fetch F candidates -> re-rank
                     those by exact halfvec cosine -> top-k  (the cheap-box path)

and report recall@k vs ground truth + median latency, swept over the candidate
pool size F. recall@k = |method_topk INTERSECT truth_topk| / k, averaged over
probes. The probe's own row is excluded everywhere (self-distance 0 would always
rank first and inflate every method equally).

Two probe modes (--mode):

  corpus  (default) -- probes are random corpus vectors. Model-free, fast, and a
                       fair geometric test of the ANN approximation.
  text              -- probes are CLIP-encoded TEXT queries (the real
                       text->image distribution your demo actually serves). Uses
                       --text-file (one query per line) or a built-in default
                       list. Needs torch + open_clip (already on the box for the
                       API). Encoding is identical to search_api.py so the probe
                       lands in the same space as the corpus.

Corpus mode answers "does binary quant preserve the vector geometry"; text mode
answers "does it preserve the answers users actually get". Run both -- text
queries probe a slightly different region than corpus vectors do.
"""
import argparse
import os
import statistics
import time

import psycopg

DSN = os.environ.get("DATABASE_URL", "postgresql:///bhl")

# Kept identical to aws/embed_s3.py / search_api.py so a text probe lands in the
# same space as the corpus vectors. Only loaded in --mode text.
MODEL_NAME = os.environ.get("BHL_CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("BHL_CLIP_PRETRAINED", "laion2b_s34b_b79k")

# A spread of content categories from the Tier-0 eval -- maps, plates, portraits,
# people, plus text/structure pages. Override with --text-file for your own set.
DEFAULT_QUERIES = [
    "a geographic map",
    "a map of a continent",
    "a colour plate of birds",
    "a botanical illustration of a flower",
    "a portrait of a person",
    "a black and white photograph of people",
    "a page of printed text",
    "a scientific diagram",
    "a table of numbers",
    "an illustration of an insect",
    "a drawing of a fish",
    "a decorative title page",
]

# Exact and halfvec-HNSW paths share one SQL string; the ONLY difference is
# whether index scans are enabled on the connection running it.
HALFVEC_SQL = """
SELECT barcode, seq
FROM page_embedding
WHERE NOT (barcode = %(qb)s AND seq = %(qs)s)
ORDER BY embedding <=> %(q)s::halfvec
LIMIT %(k)s
"""

# Inner ORDER BY rides the bit Hamming HNSW (<~>) to pull F candidates; the outer
# ORDER BY re-ranks just those F by exact halfvec cosine. Only the top-k of F is
# returned, so F is the recall/cost knob.
BIT_RERANK_SQL = """
SELECT barcode, seq
FROM (
    SELECT barcode, seq, embedding
    FROM page_embedding
    WHERE NOT (barcode = %(qb)s AND seq = %(qs)s)
    ORDER BY embedding_bit <~> binary_quantize(%(q)s::vector)::bit(512)
    LIMIT %(fetch)s
) c
ORDER BY embedding <=> %(q)s::halfvec
LIMIT %(k)s
"""


def topk(cur, sql, params):
    cur.execute(sql, params)
    return {(b, s) for (b, s) in cur.fetchall()}


def to_literal(vec):
    """float32 unit vector -> pgvector text literal, as search_api.to_literal."""
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"


def clip_text_probes(queries):
    """Encode text queries with the corpus CLIP model -> probe tuples shaped like
    the corpus ones: (qb, qs, q_literal). A text probe is not a corpus row, so
    its exclusion sentinel (qb='', qs=-1) never matches -- the WHERE NOT(...) in
    the shared SQL becomes a harmless no-op. torch/open_clip imported lazily so
    --mode corpus stays dependency-free."""
    import torch
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    model.eval()
    tok = open_clip.get_tokenizer(MODEL_NAME)
    probes = []
    for q in queries:
        with torch.no_grad():
            feats = model.encode_text(tok([q]))
            feats /= feats.norm(dim=-1, keepdim=True)  # cosine space, match corpus
        probes.append(("", -1, to_literal(feats[0].cpu().numpy())))
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=300, help="number of probes")
    ap.add_argument("--k", type=int, default=12, help="top-k to evaluate")
    ap.add_argument("--fetch", type=int, nargs="+",
                    default=[12, 50, 100, 200, 500],
                    help="candidate pool sizes F to sweep for bit+re-rank")
    ap.add_argument("--ef-search", type=int, default=100,
                    help="hnsw.ef_search for the halfvec baseline path")
    ap.add_argument("--mode", choices=["corpus", "text"], default="corpus",
                    help="probe with random corpus vectors, or CLIP text queries")
    ap.add_argument("--text-file",
                    help="text mode: file of queries, one per line "
                         "(default: built-in DEFAULT_QUERIES)")
    args = ap.parse_args()

    # autocommit so session SETs persist cleanly (no open transaction to revert).
    # gt: index scans OFF -> the ORDER BY ... LIMIT becomes a seq scan + top-N
    # sort = EXACT k-NN, our ground truth. ann: index scans ON -> uses the HNSWs.
    gt = psycopg.connect(DSN, autocommit=True)
    gt.execute("SET enable_indexscan = off")
    gt.execute("SET enable_bitmapscan = off")
    ann = psycopg.connect(DSN, autocommit=True)

    # Probe vectors, shaped (qb, qs, q_literal) either way. Corpus: halfvec comes
    # back as its text literal '[v1,v2,...]', reusable directly as a ::vector /
    # ::halfvec cast argument. Text: CLIP-encoded, sentinel (qb,qs) excludes
    # nothing.
    if args.mode == "text":
        queries = DEFAULT_QUERIES
        if args.text_file:
            with open(args.text_file) as fh:
                queries = [ln.strip() for ln in fh if ln.strip()]
        probes = clip_text_probes(queries)
    else:
        with ann.cursor() as cur:
            cur.execute("SELECT barcode, seq, embedding::text "
                        "FROM page_embedding ORDER BY random() LIMIT %s",
                        (args.queries,))
            probes = cur.fetchall()

    rec = {"halfvec_hnsw": []}
    lat = {"halfvec_hnsw": []}
    for f in args.fetch:
        rec[f"bit F={f}"] = []
        lat[f"bit F={f}"] = []

    gt_cur, ann_cur = gt.cursor(), ann.cursor()
    for (qb, qs, q) in probes:
        base = {"qb": qb, "qs": qs, "q": q, "k": args.k}

        truth = topk(gt_cur, HALFVEC_SQL, base)
        if not truth:
            continue

        # current production path: halfvec HNSW at the realistic ef_search.
        ann_cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
        t0 = time.perf_counter()
        hv = topk(ann_cur, HALFVEC_SQL, base)
        lat["halfvec_hnsw"].append((time.perf_counter() - t0) * 1000)
        rec["halfvec_hnsw"].append(len(hv & truth) / len(truth))

        # cheap-box path, swept over candidate pool F.
        for f in args.fetch:
            fetch = max(f, args.k)
            # ef_search must be >= the pool size or the bit HNSW underfills it.
            ann_cur.execute(
                f"SET hnsw.ef_search = {max(fetch, int(args.ef_search))}")
            p = dict(base, fetch=fetch)
            t0 = time.perf_counter()
            br = topk(ann_cur, BIT_RERANK_SQL, p)
            lat[f"bit F={f}"].append((time.perf_counter() - t0) * 1000)
            rec[f"bit F={f}"].append(len(br & truth) / len(truth))

    def line(label):
        r, l = rec[label], lat[label]
        print(f"  {label:14s}  recall@{args.k}={statistics.mean(r):.3f}"
              f"   median={statistics.median(l):5.1f}ms")

    n = len(rec["halfvec_hnsw"])
    print(f"\n{args.mode} probes: {n},  k={args.k},  "
          f"baseline ef_search={args.ef_search}\n")
    print("baseline (what the 128 GB box buys -- full halfvec HNSW in RAM):")
    line("halfvec_hnsw")
    print("\ncheap-box candidate (bit HNSW -> exact halfvec re-rank), by pool F:")
    for f in args.fetch:
        line(f"bit F={f}")
    print("\nrecall is vs EXACT halfvec (seq scan); 1.000 == identical to brute "
          "force.\nhigher F = better recall, slower query. pick the smallest F "
          "that clears your recall bar.")


if __name__ == "__main__":
    main()
