"""
Zero-shot label the BHL page images against a set of candidate labels.

CLIP is a dual encoder: image and text land in the same 512-d space. We encode
each candidate label as text, then score every stored page embedding against
them by cosine similarity and softmax across the label set. No re-embedding of
images is needed -- the vectors already live in page_embedding.

  # default label set, first 200 images, top-3 labels each:
  python label.py --limit 200 --html labels.html

  # your own labels, only show images confidently a map:
  python label.py --labels labels.txt --only map --min-score 0.5 --html maps.html

Labels come from --labels FILE (one per line), repeated --label, or a built-in
default set. Each is wrapped in --template before encoding ("a photo of {}").
"""
import argparse
import os
import sys

import config  # sets HF_HUB_OFFLINE; must precede open_clip/huggingface_hub

import numpy as np
import torch
import open_clip
import psycopg
from pgvector.psycopg import register_vector

# Sensible starting point for natural-history page scans. Override with
# --labels / --label for your own taxonomy.
DEFAULT_LABELS = [
    "a page of text",
    "a botanical illustration of a plant",
    "a scientific illustration of an animal",
    "a photograph",
    "a map",
    "a diagram or chart",
    "a portrait of a person",
    "a blank or cover page",
    "a table of numbers",
]

# CLIP's learned logit scale (~100); makes the softmax suitably peaked.
TEMPERATURE = 100.0


def thumb_path(barcode, seq):
    return os.path.join(config.THUMBS_DIR, barcode, f"{seq:04d}.webp")


def load_labels(args):
    if args.labels:
        with open(args.labels) as fh:
            labels = [ln.strip() for ln in fh if ln.strip()]
    elif args.label:
        labels = args.label
    else:
        labels = DEFAULT_LABELS
    if not labels:
        sys.exit("no labels given")
    return labels


def encode_labels(model, dev, labels, template):
    """Return a (n_labels, 512) normalized text-feature matrix."""
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    prompts = [template.format(l) for l in labels]
    with torch.no_grad():
        toks = tokenizer(prompts).to(dev)
        feats = model.encode_text(toks)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.cpu().numpy().astype("float32")


def fetch_embeddings(conn, limit):
    """Yield (barcode, seq, vec) for stored page embeddings."""
    sql = "SELECT barcode, seq, embedding FROM page_embedding ORDER BY barcode, seq"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def write_html(path, labels, rows, top):
    """rows: list of (barcode, seq, [(label, score), ...] sorted desc)."""
    cells = []
    for barcode, seq, scored in rows:
        lines = "<br>".join(
            f"{lab} <b>{sc:.2f}</b>" for lab, sc in scored[:top]
        )
        cells.append(
            f'<figure><img src="file://{thumb_path(barcode, seq)}">'
            f"<figcaption>{barcode}/{seq}<br>{lines}</figcaption></figure>"
        )
    html = f"""<!doctype html><meta charset=utf-8>
<style>body{{font:13px sans-serif;margin:1rem}}
.grid{{display:flex;flex-wrap:wrap;gap:8px}}
figure{{margin:0;width:180px}} img{{width:180px;border:1px solid #ccc}}
figcaption{{text-align:center;color:#444}}</style>
<h3>Labels: {", ".join(labels)}</h3>
<h3>{len(rows)} images (top {top} labels each)</h3>
<div class="grid">{''.join(cells)}</div>"""
    with open(path, "w") as fh:
        fh.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help="file with one candidate label per line")
    ap.add_argument("--label", action="append", help="a candidate label (repeatable)")
    ap.add_argument("--template", default="a photo of {}",
                    help="prompt wrapper; '{}' is the label")
    ap.add_argument("--limit", type=int, default=200, help="cap images (0 = all)")
    ap.add_argument("--top", type=int, default=3, help="labels to show per image")
    ap.add_argument("--only", help="keep only images whose top label contains this text")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop images whose top score is below this")
    ap.add_argument("--no-softmax", action="store_true",
                    help="score each label by raw cosine similarity instead of a "
                         "softmax probability across the set (lets labels be "
                         "thresholded independently; typical CLIP cosines ~0.2-0.35)")
    ap.add_argument("--html", help="write a contact-sheet HTML to this path")
    args = ap.parse_args()

    dev = config.device()
    model, _, _ = open_clip.create_model_and_transforms(
        config.MODEL_NAME, pretrained=config.PRETRAINED
    )
    model.eval().to(dev)

    labels = load_labels(args)
    text_feats = encode_labels(model, dev, labels, args.template)  # (L, 512)

    conn = psycopg.connect(config.DATABASE_URL)
    register_vector(conn)
    records = fetch_embeddings(conn, args.limit)
    conn.close()
    if not records:
        sys.exit("no embeddings found in page_embedding")

    # Stack image vectors and score the whole batch at once: (N,512)@(512,L).
    # Both sides are unit-normalized, so this dot product is cosine similarity.
    img_mat = np.stack([vec for (_, _, vec) in records]).astype("float32")
    sims = img_mat @ text_feats.T
    if args.no_softmax:
        # Raw cosine: each label stands alone, scores don't sum to 1.
        scores = sims
    else:
        # Softmax across labels, per image (scores sum to 1).
        logits = TEMPERATURE * sims
        logits -= logits.max(axis=1, keepdims=True)
        scores = np.exp(logits)
        scores /= scores.sum(axis=1, keepdims=True)

    rows = []
    for (barcode, seq, _), p in zip(records, scores):
        order = np.argsort(-p)
        scored = [(labels[i], float(p[i])) for i in order]
        top_label, top_score = scored[0]
        if args.only and args.only.lower() not in top_label.lower():
            continue
        if top_score < args.min_score:
            continue
        rows.append((barcode, seq, scored))

    for barcode, seq, scored in rows:
        lab, sc = scored[0]
        print(f"{sc:.4f}  {lab:35s}  {barcode}/{seq}")

    if args.html:
        write_html(args.html, labels, rows, args.top)
        print(f"\nwrote {args.html} ({len(rows)} images)", file=sys.stderr)


if __name__ == "__main__":
    main()
