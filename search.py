"""
Search the BHL page-image set by visual similarity.

The query is embedded with the same OpenCLIP model and matched against the
pgvector store by cosine distance (exact KNN).

  # an external image (organism photo, unseen BHL page, anything):
  python search.py --image /path/to/query.jpg -k 20 --html results.html

  # an in-set page ("images like this one"):
  python search.py --page MemoirsOfNati13Nati/7 -k 20 --html results.html

  # a text phrase ("pages that have maps / portraits / beetles"):
  python search.py --text "a map" -k 30 --html maps.html

CLIP embeds text into the same space as images, so a phrase can rank pages
directly -- no labels to precompute.

--html writes a contact sheet (query + ranked thumbnails) for eyeballing.
"""
import argparse
import os
import sys

import config  # sets HF_HUB_OFFLINE; must precede open_clip/huggingface_hub

import torch
import open_clip
from PIL import Image
import psycopg
from pgvector.psycopg import register_vector


def thumb_path(barcode, seq):
    return os.path.join(config.THUMBS_DIR, barcode, f"{seq:04d}.webp")


def embed_query(model, preprocess, dev, img):
    with torch.no_grad():
        t = preprocess(img).unsqueeze(0).to(dev)
        feat = model.encode_image(t)
        feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat.cpu().numpy().astype("float32")[0]


def embed_text(model, dev, text):
    """Embed a text phrase into the shared image/text space."""
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    with torch.no_grad():
        toks = tokenizer([text]).to(dev)
        feat = model.encode_text(toks)
        feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat.cpu().numpy().astype("float32")[0]


def write_html(path, query_label, query_src, rows):
    cells = [
        f'<figure><img src="file://{thumb_path(b, s)}">'
        f"<figcaption>{b}/{s}<br>{score:.3f}</figcaption></figure>"
        for b, s, score in rows
    ]
    html = f"""<!doctype html><meta charset=utf-8>
<style>body{{font:13px sans-serif;margin:1rem}}
.grid{{display:flex;flex-wrap:wrap;gap:8px}}
figure{{margin:0;width:160px}} img{{width:160px;border:1px solid #ccc}}
figcaption{{text-align:center;color:#444}}
.q img{{width:240px;border:3px solid #c00}}
.q .text{{font-size:1.4rem;padding:1rem;border:3px solid #c00;display:inline-block}}</style>
<h3>Query: {query_label}</h3>
<div class="q">{f'<img src="{query_src}">' if query_src else f'<span class="text">{query_label}</span>'}</div>
<h3>Top {len(rows)} matches</h3>
<div class="grid">{''.join(cells)}</div>"""
    with open(path, "w") as fh:
        fh.write(html)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="path to a query image")
    g.add_argument("--page", help="in-set page as BARCODE/SEQ")
    g.add_argument("--text", help="a text phrase, e.g. 'a map' or 'a portrait'")
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--html", help="write a contact-sheet HTML to this path")
    args = ap.parse_args()

    dev = config.device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.MODEL_NAME, pretrained=config.PRETRAINED
    )
    model.eval().to(dev)

    # Resolve the query vector and a human label / displayable source.
    self_key = None
    if args.text:
        qvec = embed_text(model, dev, args.text)
        query_label, query_src = args.text, None
    elif args.image:
        img = Image.open(args.image).convert("RGB")
        qvec = embed_query(model, preprocess, dev, img)
        query_label, query_src = args.image, f"file://{os.path.abspath(args.image)}"
    else:
        barcode, seq = args.page.rsplit("/", 1)
        seq = int(seq)
        self_key = (barcode, seq)
        p = thumb_path(barcode, seq)
        img = Image.open(p).convert("RGB")
        qvec = embed_query(model, preprocess, dev, img)
        query_label, query_src = args.page, f"file://{p}"

    conn = psycopg.connect(config.DATABASE_URL)
    register_vector(conn)
    # Fetch one extra so we can drop the query page itself when searching in-set.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT barcode, seq, 1 - (embedding <=> %s) AS score "
            "FROM page_embedding ORDER BY embedding <=> %s LIMIT %s",
            (qvec, qvec, args.k + 1),
        )
        rows = cur.fetchall()
    conn.close()

    rows = [(b, s, sc) for (b, s, sc) in rows if (b, s) != self_key][: args.k]

    for b, s, sc in rows:
        print(f"{sc:.4f}  {b}/{s}  {thumb_path(b, s)}")

    if args.html:
        write_html(args.html, query_label, query_src, rows)
        print(f"\nwrote {args.html}", file=sys.stderr)


if __name__ == "__main__":
    main()
