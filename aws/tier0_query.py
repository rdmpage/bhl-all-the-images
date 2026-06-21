"""
Tier-0 trial query: rank the locally-embedded BHL sample by CLIP similarity and
render an HTML contact sheet. Standalone (no repo imports) so it runs as-is.

    python aws/tier0_query.py --dsn postgresql:///bhl_tier0 --text "a map" -k 12 --html tier0/map.html
    python aws/tier0_query.py --dsn postgresql:///bhl_tier0 --page BARCODE/7 -k 12 --html tier0/like.html

Top hits are re-fetched from the public bucket, reduced-decoded to PNG
thumbnails next to the HTML, so you can eyeball retrieval quality directly.
"""
import argparse
import io
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import open_clip
from PIL import Image
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import psycopg
from pgvector.psycopg import register_vector

BUCKET = "bhl-open-data"
REGION = "us-east-2"
MODEL = os.environ.get("BHL_CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("BHL_CLIP_PRETRAINED", "laion2b_s34b_b79k")

S3 = boto3.client("s3", region_name=REGION,
                  config=Config(signature_version=UNSIGNED, max_pool_connections=32))


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def reduced_jp2(data, target=224):
    # Fall back to shallower reduce levels: some JP2s carry fewer resolution
    # levels than requested and would otherwise raise "broken data stream".
    short = min(Image.open(io.BytesIO(data)).size)
    rlevel = 0
    while (short >> (rlevel + 1)) >= target and rlevel < 5:
        rlevel += 1
    for r in range(rlevel, -1, -1):
        try:
            im = Image.open(io.BytesIO(data))
            if r:
                im.reduce = r
            return im.convert("RGB")
        except Exception:
            if r == 0:
                raise


def fetch_img(barcode, seq):
    """Fetch a page image, tolerating sequence zero-pad width variation."""
    last = None
    for w in (4, 5, 6, 3, 0):
        s = str(seq) if w == 0 else str(seq).zfill(w)
        key = f"images/{barcode}/{barcode}_{s}.jp2"
        try:
            data = S3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            return reduced_jp2(data)
        except Exception as e:
            last = e
    raise last


def write_html(path, label, src_png, rows, thumb_rel):
    cells = []
    for b, s, score, png in rows:
        rel = os.path.join(thumb_rel, png)
        cells.append(
            f'<figure><img src="{rel}">'
            f"<figcaption>{b}/{s}<br>{score:.3f}</figcaption></figure>"
        )
    q = (f'<img src="{os.path.join(thumb_rel, src_png)}">' if src_png
         else f'<span class="text">{label}</span>')
    html = f"""<!doctype html><meta charset=utf-8>
<style>body{{font:13px sans-serif;margin:1rem}}
.grid{{display:flex;flex-wrap:wrap;gap:8px}}
figure{{margin:0;width:170px}} img{{width:170px;border:1px solid #ccc}}
figcaption{{text-align:center;color:#444}}
.q img{{width:240px;border:3px solid #c00}}
.q .text{{font-size:1.4rem;padding:1rem;border:3px solid #c00;display:inline-block}}</style>
<h3>Query: {label}</h3>
<div class="q">{q}</div>
<h3>Top {len(rows)} matches</h3>
<div class="grid">{''.join(cells)}</div>"""
    with open(path, "w") as fh:
        fh.write(html)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--page", help="in-set page as BARCODE/SEQ")
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL",
                                                    "postgresql:///bhl_tier0"))
    ap.add_argument("-k", type=int, default=12)
    ap.add_argument("--html", required=True)
    args = ap.parse_args()

    dev = device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, pretrained=PRETRAINED)
    model.eval().to(dev)

    conn = psycopg.connect(args.dsn)
    register_vector(conn)

    self_key = None
    src_png = None
    thumb_dir = os.path.join(os.path.dirname(args.html) or ".", "tier0_thumbs")
    os.makedirs(thumb_dir, exist_ok=True)

    if args.text:
        tok = open_clip.get_tokenizer(MODEL)
        with torch.no_grad():
            feat = model.encode_text(tok([args.text]).to(dev))
            feat = torch.nn.functional.normalize(feat, dim=-1)
        qvec = feat.cpu().numpy().astype(np.float32)[0]
        label = args.text
    else:
        barcode, seq = args.page.rsplit("/", 1)
        seq = int(seq)
        self_key = (barcode, seq)
        with conn.cursor() as cur:
            cur.execute("SELECT embedding FROM page_embedding "
                        "WHERE barcode=%s AND seq=%s", (barcode, seq))
            row = cur.fetchone()
        if not row:
            sys.exit(f"page {args.page} not in {args.dsn}")
        qvec = np.asarray(row[0], dtype=np.float32)
        label = args.page
        img = fetch_img(barcode, seq)
        src_png = f"query_{barcode}_{seq}.png"
        img.save(os.path.join(thumb_dir, src_png))

    with conn.cursor() as cur:
        cur.execute(
            "SELECT barcode, seq, 1 - (embedding <=> %s) AS score "
            "FROM page_embedding ORDER BY embedding <=> %s LIMIT %s",
            (qvec, qvec, args.k + 1),
        )
        hits = cur.fetchall()
    conn.close()

    hits = [(b, s, sc) for (b, s, sc) in hits if (b, s) != self_key][:args.k]

    rows = []
    for b, s, sc in hits:
        png = f"{b}_{s}.png"
        try:
            fetch_img(b, s).save(os.path.join(thumb_dir, png))
        except Exception as e:
            print(f"  ! fetch {b}/{s}: {e}", file=sys.stderr)
            continue
        rows.append((b, s, sc, png))
        print(f"{sc:.4f}  {b}/{s}")

    write_html(args.html, label, src_png, rows, "tier0_thumbs")
    print(f"\nwrote {args.html}", file=sys.stderr)


if __name__ == "__main__":
    main()
