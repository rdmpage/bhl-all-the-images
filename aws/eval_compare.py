"""
Three-way visual side-by-side: for each standard query, show the top-k hits from
each model as a row of thumbnails, framed green (relevant) / red (not) by the
hand labels, with cosine + label. Writes one HTML page.

    python aws/eval_compare.py --k 5 --html tier0/eval/compare.html
"""
import argparse
import io
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import open_clip
from PIL import Image
import psycopg
from pgvector.psycopg import register_vector
import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "bhl-open-data"
REGION = "us-east-2"
S3 = boto3.client("s3", region_name=REGION,
                  config=Config(signature_version=UNSIGNED, max_pool_connections=32))

MODELS = [
    ("ViT-B/32", "ViT-B-32", "laion2b_s34b_b79k", "postgresql:///bhl_tier0_b"),
    ("ViT-B/16", "ViT-B-16", "laion2b_s34b_b88k", "postgresql:///bhl_tier0_b16"),
    ("ViT-L/14", "ViT-L-14", "laion2b_s32b_b82k", "postgresql:///bhl_tier0_l"),
]
QUERY_RELEVANT = {
    "a geographic map": {"map"},
    "a map of a continent": {"map"},
    "a colour plate": {"plate"},
    "a colour plate of birds": {"plate"},
    "a portrait of a person": {"portrait"},
    "a photograph of people": {"photo"},
}


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def reduced_jp2(data, target=200):
    short = min(Image.open(io.BytesIO(data)).size)
    r = 0
    while (short >> (r + 1)) >= target and r < 5:
        r += 1
    for rr in range(r, -1, -1):
        try:
            im = Image.open(io.BytesIO(data))
            if rr:
                im.reduce = rr
            return im.convert("RGB")
        except Exception:
            if rr == 0:
                raise


def thumb(barcode, seq, outdir):
    path = os.path.join(outdir, f"{barcode}_{seq}.png")
    if os.path.exists(path):
        return os.path.basename(path)
    for w in (4, 5, 6, 3, 0):
        s = str(seq) if w == 0 else str(seq).zfill(w)
        key = f"images/{barcode}/{barcode}_{s}.jp2"
        try:
            im = reduced_jp2(S3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
            im.thumbnail((200, 200))
            im.save(path)
            return os.path.basename(path)
        except Exception:
            pass
    return None


def load_labels(path):
    labels = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            labels[(p[0], int(p[1]))] = p[2]
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--labels", default="tier0/eval/labels.tsv")
    ap.add_argument("--html", default="tier0/eval/compare.html")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    tdir = os.path.join(os.path.dirname(args.html), "compare_thumbs")
    os.makedirs(tdir, exist_ok=True)
    dev = device()

    # results[query][model_disp] = (p@k, [(b,s,score,label,thumbfile), ...])
    results = {q: {} for q in QUERY_RELEVANT}
    for disp, name, pretrained, dsn in MODELS:
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        model.eval().to(dev)
        tok = open_clip.get_tokenizer(name)
        conn = psycopg.connect(dsn)
        register_vector(conn)
        for q, relevant in QUERY_RELEVANT.items():
            with torch.no_grad():
                f = model.encode_text(tok([q]).to(dev))
                f = torch.nn.functional.normalize(f, dim=-1)
            v = f.cpu().numpy().astype(np.float32)[0]
            with conn.cursor() as cur:
                cur.execute("SELECT barcode, seq, 1-(embedding<=>%s) FROM page_embedding "
                            "ORDER BY embedding <=> %s LIMIT %s", (v, v, args.k))
                hits = cur.fetchall()
            cells, nrel = [], 0
            for b, s, sc in hits:
                lab = labels.get((b, s), "?")
                if lab in relevant:
                    nrel += 1
                cells.append((b, s, float(sc), lab, thumb(b, s, tdir)))
            results[q][disp] = (nrel / args.k, cells)
        conn.close()

    relset = {q: r for q, r in QUERY_RELEVANT.items()}
    blocks = []
    for q in QUERY_RELEVANT:
        rows = []
        for disp, *_ in MODELS:
            p, cells = results[q][disp]
            cs = []
            for b, s, sc, lab, tf in cells:
                ok = lab in relset[q]
                color = "#2a2" if ok else ("#999" if lab in ("blank", "?") else "#c33")
                img = (f'<img src="compare_thumbs/{tf}">' if tf else
                       '<div class="missing">no img</div>')
                cs.append(
                    f'<figure style="border-color:{color}">{img}'
                    f'<figcaption>{sc:.3f}<br><b>{lab}</b><br>{b}/{s}</figcaption></figure>')
            rows.append(
                f'<div class="row"><div class="ml">{disp}<br>'
                f'<span class="p">p@{args.k}={p:.2f}</span></div>'
                f'<div class="cells">{"".join(cs)}</div></div>')
        blocks.append(f'<section><h2>{q} '
                      f'<small>(relevant = {"/".join(relset[q])})</small></h2>'
                      f'{"".join(rows)}</section>')

    html = f"""<!doctype html><meta charset=utf-8>
<style>body{{font:13px -apple-system,sans-serif;margin:1.2rem;background:#fafafa}}
h2{{margin:1.4rem 0 .4rem;border-top:2px solid #ddd;padding-top:.8rem}}
small{{color:#888;font-weight:normal}}
.row{{display:flex;align-items:center;gap:10px;margin:4px 0}}
.ml{{width:78px;flex:none;font-weight:bold;text-align:right;color:#333}}
.ml .p{{font-weight:normal;color:#666}}
.cells{{display:flex;gap:6px}}
figure{{margin:0;width:120px;border:3px solid #999;border-radius:3px;overflow:hidden;background:#fff}}
img{{width:120px;height:150px;object-fit:cover;display:block}}
.missing{{width:120px;height:150px;display:flex;align-items:center;justify-content:center;color:#aaa}}
figcaption{{font-size:11px;text-align:center;color:#444;padding:2px}}
</style>
<h1>BHL image-query: B/32 vs B/16 vs ViT-L &mdash; top-{args.k}</h1>
<p>Green = relevant (matches hand label), red = wrong, grey = blank/unlabelled.</p>
{"".join(blocks)}"""
    with open(args.html, "w") as fh:
        fh.write(html)
    print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
