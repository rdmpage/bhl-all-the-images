"""
Build a labelled-eval pool for precision@k.

Pools the top-k pages both models return for the standard queries, adds some
random background pages, dedupes, fetches thumbnails, and lays them out as
numbered montage grids for manual labelling.

    python aws/eval_pool.py            # writes tier0/eval/{pool.tsv,montage_NN.png}

Then label each montage cell by its printed yellow index into a dict and emit
tier0/eval/labels.tsv (barcode<TAB>seq<TAB>label); score with eval_score.py.

Pooling the retrieved sets is the right scope for *precision*@k: you only need
to judge what gets retrieved. (Recall would need exhaustive labels.)
"""
import argparse
import io
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import open_clip
from PIL import Image, ImageDraw, ImageFont
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
    ("ViT-B-32", "laion2b_s34b_b79k", "postgresql:///bhl_tier0_b"),
    ("ViT-L-14", "laion2b_s32b_b82k", "postgresql:///bhl_tier0_l"),
]
QUERIES = [
    "a geographic map", "a map of a continent",
    "a colour plate", "a colour plate of birds",
    "a portrait of a person", "a photograph of people",
]


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def reduced_jp2(data, target=256):
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


def fetch(barcode, seq):
    for w in (4, 5, 6, 3, 0):
        s = str(seq) if w == 0 else str(seq).zfill(w)
        key = f"images/{barcode}/{barcode}_{s}.jp2"
        try:
            return reduced_jp2(S3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
        except Exception:
            pass
    return None


def pool_candidates(k):
    cand = set()
    dev = device()
    for model_name, pretrained, dsn in MODELS:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        model.eval().to(dev)
        tok = open_clip.get_tokenizer(model_name)
        conn = psycopg.connect(dsn)
        register_vector(conn)
        for q in QUERIES:
            with torch.no_grad():
                f = model.encode_text(tok([q]).to(dev))
                f = torch.nn.functional.normalize(f, dim=-1)
            v = f.cpu().numpy().astype(np.float32)[0]
            with conn.cursor() as cur:
                cur.execute("SELECT barcode, seq FROM page_embedding "
                            "ORDER BY embedding <=> %s LIMIT %s", (v, k))
                cand.update(cur.fetchall())
        conn.close()
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="top-k pooled per query per model")
    ap.add_argument("--random", type=int, default=30, help="random background pages")
    ap.add_argument("--cap", type=int, default=100, help="max pool size to label")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cell", type=int, default=240)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--manifest", default="tier0/wide.tsv")
    ap.add_argument("--out", default="tier0/eval")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cand = pool_candidates(args.k)

    bg = [l.rstrip("\n").split("\t") for l in open(args.manifest)]
    for row in rng.sample(bg, min(args.random, len(bg))):
        cand.add((row[1], int(row[2])))

    cand = sorted(cand)
    if len(cand) > args.cap:
        cand = sorted(rng.sample(cand, args.cap))
    print(f"pool: {len(cand)} candidate pages; fetching thumbnails...")

    os.makedirs(args.out, exist_ok=True)
    font = ImageFont.load_default(size=26)
    cells = []
    for b, s in cand:
        im = fetch(b, s)
        if im is None:
            continue
        im.thumbnail((args.cell, args.cell))
        canvas = Image.new("RGB", (args.cell, args.cell), (255, 255, 255))
        canvas.paste(im, ((args.cell - im.width) // 2, (args.cell - im.height) // 2))
        cells.append((b, s, canvas))

    with open(os.path.join(args.out, "pool.tsv"), "w") as pf:
        for i, (b, s, _) in enumerate(cells):
            pf.write(f"{i}\t{b}\t{s}\n")

    per = args.cols * args.rows
    n_mont = (len(cells) + per - 1) // per
    for mi in range(n_mont):
        chunk = cells[mi * per:(mi + 1) * per]
        mont = Image.new("RGB", (args.cols * args.cell, args.rows * args.cell),
                         (40, 40, 40))
        d = ImageDraw.Draw(mont)
        for j, (b, s, c) in enumerate(chunk):
            gi = mi * per + j
            x = (j % args.cols) * args.cell
            y = (j // args.cols) * args.cell
            mont.paste(c, (x, y))
            d.rectangle([x, y, x + 38, y + 24], fill=(0, 0, 0))
            d.text((x + 3, y + 1), str(gi), fill=(255, 255, 0), font=font)
        mont.save(os.path.join(args.out, f"montage_{mi:02d}.png"))

    print(f"{len(cells)} pages -> {n_mont} montages in {args.out}/")
    print("Label each cell by its yellow index, then write labels.tsv")


if __name__ == "__main__":
    main()
