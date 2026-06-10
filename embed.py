"""
Embed every BHL page thumbnail with OpenCLIP and store the vectors in pgvector.

  thumbs:  {THUMBS_DIR}/{BarCode}/{seq}.webp
  output:  page_embedding(barcode, seq, embedding vector(512))

Re-runnable: keys already in the table are skipped, so a second run only
embeds what is missing. Image decode/preprocess is parallelised across CPU
workers; encoding runs on the best available device (MPS/CUDA/CPU).

  python embed.py [--limit N] [--batch-size 256] [--workers 6]
"""
import argparse
import glob
import os
import sys
import time

import config  # sets HF_HUB_OFFLINE; must precede open_clip/huggingface_hub

import numpy as np
import torch
import open_clip
from PIL import Image
import psycopg
from pgvector.psycopg import register_vector


def list_thumbs(thumbs_dir):
    """Yield (barcode, seq, path) for every thumbnail on disk."""
    for path in glob.iglob(os.path.join(thumbs_dir, "*", "*.webp")):
        seq_str = os.path.splitext(os.path.basename(path))[0]
        barcode = os.path.basename(os.path.dirname(path))
        try:
            seq = int(seq_str)
        except ValueError:
            continue
        yield barcode, seq, path


class ThumbDataset(torch.utils.data.Dataset):
    """Returns (barcode, seq, preprocessed_tensor); None on unreadable image."""

    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        barcode, seq, path = self.items[i]
        try:
            img = Image.open(path).convert("RGB")
            return barcode, seq, self.preprocess(img)
        except Exception:
            return barcode, seq, None


def collate(batch):
    keys, tensors = [], []
    for barcode, seq, t in batch:
        if t is None:
            continue
        keys.append((barcode, seq))
        tensors.append(t)
    if not tensors:
        return [], None
    return keys, torch.stack(tensors)


def existing_keys(conn):
    """Set of (barcode, seq) already embedded, for resume."""
    with conn.cursor() as cur:
        cur.execute("SELECT barcode, seq FROM page_embedding")
        return set(cur.fetchall())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap pages (testing)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    dev = config.device()
    print(f"device: {dev}", file=sys.stderr)

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.MODEL_NAME, pretrained=config.PRETRAINED
    )
    model.eval().to(dev)

    conn = psycopg.connect(config.DATABASE_URL)
    register_vector(conn)

    seen = existing_keys(conn)
    print(f"resume: {len(seen)} already embedded", file=sys.stderr)

    items = [
        (b, s, p)
        for (b, s, p) in list_thumbs(config.THUMBS_DIR)
        if (b, s) not in seen
    ]
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} thumbs to embed", file=sys.stderr)
    if not items:
        return

    loader = torch.utils.data.DataLoader(
        ThumbDataset(items, preprocess),
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate,
    )

    done = 0
    t0 = time.time()
    with conn.cursor() as cur, torch.no_grad():
        for keys, tensors in loader:
            if tensors is None:
                continue
            feats = model.encode_image(tensors.to(dev))
            feats = torch.nn.functional.normalize(feats, dim=-1)
            vecs = feats.cpu().numpy().astype(np.float32)
            cur.executemany(
                "INSERT INTO page_embedding (barcode, seq, embedding) "
                "VALUES (%s, %s, %s) ON CONFLICT (barcode, seq) DO NOTHING",
                [(b, s, vecs[i]) for i, (b, s) in enumerate(keys)],
            )
            conn.commit()
            done += len(keys)
            if done % (args.batch_size * 8) < args.batch_size:
                rate = done / max(1e-9, time.time() - t0)
                left = (len(items) - done) / rate if rate else 0
                print(
                    f"{done}/{len(items)}  {rate:.0f} img/s  ~{left/60:.0f}m left",
                    file=sys.stderr,
                )
    print(f"done: embedded {done} in {(time.time()-t0)/60:.1f}m", file=sys.stderr)
    conn.close()


if __name__ == "__main__":
    main()
