"""
Score precision@k per query per model against a hand-labelled eval set.

    python aws/eval_score.py --labels tier0/eval/labels.tsv --k 5

Reads labels.tsv (barcode<TAB>seq<TAB>label), runs each standard query against
each model's DB, and reports precision@k = (relevant in top-k) / k, where a hit
is relevant if its page label is in the query's relevant-label set. Pages
retrieved but missing from labels.tsv are counted as non-relevant and flagged
(keep the label pool == the retrieved pool so this stays ~0).

Label vocabulary: blank | text | map | plate | portrait | photo | other
"""
import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import open_clip
import psycopg
from pgvector.psycopg import register_vector

MODELS = [
    ("ViT-B/32", "ViT-B-32", "laion2b_s34b_b79k", "postgresql:///bhl_tier0_b"),
    ("ViT-B/16", "ViT-B-16", "laion2b_s34b_b88k", "postgresql:///bhl_tier0_b16"),
    ("ViT-L/14", "ViT-L-14", "laion2b_s32b_b82k", "postgresql:///bhl_tier0_l"),
]
# query -> set of page labels that count as relevant. The bird query measures
# plate-ness (the coarse label set carries no subject), noted in the report.
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


def load_labels(path):
    labels = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3 and parts[2].strip():
            labels[(parts[0], int(parts[1]))] = parts[2].strip().lower()
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="tier0/eval/labels.tsv")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    labels = load_labels(args.labels)
    print(f"{len(labels)} labelled pages\n")
    dev = device()
    unlabelled = 0

    for disp, model_name, pretrained, dsn in MODELS:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        model.eval().to(dev)
        tok = open_clip.get_tokenizer(model_name)
        conn = psycopg.connect(dsn)
        register_vector(conn)
        print(f"=== {disp} ===")
        precisions = []
        for q, relevant in QUERY_RELEVANT.items():
            with torch.no_grad():
                f = model.encode_text(tok([q]).to(dev))
                f = torch.nn.functional.normalize(f, dim=-1)
            v = f.cpu().numpy().astype(np.float32)[0]
            with conn.cursor() as cur:
                cur.execute("SELECT barcode, seq FROM page_embedding "
                            "ORDER BY embedding <=> %s LIMIT %s", (v, args.k))
                hits = cur.fetchall()
            tags = []
            hit_relevant = 0
            for b, s in hits:
                lab = labels.get((b, s))
                if lab is None:
                    unlabelled += 1
                    tags.append("?")
                    continue
                tags.append(lab)
                if lab in relevant:
                    hit_relevant += 1
            p = hit_relevant / args.k
            precisions.append(p)
            print(f"  p@{args.k}={p:.2f}  {q:<26} [{' '.join(tags)}]")
        print(f"  mean p@{args.k} = {sum(precisions) / len(precisions):.3f}\n")
        conn.close()

    if unlabelled:
        print(f"note: {unlabelled} retrieved pages were unlabelled "
              f"(counted as non-relevant)")


if __name__ == "__main__":
    main()
