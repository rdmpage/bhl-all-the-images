"""
Hetzner-side search service: the one box that holds the vectors also runs CLIP.

A query (text, or an uploaded image) is encoded by the *same* OpenCLIP model
used to embed the corpus, then matched against page_embedding with pgvector's
cosine operator. Results carry public S3 webp URLs so a downstream demo (PHP on
Heroku, a static page, whatever) is pure presentation -- it renders <img> tags
and never touches Postgres or torch. This is the deliberate split: the model +
ANN live next to the data; the UI just calls JSON over HTTPS.

    pip install -r requirements-api.txt
    export DATABASE_URL=postgresql:///bhl
    uvicorn search_api:app --host 127.0.0.1 --port 8000

    # text query
    curl 'http://127.0.0.1:8000/search?q=a+colour+plate+of+birds&k=12'
    # image query
    curl -F file=@some_page.jpg 'http://127.0.0.1:8000/search?k=12'

The CLIP weights download once from HuggingFace on first start (~600 MB) and are
cached; set HF_HUB_OFFLINE=1 afterwards (or bake them, as the AWS Dockerfile
does) for a no-network start.

halfvec note: the stored column is halfvec(512); the query vector is sent as a
pgvector text literal and cast `%s::halfvec`, so no pgvector psycopg adapter is
needed on the read path.
"""
import io
import os

import numpy as np
import torch
import open_clip
from PIL import Image
from fastapi import (FastAPI, UploadFile, File, Query, HTTPException, Header,
                     Depends)
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import ConnectionPool

# --- config (kept identical to aws/embed_s3.py so query == corpus space) ------
MODEL_NAME = os.environ.get("BHL_CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("BHL_CLIP_PRETRAINED", "laion2b_s34b_b79k")
DSN = os.environ.get("DATABASE_URL", "postgresql:///bhl")
# recall/speed knob. Default 300: at 100, recall@12 vs exact on real text queries
# measured only ~0.875 (the corpus-vector geometry held up far better, ~0.985) --
# text queries land in sparse regions where the HNSW misses more, and a higher
# ef_search buys most of that back for a few ms. Raise further if recall matters
# more than latency. (See db/bq_eval.sql + hetzner/bq_recall_eval.py, 2026-06-29.)
EF_SEARCH = int(os.environ.get("BHL_HNSW_EF_SEARCH", "300"))
API_KEY = os.environ.get("BHL_SEARCH_KEY", "")  # if set, callers must send it

# Public source bucket: the same web/ webp derivatives the embedder read. These
# are world-readable over plain HTTPS, so the UI can <img src> them directly.
S3_BASE = os.environ.get(
    "BHL_S3_BASE", "https://bhl-open-data.s3.us-east-2.amazonaws.com")

# --- model: loaded once at process start, reused for every request ------------
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model, _, _preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED)
_model.eval().to(_device)
_tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# A small pool: text/image encode is the slow part (tens of ms on CPU), the SQL
# is sub-ms against an in-RAM HNSW, so a couple of connections is plenty.
_pool = ConnectionPool(DSN, min_size=1, max_size=4, open=True)

app = FastAPI(title="BHL image search (dry run)")
# Open CORS so a browser/Heroku-hosted demo on another origin can call this.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def require_key(x_api_key: str = Header(default="")):
    """Gate the search endpoints on the X-API-Key header. No-op (open) when
    BHL_SEARCH_KEY is unset, so dev/local stays frictionless."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="missing or invalid API key")


def image_url(barcode, seq, size="medium"):
    """Reconstruct the public webp URL for a page.

    Filenames are 4-digit zero-padded (web/<bc>/<bc>_0007_medium.webp) but seq
    is stored as a bare int, so re-pad here. CAVEAT: a handful of items exceed
    9999 pages and use 5-digit padding -- fine for the dry-run sample; for the
    full run, persist the original key (or padding width) in the parquet rather
    than re-deriving it. Sizes: thumb, small, medium, large, full."""
    return f"{S3_BASE}/web/{barcode}/{barcode}_{int(seq):04d}_{size}.webp"


def to_literal(vec):
    """float32 unit vector -> pgvector text literal '[v1,v2,...]'."""
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"


def encode_text(text):
    with torch.no_grad():
        feats = _model.encode_text(_tokenizer([text]).to(_device))
        feats /= feats.norm(dim=-1, keepdim=True)  # cosine space; match corpus
    return feats[0].cpu().numpy().astype(np.float32)


def encode_image(data):
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not decode uploaded image")
    with torch.no_grad():
        feats = _model.encode_image(_preprocess(im).unsqueeze(0).to(_device))
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype(np.float32)


def search(qvec, k, size):
    lit = to_literal(qvec)
    with _pool.connection() as conn:
        with conn.cursor() as cur:
            # SET takes no bind parameters (Postgres grammar wants a literal),
            # so interpolate the int directly -- EF_SEARCH is int(os.environ...).
            cur.execute(f"SET hnsw.ef_search = {int(EF_SEARCH)}")
            cur.execute(
                "SELECT barcode, seq, 1 - (embedding <=> %s::halfvec) AS score "
                "FROM page_embedding "
                "ORDER BY embedding <=> %s::halfvec LIMIT %s",
                (lit, lit, k),
            )
            rows = cur.fetchall()
    return [
        {"barcode": b, "seq": s, "score": round(float(score), 4),
         "thumb_url": image_url(b, s, "thumb"),
         "image_url": image_url(b, s, size)}
        for (b, s, score) in rows
    ]


@app.get("/healthz")
def healthz():
    with _pool.connection() as conn:
        n = conn.execute("SELECT count(*) FROM page_embedding").fetchone()[0]
    return {"ok": True, "model": f"{MODEL_NAME}/{PRETRAINED}",
            "device": _device, "vectors": n}


@app.get("/search", dependencies=[Depends(require_key)])
def search_text(q: str = Query(..., description="text query"),
                k: int = 12, size: str = "medium"):
    return {"query": q, "k": k, "results": search(encode_text(q), k, size)}


@app.post("/search", dependencies=[Depends(require_key)])
async def search_image(file: UploadFile = File(...),
                       k: int = 12, size: str = "medium"):
    qvec = encode_image(await file.read())
    return {"query": f"image:{file.filename}", "k": k,
            "results": search(qvec, k, size)}
