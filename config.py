"""Shared configuration for the BHL image-embedding pipeline.

Import this module *before* open_clip so the HF_HUB_OFFLINE setting below takes
effect (huggingface_hub reads it at import time).
"""
import os

# The CLIP weights are cached on disk after the first run, so default to offline
# mode: it skips a network round-trip to the HF Hub on every load and silences
# the "sending unauthenticated requests" warning. Override with HF_HUB_OFFLINE=0
# to allow downloads, e.g. when switching to a model you have not cached yet.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

# Where the page thumbnails live. These are produced/cached by the sibling
# viewer repo (bhl-all-the-pages) as cache/thumbs/{BarCode}/{seq}.webp.
THUMBS_DIR = os.environ.get(
    "BHL_THUMBS_DIR",
    os.path.expanduser("~/Sites/bhl-all-the-pages/cache/thumbs"),
)

# Postgres (pgvector). Default DB is a dedicated 'bhl' database on the local
# Postgres.app instance, owned by the current user.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql:///bhl")

# OpenCLIP model. ViT-B/32 -> 512-d. Changing these means re-embedding the set.
MODEL_NAME = os.environ.get("BHL_CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("BHL_CLIP_PRETRAINED", "laion2b_s34b_b79k")
EMBED_DIM = 512


def device() -> str:
    """Best available torch device. MPS if usable (needs macOS 14+ on recent
    torch), otherwise CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
