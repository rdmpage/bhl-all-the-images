# Worker image for the BHL embedding Batch array job.
#   docker build -t bhl-embed aws/
#   (push to ECR, reference from the Batch job definition)
FROM python:3.11-slim

# libopenjp2 backs Pillow's JPEG-2000 decode (most Pillow wheels bundle it,
# but install it explicitly so reduced-level decode is always available).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Bake the CLIP weights into the image so 1000 array tasks don't each pull them
# from the HF hub on cold start.
RUN python -c "import open_clip; \
open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"

COPY embed_s3.py .
ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["python", "embed_s3.py"]
