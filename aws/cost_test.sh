#!/usr/bin/env bash
#
# Turnkey AWS cost-measurement run for the BHL embedding pipeline.
#
# Run this ON a fresh Amazon Linux 2023 EC2 instance, in us-east-2 (same region
# as s3://bhl-open-data, so image reads are free and the throughput you measure
# is the real in-region rate). It installs deps, samples a bounded subset,
# times the embed with ViT-B/32 (the PoC model), and prints true cost figures:
# $/1000 pages and the extrapolated full-corpus cost.
#
#   git clone <repo> && cd <repo>
#   bash aws/cost_test.sh                 # auto-detects instance type for pricing
#   bash aws/cost_test.sh --pages 20000   # bigger sample = steadier number
#   bash aws/cost_test.sh --rate 0.357    # override the on-demand $/hr
#
# No AWS credentials needed: the source bucket is read unsigned and output
# Parquet is written to local disk. Reads nothing of yours, writes nothing to S3.
#
# !! When it finishes: TERMINATE THE INSTANCE. A forgotten box is the only thing
#    here that quietly runs up a bill.
set -euo pipefail

# ---- config (override via flags or env) -------------------------------------
TOTAL_PAGES=${TOTAL_PAGES:-63000000}   # full BHL, for extrapolation
TARGET=${TARGET:-10000}                # pages to sample for the measurement
PAGES_PER_ITEM=${PAGES_PER_ITEM:-25}
HOURLY_RATE=${HOURLY_RATE:-}           # on-demand $/hr; auto-detected if empty
SPOT_FACTOR=${SPOT_FACTOR:-0.35}       # rough Spot discount for the production estimate
SKIP_SETUP=${SKIP_SETUP:-0}
FORCE=${FORCE:-0}

while [ $# -gt 0 ]; do
  case "$1" in
    --pages) TARGET="$2"; shift 2;;
    --rate) HOURLY_RATE="$2"; shift 2;;
    --total) TOTAL_PAGES="$2"; shift 2;;
    --skip-setup) SKIP_SETUP=1; shift;;
    --force) FORCE=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="$(pwd)/cost_test_work"
mkdir -p "$WORK"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---- guardrail: confirm we're in-region --------------------------------------
say "Checking region (must be us-east-2 for free in-region reads)"
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 120" 2>/dev/null || true)
META() { curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null || true; }
REGION=$(META placement/region)
ITYPE=$(META instance-type)
echo "  region=${REGION:-unknown}  instance-type=${ITYPE:-unknown}"
if [ -n "$REGION" ] && [ "$REGION" != "us-east-2" ] && [ "$FORCE" != "1" ]; then
  echo "  REFUSING: region is '$REGION', not us-east-2. You'd pay egress and"
  echo "  measure the wrong rate. Relaunch in us-east-2, or pass --force." >&2
  exit 1
fi
[ -z "$REGION" ] && echo "  (couldn't reach instance metadata; make sure you ARE in us-east-2)"

# ---- setup -------------------------------------------------------------------
PYTHON=python3
command -v python3.11 >/dev/null 2>&1 && PYTHON=python3.11
if [ "$SKIP_SETUP" != "1" ]; then
  say "Installing deps (CPU torch; one-time)"
  command -v "$PYTHON" >/dev/null 2>&1 || sudo dnf install -y python3.11 || true
  PYTHON=python3.11; command -v python3.11 >/dev/null 2>&1 || PYTHON=python3
  rm -rf "$WORK/.venv"   # always start clean: a half-built venv from a failed
                         # earlier run is the usual source of import errors
  "$PYTHON" -m venv "$WORK/.venv"
  # CPU-only torch AND torchvision from the SAME index, so their versions match.
  # (open_clip pulls in torchvision; a PyPI torchvision against a CPU-index torch
  # gives "operator torchvision::nms does not exist".) This also avoids the
  # multi-GB CUDA build on a CPU box.
  "$WORK/.venv/bin/pip" -q install --upgrade pip
  "$WORK/.venv/bin/pip" -q install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  "$WORK/.venv/bin/pip" -q install open_clip_torch pillow numpy boto3 pyarrow
  # Fail fast & clearly if the torch/torchvision pair is mismatched.
  "$WORK/.venv/bin/python" -c "import torch, torchvision, open_clip" \
    || { echo "ERROR: torch/torchvision/open_clip import failed (version mismatch)" >&2; exit 1; }
fi
VENV="$WORK/.venv/bin/python"
VCPU=$(nproc)
echo "  python=$($VENV --version 2>&1)  vCPU=$VCPU"

# ---- sample a bounded subset -------------------------------------------------
say "Sampling ~$TARGET pages from the public bucket"
ITEMS=$(( (TARGET + PAGES_PER_ITEM - 1) / PAGES_PER_ITEM ))
"$VENV" "$SCRIPT_DIR/sample_manifest.py" \
  --items "$ITEMS" --pages-per-item "$PAGES_PER_ITEM" --out "$WORK/test.tsv"
echo "  manifest: $(wc -l < "$WORK/test.tsv") pages"

# ---- timed embed (ViT-B/32, blank filter on) ---------------------------------
# Source defaults to the webp derivative (web/, ~38x faster to decode than JP2);
# override with SOURCE=jp2 to measure the archival path for comparison.
SOURCE=${SOURCE:-webp}
WEBP_SIZE=${WEBP_SIZE:-medium}   # ~JP2-equivalent retrieval; 'small' is lossy
say "Timed embed (ViT-B/32, source=$SOURCE). Weights download once, before timing."
# HF_HUB_OFFLINE=0 so the first run can fetch the weights; embed_s3 starts its
# clock AFTER model load, so the reported minutes are pure decode+encode.
HF_HUB_OFFLINE=0 BHL_CLIP_MODEL=ViT-B-32 BHL_CLIP_PRETRAINED=laion2b_s34b_b79k \
  "$VENV" "$SCRIPT_DIR/embed_s3.py" --shard "$WORK/test.tsv" --out "$WORK/out" \
  --source "$SOURCE" --webp-size "$WEBP_SIZE" \
  --batch-size 128 --fetch-workers $(( VCPU * 4 )) --min-std 10 \
  2>&1 | tee "$WORK/embed.log"

# ---- parse throughput from embed_s3's own steady-state report ----------------
LINE=$(grep '^wrote ' "$WORK/embed.log" | tail -1)
VECTORS=$(echo "$LINE" | sed -E 's/.*: ([0-9,]+) vectors.*/\1/' | tr -d ',')
MINUTES=$(echo "$LINE" | sed -E 's/.* in ([0-9.]+)m.*/\1/')

# ---- price ($/hr): flag override, else a small built-in c7i table ------------
if [ -z "$HOURLY_RATE" ]; then
  case "$ITYPE" in
    c7i.large)     HOURLY_RATE=0.08925;;
    c7i.xlarge)    HOURLY_RATE=0.1785;;
    c7i.2xlarge)   HOURLY_RATE=0.357;;
    c7i.4xlarge)   HOURLY_RATE=0.714;;
    c7i.8xlarge)   HOURLY_RATE=1.428;;
    c7i.12xlarge)  HOURLY_RATE=2.142;;
    c7i.16xlarge)  HOURLY_RATE=2.856;;
    *) HOURLY_RATE="";;
  esac
  [ -n "$HOURLY_RATE" ] && echo "  using built-in us-east-2 on-demand price for $ITYPE: \$$HOURLY_RATE/hr (approx)"
fi

# ---- report ------------------------------------------------------------------
say "RESULTS"
awk -v v="$VECTORS" -v m="$MINUTES" -v vcpu="$VCPU" -v total="$TOTAL_PAGES" \
    -v rate_hr="$HOURLY_RATE" -v spot="$SPOT_FACTOR" -v itype="$ITYPE" 'BEGIN{
  secs = m*60; rate = v/secs;
  printf "  instance         : %s (%d vCPU)\n", (itype=="" ? "?" : itype), vcpu;
  printf "  pages embedded   : %d in %.1f min\n", v, m;
  printf "  throughput       : %.1f img/s  (%.2f img/s per vCPU)\n", rate, rate/vcpu;
  if (rate_hr=="") {
    printf "\n  (no $/hr known — re-run with --rate <on-demand $/hr> for dollars)\n";
    printf "  formula: $/1k = ($/hr) / (%.1f*3600) * 1000\n", rate;
  } else {
    per1k = (rate_hr/(rate*3600))*1000;
    full_od = total/rate/3600*rate_hr;
    printf "  on-demand $/hr   : %.4f\n", rate_hr;
    printf "  cost / 1k pages  : $%.5f\n", per1k;
    printf "\n  EXTRAPOLATED to %d pages:\n", total;
    printf "    on-demand, this instance : $%.0f   (%.0f instance-hours)\n", full_od, total/rate/3600;
    printf "    on Spot (~%.2fx)          : $%.0f   <-- what production would pay\n", spot, full_od*spot;
  }
  printf "\n  Note: embedding only. Add the Hetzner vector host (~EUR100/mo) for serving.\n";
}'

say "DONE — now TERMINATE this instance (it keeps billing until you do)."
