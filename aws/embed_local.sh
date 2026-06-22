#!/usr/bin/env bash
#
# Embed a whole manifest locally, shard by shard -- the no-AWS path used to make
# the Tier-0 vectors, scaled to a real corpus. Runs embed_s3.py once per
# shard-NNNNN.tsv, reading the public bucket's webp derivatives. Resumable:
# embed_s3.py skips any shard whose .parquet already exists, so just rerun this
# if it's interrupted (spot reclaim, a dropped link, Ctrl-C).
#
#   bash aws/embed_local.sh                      # manifest/ -> out/
#   bash aws/embed_local.sh manifest/ out/       # explicit dirs
#
# Knobs (env):
#   BHL_MIN_STD     blank-page filter; drop pages with grayscale stddev < this
#                   (default 10 -- cuts blank/cream pages that otherwise become
#                   null attractors in retrieval; set 0 to keep everything)
#   BHL_WEBP_SIZE   webp derivative size (default medium; ~JP2-equivalent p@5)
#   BHL_SOURCE      webp | jp2 (default webp; ~38x faster decode)
#   PYTHON          interpreter (auto: ./.venv/bin/python if present, else python3)
#
set -uo pipefail

MANIFEST="${1:-manifest}"
OUT="${2:-out}"
SOURCE="${BHL_SOURCE:-webp}"
SIZE="${BHL_WEBP_SIZE:-medium}"
MIN_STD="${BHL_MIN_STD:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$SCRIPT_DIR/../.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/../.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

shopt -s nullglob
shards=( "$MANIFEST"/shard-*.tsv )
total=${#shards[@]}
if [ "$total" -eq 0 ]; then
  echo "no shard-*.tsv files in '$MANIFEST' -- run manifest_from_ids.py first" >&2
  exit 1
fi

echo "embedding $total shards from '$MANIFEST' -> '$OUT'" >&2
echo "  source=$SOURCE size=$SIZE min-std=$MIN_STD  python=$PYTHON" >&2

start=$(date +%s)
i=0; failed=0
for s in "${shards[@]}"; do
  i=$((i + 1))
  echo "[$i/$total] $s" >&2
  if ! "$PYTHON" "$SCRIPT_DIR/embed_s3.py" \
        --shard "$s" --out "$OUT" \
        --source "$SOURCE" --webp-size "$SIZE" --min-std "$MIN_STD"; then
    echo "  !! shard failed (continuing): $s" >&2
    failed=$((failed + 1))
  fi
done

elapsed=$(( $(date +%s) - start ))
echo "" >&2
echo "loop done in $((elapsed / 60))m $((elapsed % 60))s; $failed shard(s) failed" >&2

# Final tally straight from the parquet metadata (no full read).
"$PYTHON" - "$OUT" <<'PY'
import sys, glob
import pyarrow.parquet as pq
out = sys.argv[1]
files = sorted(glob.glob(f"{out}/*.parquet"))
n = sum(pq.read_metadata(f).num_rows for f in files)
print(f"embedded {n:,} vectors across {len(files)} parquet files -> {out}/")
PY

if [ "$failed" -ne 0 ]; then
  echo "NOTE: $failed shard(s) failed -- rerun this script to retry them "\
       "(finished shards skip)." >&2
  exit 1
fi
