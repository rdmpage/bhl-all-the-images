# Hetzner dry run — prove the whole serve chain on the Tier-0 vectors

> ✅ **Validated end-to-end on 2026-06-22** — a live Hetzner Cloud box running
> Ubuntu 26.04 / Postgres 18.4 / pgvector (built from source). Text *and*
> image-similarity search both return coherent results through the PHP demo.
> The steps below are the actual recipe that worked, gotchas folded in.

Goal: stand up the *entire* serving half (pgvector + halfvec + HNSW + a CLIP
search API) on a small, cheap Hetzner Cloud box, fed by the ~1.2k Tier-0 vectors
you already have on disk. **No AWS spend** — this validates plumbing, not scale.
The same steps re-run unchanged when you later load a 1M-page slice or the full
63M corpus; only the box size and load time change.

What this de-risks before the real run:
- `halfvec` on **pgvector ≥ 0.7** — never actually exercised yet (local is 0.4.1).
- HNSW build + cosine query against `halfvec_cosine_ops`.
- the CLIP-in-the-loop search API (text *and* image queries).
- the public S3 webp image URLs the UI will render.

---

## 0. Box

Hetzner Cloud, ~4 vCPU / 8 GB, **hourly billing — destroy it when done**. Any
small box works; pick by what's in stock:
- **CX32** (x86) or — often cheaper / more available — **CAX21** (Ampere ARM64).
- 8 GB just gives CPU torch headroom; the 1.2k vectors are nothing. (The 128 GB
  dedicated box only matters at 63M.)

On ARM, see the note in step 5 about the torch wheel.

```bash
ssh root@<box-ip>
apt update && apt -y upgrade
apt -y install git
# venv needs the *version-matched* package, not the generic one. On Ubuntu 26.04
# the default python3 is 3.14, so it's python3.14-venv (the generic
# `python3-venv` does NOT pull ensurepip and `python3 -m venv` then fails):
apt -y install "python$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')-venv"
```

## 1. Postgres (distro default) + pgvector ≥ 0.7

We need only **Postgres ≥ 14** and **pgvector ≥ 0.7** (for `halfvec`) — do NOT
pin a major version. Install whatever the distro ships:

```bash
apt -y install postgresql postgresql-contrib
psql --version          # note the major, e.g. 17 or 18
```

Then pgvector. On a fresh Ubuntu release the `postgresql-NN-pgvector` package is
often not in the repos yet, so the reliable route is a source build. Build the
**latest release tag** — do NOT pin v0.8.0, which predates Postgres 18 and won't
compile against its headers:

```bash
# try the package first; skip to the source build if it's missing:
apt -y install postgresql-server-dev-all build-essential
git clone https://github.com/pgvector/pgvector.git
cd pgvector
git checkout "$(git describe --tags "$(git rev-list --tags --max-count=1)")"   # newest tag
make && sudo make install && cd ~
#  if make still errors on a brand-new PG major, drop the checkout and build HEAD
```

Create a DB role for the OS user, then the database. Postgres only makes the
`postgres` superuser by default, so `createdb` as e.g. root fails with
`role "<user>" does not exist` until you add one:

```bash
sudo -u postgres createuser --superuser "$(whoami)"
createdb bhl
psql -d bhl -c "CREATE EXTENSION vector; SELECT extversion FROM pg_extension WHERE extname='vector';"
#  expect >= 0.7.0  (peer auth maps the OS user -> same-named role over the socket)
```

> The role must match whatever OS user later runs `load_parquet.py` and
> `search_api.py` (both default to `postgresql:///bhl` = peer auth, no password).
> If you run them as root, the `root` role above covers it.

Postgres stays bound to localhost (default) — only the API port is ever exposed.

## 2. Code + schema

```bash
git clone https://github.com/rdmpage/bhl-all-the-images.git
cd bhl-all-the-images
# (already cloned earlier? just `git pull origin main` to pick up new commits)
psql -d bhl -f db/schema_hetzner.sql        # bare halfvec(512) table
```

## 3. Get the Tier-0 vectors onto the box

The `tier0/` parquet is gitignored, so copy it up from your laptop (it's tiny):

```bash
# on the laptop, from the repo root:
scp tier0/out_web_medium/wide.parquet root@<box-ip>:~/bhl-all-the-images/hetzner/
```

(`out_web_medium` = webp/medium, ViT-B/32 — the set that matches the production
plan. `out_b_wide` is the JP2 equivalent if you'd rather compare.)

## 4. Load + index

```bash
cd ~/bhl-all-the-images/hetzner
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python load_parquet.py --src wide.parquet --dsn postgresql:///bhl
#  -> "done: ~1,205 vectors loaded"   <-- first real halfvec COPY

psql -d bhl -f ../db/index_hetzner.sql      # PK + HNSW; instant at this size
```

## 5. Search API

```bash
# x86 box:
pip install -r requirements-api.txt --extra-index-url https://download.pytorch.org/whl/cpu
# ARM box (CAX): drop the extra index -- that wheel index is x86-only; PyPI has
# the aarch64 CPU torch wheel:
#   pip install -r requirements-api.txt
export DATABASE_URL=postgresql:///bhl
uvicorn search_api:app --host 127.0.0.1 --port 8000
#  first start downloads CLIP weights (~600 MB) once, then caches them
```

Smoke-test from a second shell on the box:

```bash
curl 'http://127.0.0.1:8000/healthz'
curl 'http://127.0.0.1:8000/search?q=a+colour+plate+of+birds&k=6'
curl -F file=@/path/to/a/page.jpg 'http://127.0.0.1:8000/search?k=6'
```

Each result has `score`, `thumb_url`, and `image_url` pointing straight at the
public S3 webp — open one in a browser to confirm retrieval looks sane (compare
against `tier0/q_*.html` from the local trial).

## 6. Expose it to the demo

**Quick & dirty (HTTP by IP, while messing around)** — bind to all interfaces
and let the PHP demo curl `http://<box-ip>:8000`. The demo's API call is
server-side, so plain HTTP is fine (no CORS, no mixed-content):

```bash
uvicorn search_api:app --host 0.0.0.0 --port 8000
# ensure inbound TCP 8000 is allowed: if a Hetzner Cloud Firewall is attached,
# add a rule; if ufw is active on the box, `ufw allow 8000`.
```

> Caveat: this is an unauthenticated API on a public IP. Fine for a short-lived
> dry-run box you tear down; do NOT leave it up. Add an API key / Caddy
> basic-auth (below) before anything resembling real traffic.

**Private (no port opened)** — SSH tunnel instead:

```bash
ssh -N -L 8000:127.0.0.1:8000 root@<box-ip>
# now http://127.0.0.1:8000/search?... works locally; point the PHP demo there
```

**Proper (public HTTPS, for a Heroku-hosted demo to reach)** — Caddy gives you
auto-TLS if you point a (sub)domain at the box:

```bash
apt -y install caddy
# /etc/caddy/Caddyfile:
#   search.example.org {
#       reverse_proxy 127.0.0.1:8000
#   }
systemctl reload caddy
```

Run uvicorn under **systemd** so it survives SSH disconnects, restarts on crash,
and comes back on reboot (a foreground `uvicorn` dies with SIGHUP when your
connection drops). Use the bundled unit:

```bash
cp bhl-search.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now bhl-search
systemctl status bhl-search        # active (running); journalctl -u bhl-search -f for logs
```

The demo then calls `http://<box-ip>:8000/search?q=...` (or the Caddy HTTPS URL)
— the same curl-an-HTTP-endpoint shape as the existing `bhl-elastic-test` site.

> For a public endpoint, add a cheap guard before any real traffic: an API key
> header checked in `search_api.py`, or Caddy basic-auth / rate-limiting. The
> dry run behind an SSH tunnel needs none of this.

## 7. Run the demo front-end

`demo/index.php` is a thin PHP page that curls the API **server-side** (so plain
HTTP to the box IP is fine — no CORS, no mixed-content) and renders results as a
thumbnail grid. It runs anywhere with PHP; easiest is the built-in server on
your laptop, pointed at the box via the `BHL_SEARCH_API` env var:

```bash
# on the laptop, in the repo:
export BHL_SEARCH_API=http://<box-ip>:8000     # the IP from step 6
php -S localhost:8080 -t demo
# open http://localhost:8080/
```

Text box = phrase search; file picker = "find similar pages" (image upload).
If `BHL_SEARCH_API` is unset the page says so rather than erroring cryptically.

## 8. Tear down

The box is hourly-billed and holds nothing precious (vectors are reproducible
from the parquet). **Delete it from the Hetzner console when finished** —
re-running this file rebuilds it in minutes.

---

### When you graduate to a real corpus

- Bigger slice / full run: same steps; swap step 3 for `aws s3 sync s3://MY-BUCKET/out/`
  (or `load_parquet.py --src s3://...`) and give step 4's index build RAM + hours.
- Move Postgres to the 128 GB dedicated box; the API can stay on the same box
  (co-located) or a small companion — keep it next to the DB, not on Heroku.
- Persist the original image key (or padding width) in the parquet so
  `image_url()` doesn't have to assume 4-digit `seq` padding.
