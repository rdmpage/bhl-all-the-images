# Embed a corpus on one EC2 box → load onto Hetzner

A checklist for embedding a real corpus (e.g. a BioStor item set) on a single
in-region EC2 instance, then moving just the vectors to the Hetzner serving box.
AWS reads the images in-region (free, fast); you only ever move the ~800 MB of
parquet out, never the tens of GB of images.

For the local, zero-AWS alternative see `embed_local.sh`'s header; for the
serving half see `../hetzner/dry_run.md`.

---

## 0. Before you launch

- Build the manifest **on your Mac** first (it's small and you want to eyeball
  the page count + coverage before paying for a box):
  ```bash
  .venv/bin/python aws/manifest_from_ids.py --ids items.txt --out manifest/ --shards 128
  #  -> "done: N pages from M items ..."   and manifest/missing_ids.txt
  ```
- Confirm an **AWS Budget alarm** exists. A forgotten instance is the only thing
  here that runs up a real bill.

## 1. Launch one instance

- **Region: us-east-2** (same as `s3://bhl-open-data` — in-region reads are free
  and fast; anywhere else pays egress and crawls).
- **AMI: Amazon Linux 2023.** Your existing keypair + security group (SSH 22)
  from the cost test.
- **On-demand, not Spot, for a single-box run.** Output sits on local disk, so a
  Spot reclaim loses progress. At ~$1.70 total it's not worth the risk. (Spot is
  worth it once you write output to S3 and run several boxes — see the end.)
- **Size sets wall-clock; total cost is ~fixed (~$1.70 for 635K pages):**

  | instance | vCPU | ~throughput | 635K pages | ~$/hr (us-east-2 on-demand) |
  |---|---|---|---|---|
  | c7i.2xlarge | 8 | ~38 img/s | ~4.7h | 0.357 |
  | c7i.4xlarge | 16 | ~75 img/s | ~2.4h | 0.714 |
  | c7i.8xlarge | 32 | ~150 img/s | ~1.2h | 1.428 |

> SSH key: if `ssh`/`scp` says *"UNPROTECTED PRIVATE KEY FILE"* and ignores the
> key, fix its perms: `chmod 600 <key.pem>` (it refuses world-readable keys).

## 2. Install (no AWS credentials needed)

The source bucket is read **unsigned** (public) and parquet is written to local
disk, so there's nothing of yours to configure — no `aws configure`, no IAM.

```bash
sudo dnf install -y git python3.11 tmux     # AL2023 ships none of these by default
git clone https://github.com/rdmpage/bhl-all-the-images.git
cd bhl-all-the-images
python3.11 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip

# torch + torchvision from the SAME cpu index, in one step. Do NOT use
# `-r requirements.txt --extra-index-url ...`: open_clip pulls torchvision, and
# a PyPI torchvision against a cpu-index torch fails at runtime with
# "operator torchvision::nms does not exist". One index keeps them matched.
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install open_clip_torch pillow numpy boto3 pyarrow

python -c "import torch, torchvision, open_clip" && echo "deps OK"
```

Install at the **repo-root `.venv`** (as above) so `embed_local.sh` auto-detects
it and you can run the embed with no flags.

## 3. Send the manifest up

Reuse the one you built and checked in step 0 (small, ~tens of MB):

```bash
# from your Mac:
scp -i <key.pem> -r manifest/ ec2-user@<ec2-ip>:~/bhl-all-the-images/
```

(Or rebuild it on the box — listing is in-region and fast — but reusing the
validated one guarantees the same item set.)

## 4. Embed — **inside tmux**

This is a multi-hour run; a foreground process dies with SIGHUP the moment your
SSH connection drops (we lost a run to exactly this overnight). Run it in tmux so
it survives disconnects and you can reattach to watch:

```bash
tmux new -s embed
. .venv/bin/activate
bash aws/embed_local.sh manifest/ out/
# detach: Ctrl-b then d   |   reattach later: tmux attach -t embed
```

In-region reads of the webp derivatives; webp/medium + blank filter (min-std 10)
by default. The CLIP weights download once on shard 0 (`embed_local.sh` clears
`HF_HUB_OFFLINE` so the fetch isn't blocked), then cache for the rest. Resumable
— if it dies, rerun and finished shards skip. It ends with `loop done ... 0
shard(s) failed` and an `embedded N vectors` tally — read both before moving on.

> Expect yield somewhat below your manifest's page count: the blank filter drops
> blank/cream pages (e.g. 635K manifest pages → ~589K vectors). That's correct.

## 5. Pull the parquet out, then up to Hetzner

You move only the vectors (~800 MB), not the images:

```bash
# from your Mac:
mkdir -p out
scp -i <key.pem> 'ec2-user@<ec2-ip>:~/bhl-all-the-images/out/*.parquet' ./out/
scp ./out/*.parquet root@<hetzner-ip>:~/bhl-all-the-images/out/
```

## 6. TERMINATE the EC2 instance

The parquet is safe off-box now, so the instance (and its local disk) is
disposable. **Terminate it** — it bills until you do.

## 7. Load on Hetzner

The serving box is already set up (`../hetzner/dry_run.md`); just swap the data.
Directory load is **not idempotent** (plain COPY), so start from a clean table:

```bash
psql -d bhl -c "TRUNCATE page_embedding;"     # clear the Tier-0 / previous set
cd ~/bhl-all-the-images/hetzner
python load_parquet.py --src ../out/ --dsn postgresql:///bhl
psql -d bhl -f ../db/index_hetzner.sql        # rebuild HNSW (minutes at <1M)
# restart uvicorn
```

635K vectors ≈ ~1.3 GB + index — fits the existing 8 GB box, no resize. (Resize
only past ~2–3M.)

---

## When you scale up (full BioStor / multiple boxes)

The single-box, local-disk, scp path above is simplest for <1M pages on one box.
For a bigger corpus you want several boxes and Spot, which means output must
survive reclaim — i.e. write to **your own S3 bucket** instead of local disk:

- Create one bucket; give the instances creds (an instance role, or `aws
  configure` for a one-off).
- Run `embed_s3.py --out s3://MY-BUCKET/out/` per shard across N boxes — shards
  are independent and per-shard idempotent, so N boxes share the work and Spot
  reclaim just means relaunch (done shards skip).
- Hetzner pulls straight from S3: `load_parquet.py --src s3://MY-BUCKET/out/`
  (~free egress within the AWS free tier), no scp.
- Move Postgres to a bigger box (16–32 GB for a few-M corpus; 128 GB for all 63M).
