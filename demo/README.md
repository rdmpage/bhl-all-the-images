# BHL image-search demo (PHP front-end)

A single-file PHP page (`index.php`) over the Hetzner search API. The browser
only ever talks to *this* page; the page curls the API **server-side**, so:
- plain HTTP to the box IP is fine (no CORS, no mixed-content even if this page
  is served over HTTPS), and
- the API key never reaches the browser.

It renders results as a thumbnail grid linking to the public S3 webp images —
nothing is proxied or hosted here. Text box = phrase search; file picker = "find
similar pages" (image upload). **This is PHP, not Python** — the CLIP/torch half
lives on Hetzner; the host here only needs PHP with the `curl` extension.

## Configure

Two settings, read via `getenv()`:

| var | meaning |
|---|---|
| `BHL_SEARCH_API` | search API base URL, e.g. `http://<hetzner-ip>:8000` |
| `BHL_SEARCH_KEY` | must match the server's `BHL_SEARCH_KEY`; sent as `X-API-Key` |

Local dev uses the `env.php` pattern (gitignored): copy the template and edit.

```bash
cp env-template.php env.php      # then set the two values in env.php
```

`index.php` does `if (file_exists(__DIR__.'/env.php')) include 'env.php';`, and
`env.php` just `putenv()`s the two vars. If `BHL_SEARCH_KEY` is empty the page
simply doesn't send the header (fine when the server runs open).

## Run it

**Built-in PHP server (quickest):**
```bash
php -S localhost:8080 -t .       # from this demo/ dir; open http://localhost:8080/
```

**Apache (e.g. a Mac already serving test projects):** drop `index.php` +
`env.php` in the docroot. Confirm curl is enabled: `php -m | grep curl`.

**Heroku (public URL):** Heroku has an official **PHP** buildpack. Deploy a small
app with `index.php` + a minimal `composer.json` at its root, then set the config
vars instead of using `env.php`:
```bash
heroku config:set BHL_SEARCH_API=http://<hetzner-ip>:8000 \
                  BHL_SEARCH_KEY=<your-uuid>
```
(Server-to-server curl, so Heroku-over-HTTPS calling the box over HTTP is fine.)

## Notes

- The server enforces the key only when its `BHL_SEARCH_KEY` is set; otherwise it
  runs open and the header is ignored — so old/keyless setups keep working.
- `env.php` is gitignored (see `.gitignore`); never commit real keys. Keep the
  server's key in `/etc/bhl-search.env` (see `../hetzner/bhl-search.service`).
- Plain HTTP on a public IP is OK for a demo because the key guards it; for
  anything more, put the API behind Caddy/HTTPS (see `../hetzner/dry_run.md` §6)
  — the same `X-API-Key` header rides along unchanged.
