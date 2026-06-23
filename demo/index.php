<?php
/**
 * BHL image-search demo — a thin PHP front-end over the Hetzner search API.
 *
 * The browser only ever talks to *this* PHP page; the page curls the search API
 * server-side (so no CORS, and the plain-HTTP API call never triggers
 * mixed-content blocking even if this page is later served over HTTPS). Results
 * carry public S3 webp URLs, rendered directly as <img> — nothing is proxied.
 *
 * Point it at your box:  export BHL_SEARCH_API=http://<box-ip>:8000
 * Run locally:           php -S localhost:8080 -t demo
 * then open http://localhost:8080/
 */

// Local dev: env.php (gitignored) does putenv() for these. Production (Heroku):
// set the same names as config vars. See env-template.php.
if (file_exists(dirname(__FILE__) . '/env.php')) {
    include 'env.php';
}

$API = getenv('BHL_SEARCH_API') ?: 'http://CHANGE-ME:8000';
$KEY = getenv('BHL_SEARCH_KEY') ?: '';   // sent as X-API-Key if set

$q       = isset($_GET['q']) ? trim($_GET['q']) : '';
$k       = max(1, min(48, (int)($_GET['k'] ?? 12)));
$results = null;
$error   = null;
$mode    = null;   // 'text' | 'image'

/** Call the search API and decode its JSON, or set $error. */
function call_api($url, $post_file = null) {
    global $error, $KEY;
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 30);
    if ($KEY !== '') {
        curl_setopt($ch, CURLOPT_HTTPHEADER, ["X-API-Key: $KEY"]);
    }
    if ($post_file !== null) {
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, ['file' => $post_file]);
    }
    $body = curl_exec($ch);
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $cerr = curl_error($ch);
    curl_close($ch);
    if ($body === false)        { $error = "Could not reach API: $cerr"; return null; }
    if ($code !== 200)          { $error = "API returned HTTP $code: $body"; return null; }
    $data = json_decode($body, true);
    if (!is_array($data))       { $error = "Bad JSON from API"; return null; }
    return $data;
}

// Image-similarity query: forward the uploaded file to POST /search.
if (!empty($_FILES['image']['tmp_name']) && is_uploaded_file($_FILES['image']['tmp_name'])) {
    $mode = 'image';
    $cfile = new CURLFile($_FILES['image']['tmp_name'],
                          $_FILES['image']['type'] ?: 'application/octet-stream',
                          $_FILES['image']['name']);
    $data = call_api($API . '/search?k=' . $k, $cfile);
    $results = $data['results'] ?? null;
}
// Text query.
elseif ($q !== '') {
    $mode = 'text';
    $url = $API . '/search?' . http_build_query(['q' => $q, 'k' => $k]);
    $data = call_api($url);
    $results = $data['results'] ?? null;
}

function h($s) { return htmlspecialchars((string)$s, ENT_QUOTES, 'UTF-8'); }
?>
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BHL image search</title>
<style>
  :root { --ink:#1a1a1a; --muted:#6b7280; --line:#e5e7eb; --accent:#2f5d50; }
  * { box-sizing: border-box; }
  body { font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         color: var(--ink); margin: 0; background: #fafaf9; }
  header { padding: 2rem 1.5rem 1rem; max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
  .sub { color: var(--muted); font-size: .9rem; margin: 0; }
  form { max-width: 1100px; margin: 0 auto; padding: 0 1.5rem; }
  .row { display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; margin: .75rem 0; }
  input[type=text] { flex: 1; min-width: 240px; padding: .6rem .8rem; font-size: 1rem;
                     border: 1px solid var(--line); border-radius: 8px; }
  input[type=number] { width: 5rem; padding: .6rem; border: 1px solid var(--line); border-radius: 8px; }
  button { padding: .6rem 1.1rem; font-size: 1rem; border: 0; border-radius: 8px;
           background: var(--accent); color: #fff; cursor: pointer; }
  button.alt { background: #fff; color: var(--accent); border: 1px solid var(--accent); }
  .or { color: var(--muted); font-size: .85rem; text-align: center; margin: .25rem 0; }
  .err { max-width: 1100px; margin: 1rem auto; padding: .8rem 1rem; background: #fef2f2;
         border: 1px solid #fecaca; border-radius: 8px; color: #991b1b; font-size: .9rem; }
  .meta { max-width: 1100px; margin: 1rem auto .25rem; padding: 0 1.5rem; color: var(--muted); font-size: .9rem; }
  .grid { max-width: 1100px; margin: 0 auto; padding: 1rem 1.5rem 3rem;
          display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
  .card { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: #fff;
          text-decoration: none; color: inherit; transition: box-shadow .15s; }
  .card:hover { box-shadow: 0 4px 16px rgba(0,0,0,.1); }
  .card img { width: 100%; height: 170px; object-fit: cover; display: block; background: #f1f1ee; }
  .card .cap { padding: .5rem .6rem; font-size: .78rem; }
  .card .score { color: var(--accent); font-weight: 600; }
  .card .id { color: var(--muted); word-break: break-all; }
</style>
</head>
<body>
<header>
  <h1>BHL image search <span style="color:var(--muted);font-weight:400">— dry run</span></h1>
  <p class="sub">CLIP page embeddings over pgvector. Search by phrase, or upload an image to find visually similar pages.</p>
</header>

<form method="get" action="">
  <div class="row">
    <input type="text" name="q" value="<?= h($q) ?>" placeholder="e.g. a colour plate of birds" autofocus>
    <input type="number" name="k" value="<?= $k ?>" min="1" max="48" title="number of results">
    <button type="submit">Search</button>
  </div>
</form>

<form method="post" action="" enctype="multipart/form-data">
  <div class="or">— or —</div>
  <div class="row">
    <input type="file" name="image" accept="image/*" required>
    <input type="number" name="k" value="<?= $k ?>" min="1" max="48" title="number of results">
    <button class="alt" type="submit">Find similar pages</button>
  </div>
</form>

<?php if ($error): ?>
  <div class="err"><?= h($error) ?></div>
  <?php if (strpos($API, 'CHANGE-ME') !== false): ?>
    <div class="meta">Set the API address first:
      <code>export BHL_SEARCH_API=http://&lt;box-ip&gt;:8000</code> before starting PHP.</div>
  <?php endif; ?>
<?php endif; ?>

<?php if ($results !== null): ?>
  <div class="meta">
    <?= count($results) ?> result<?= count($results) === 1 ? '' : 's' ?>
    <?php if ($mode === 'text'): ?>for &ldquo;<?= h($q) ?>&rdquo;<?php endif; ?>
    <?php if ($mode === 'image'): ?>similar to the uploaded image<?php endif; ?>
  </div>
  <div class="grid">
    <?php foreach ($results as $r): ?>
      <a class="card" href="<?= h($r['image_url']) ?>" target="_blank" rel="noopener">
        <img src="<?= h($r['thumb_url']) ?>" loading="lazy" alt="">
        <div class="cap">
          <div class="score"><?= h(number_format((float)$r['score'], 3)) ?></div>
          <div class="id"><?= h($r['barcode']) ?> · p<?= h($r['seq']) ?></div>
        </div>
      </a>
    <?php endforeach; ?>
  </div>
<?php endif; ?>
</body>
</html>
