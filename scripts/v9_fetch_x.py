"""v9 X (Twitter) sourcing -- download ORIGINAL-resolution REAL PHOTOS from target accounts'
timelines (or a search query) into data/v9_x/ via the X API v2 (pay-per-use).

WHY: X has amateur deep-focus selfies/mirror-selfies that Pexels lacks (right content), but every
upload is re-encoded to JPEG (diseased encoding). The v9 curate gates (>=1536 + Laplacian + grid
background-sharpness) are the safety net. Targeting good accounts beats search.

COST (pay-per-use): the docs list $0.005/post read, but OBSERVED billing was ~$0.02 per ~300 reads
(~$0.00007/read, 2026-06-05). i.e. cost is NEGLIGIBLE -> the real bottleneck is YIELD (most X photos
are < 1536px), not money. The printed $ below is a ROUGH estimate; the X console is the source of
truth. Because reads are cheap, the right lever for more images is a BIG --max-reads (not loosening
the >=1536 filter, which would only download soft sub-1536 junk that curate drops). To raise yield:
  - search mode auto-adds  has:images -is:retweet  + real-photo negatives  -> fewer junk reads.
  - page size shrinks near --max-reads so you never overshoot the budget by a whole page.
  - multi-image posts give up to 4 images for ONE read (free value).
  - timeline mode of a few good accounts has far higher yield/read than search firehose.
Always set a SPENDING LIMIT in the X console; --max-reads is only a soft guard.

Setup: X dev account + pay-per-use credits, then App-only Bearer Token in .env:
    X_BEARER_TOKEN=AAAA...

Run:
  # preview the exact query + cost shape WITHOUT spending anything:
  python scripts/v9_fetch_x.py --query "mirror selfie" --dry-run
  # search (recent 7d, or --full-archive back to 2006). Real-photo filter is auto-added:
  python scripts/v9_fetch_x.py --query "mirror selfie" --max-reads 500 --full-archive
  # MORE keywords in one cheap pass via OR (parentheses + OR):
  python scripts/v9_fetch_x.py --query "(selfie OR \"mirror selfie\" OR ootd OR \"full body\")" --max-reads 800 --full-archive
  # timeline of specific accounts (cheapest yield/read):
  python scripts/v9_fetch_x.py --handles acct1,acct2 --max-reads 1000

Real-photo only: search auto-appends negatives (-anime -hentai -art -illustration -drawing
-cartoon -3d -cgi -render -aiart). Disable with --no-auto-filter. NOTE: this only filters
explicitly-tagged art; still eyeball data/v9_x and delete any 2D/CGI that slips through.

NSFW: include adult terms in --query; your X app must have sensitive-media enabled. The anime/
hentai negatives keep it to REAL photos. LEGAL ADULTS ONLY -- the stage-3 captioner's underage
block stays on, non-negotiable.

ToS: training-data use of X content is a gray area; NSFW of real people = consent/legal. Your call.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.x.com/2"
PRICE_READ = 0.00007  # ROUGH: observed ~$0.02 per ~300 reads (2026-06-05). X console = source of truth.

# search-mode auto-filter: image posts only, no retweet-dups, and REAL PHOTOS only (exclude art/cgi/ai).
# Each negative also shrinks the result set -> fewer paid reads. Override with --no-auto-filter.
AUTO_FILTER = ("has:images -is:retweet -anime -hentai -art -illustration "
               "-drawing -cartoon -3d -cgi -render -aiart")


def get_token():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    tok = (os.environ.get("X_BEARER_TOKEN") or "").strip().strip('"').strip("'").strip()
    if not tok:
        sys.stderr.write("X_BEARER_TOKEN not set. Add it to .env as X_BEARER_TOKEN=...\n")
        sys.exit(1)
    return tok


def api_get(url, token):
    """GET with bearer auth; retry on 429/5xx with exponential backoff."""
    for attempt in range(6):
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "anima-v9/1.0 (research dataset tool)",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 5:
                wait = 2 ** attempt * 5
                sys.stderr.write(f"  HTTP {e.code}; backoff {wait}s (attempt {attempt + 1})\n")
                time.sleep(wait)
                continue
            sys.stderr.write(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}\n")
            raise
    raise RuntimeError("api_get: exhausted retries")


def resolve_user(handle, token):
    data = api_get(f"{API}/users/by/username/{urllib.parse.quote(handle)}", token)
    uid = data.get("data", {}).get("id")
    if not uid:
        raise RuntimeError(f"could not resolve handle '{handle}': {data}")
    return uid


def _media_params():
    return "expansions=attachments.media_keys&media.fields=url,width,height,type"


def iter_pages(base_url, token, max_reads, reads_state):
    """Yield each page's includes.media list. Page size shrinks near the budget edge so we never
    overshoot --max-reads by a whole page (minimum cost). reads_state = mutable [posts_read]."""
    token_param = None
    while reads_state[0] < max_reads:
        page = max(10, min(100, max_reads - reads_state[0]))   # don't fetch 100 if budget almost spent
        url = base_url + f"&max_results={page}"
        if token_param:
            url += f"&pagination_token={token_param}" if "/users/" in base_url else f"&next_token={token_param}"
        data = api_get(url, token)
        posts = data.get("data", []) or []
        reads_state[0] += len(posts)
        yield data.get("includes", {}).get("media", []) or []
        token_param = data.get("meta", {}).get("next_token")
        if not token_param or not posts:
            return


def save_media(media_list, out, min_short, saved_state):
    """Download every >=min_short photo in the page. media_key skip = no re-download / no dup cost."""
    for m in media_list:
        if m.get("type") != "photo":
            continue
        w, h = m.get("width", 0), m.get("height", 0)
        if min(w, h) < min_short:
            continue
        key, url = m.get("media_key"), m.get("url")
        if not key or not url:
            continue
        dest = out / f"x_{key}.jpg"
        if dest.exists():
            continue
        orig = url + ("&" if "?" in url else "?") + "name=orig"
        try:
            req = urllib.request.Request(orig, headers={"User-Agent": "anima-v9/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            saved_state[0] += 1
        except Exception as e:
            sys.stderr.write(f"  download failed {key}: {e!r}\n")


def _run(base, token, args, saved_state, reads_state):
    for media in iter_pages(base, token, args.max_reads, reads_state):
        save_media(media, args._out, args.min_short, saved_state)
        print(f"  saved={saved_state[0]} posts_read={reads_state[0]} ~${reads_state[0]*PRICE_READ:.3f} (rough)")
        # cost guard: most X photos are < 1536px (esp. mirror selfies). If a query yields 0 keepers
        # in the first ~100 reads, it's a low-res query -> stop now (~$0.50) instead of burning the budget.
        if reads_state[0] >= 100 and saved_state[0] == 0:
            sys.stderr.write("  ABORT: 0 images >= --min-short in first ~100 reads -> low-res query; "
                             "stopping to save cost. Try a broader query or lower --min-short (but <1536 "
                             "upscales soft and is dropped by curate anyway).\n")
            return


def main():
    ap = argparse.ArgumentParser(description="Download high-res REAL photos from X into data/v9_x/.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--handles", help="comma-separated account handles (no @) for timeline mode")
    g.add_argument("--query", help="search query; wrap multiple terms in (a OR b OR c)")
    ap.add_argument("--full-archive", action="store_true", help="full-archive search (back to 2006)")
    ap.add_argument("--max-reads", type=int, default=2000, help="soft cap on posts read (~$0.00007 each = cheap; raise freely for more yield)")
    ap.add_argument("--min-short", type=int, default=1536, help="min short side (px) to keep")
    ap.add_argument("--no-auto-filter", action="store_true", help="do NOT auto-add has:images -is:retweet + real-photo negatives")
    ap.add_argument("--dry-run", action="store_true", help="print the query/URL + budget, spend NOTHING")
    ap.add_argument("--out", default="data/v9_x")
    args = ap.parse_args()

    args._out = Path(args.out)
    saved_state, reads_state = [0], [0]
    est = args.max_reads * PRICE_READ
    print(f"budget: up to {args.max_reads} reads ~= ${est:.3f} max (rough; X console is truth. reads are cheap -> raise freely)")

    if args.dry_run:
        if args.query:
            q = args.query if args.no_auto_filter else f"{args.query} {AUTO_FILTER}"
            ep = "tweets/search/all" if args.full_archive else "tweets/search/recent"
            print(f"DRY RUN (no spend)\n  endpoint: {ep}\n  query: {q}\n  url: {API}/{ep}?query=<encoded>&{_media_params()}&max_results=<=100")
        else:
            print(f"DRY RUN (no spend)\n  timeline handles: {args.handles}\n  per handle: {API}/users/<id>/tweets?exclude=replies,retweets&{_media_params()}&max_results=<=100")
        return

    token = get_token()
    args._out.mkdir(parents=True, exist_ok=True)

    if args.handles:
        for handle in [h.strip().lstrip("@") for h in args.handles.split(",") if h.strip()]:
            if reads_state[0] >= args.max_reads:
                break
            print(f"timeline @{handle} ...")
            try:                                  # one bad handle must not abort the rest of the run
                uid = resolve_user(handle, token)
            except Exception as e:
                sys.stderr.write(f"  skip @{handle}: {e}\n")
                continue
            _run(f"{API}/users/{uid}/tweets?exclude=replies,retweets&{_media_params()}",
                 token, args, saved_state, reads_state)
    else:
        q = args.query if args.no_auto_filter else f"{args.query} {AUTO_FILTER}"
        ep = "tweets/search/all" if args.full_archive else "tweets/search/recent"
        _run(f"{API}/{ep}?query={urllib.parse.quote(q)}&{_media_params()}",
             token, args, saved_state, reads_state)

    print(f"\nDone. Saved {saved_state[0]} images (>= {args.min_short}px) -> {args._out}/")
    print(f"Cost: {reads_state[0]} post-reads ~= ${reads_state[0]*PRICE_READ:.3f} (ROUGH; check X console). "
          f"Reads are cheap -> if yield is low, raise --max-reads rather than lowering --min-short.")


if __name__ == "__main__":
    main()
