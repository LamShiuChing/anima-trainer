"""v9 X (Twitter) sourcing -- download ORIGINAL-resolution photos from target accounts' timelines
(or a search query) into data/v9_x/ via the X API v2 (pay-per-use).

WHY: X has amateur deep-focus selfies/mirror-selfies that Pexels lacks (right content), but every
upload is re-encoded to JPEG (diseased encoding). The v9 curate gates (>=1536 + Laplacian + grid
background-sharpness) are the safety net -> expect ~10% yield. Targeting good accounts beats search.

Setup (one time):
  1. X developer account + project/app, enable pay-per-use, and SET A SPENDING LIMIT in the console.
     Pricing (verified 2026-06-05): post read $0.005, media read $0.005. See
     https://docs.x.com/x-api/getting-started/pricing
  2. App-only Bearer Token -> add to .env in project root (same file as GEMINI_API_KEY):
        X_BEARER_TOKEN=AAAA...
  3. stdlib urllib only; dotenv optional.

Run:
  # timeline mode (recommended) -- one or more handles (no @), comma-separated
  python scripts/v9_fetch_x.py --handles someacct,another --max-reads 2000
  # search mode -- recent (last 7 days) by default, or --full-archive (back to 2006)
  python scripts/v9_fetch_x.py --query "mirror selfie full body" --max-reads 1000 --full-archive

Notes:
  - Filters min(width,height) >= --min-short (default 1536) from API metadata BEFORE downloading.
  - Downloads name=orig (largest, ~<=4096). Skips media_keys already on disk (safe to re-run).
  - --max-reads caps posts fetched (cost guard). The script prints a running $ estimate.
  - ToS: training-data use of X content is a gray area; NSFW of real people = consent/legal. Your call.
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
PRICE_READ = 0.005  # USD per post read AND per media resource (pay-per-use, 2026-06-05)


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
    return ("expansions=attachments.media_keys"
            "&media.fields=url,width,height,type&max_results=100")


def iter_pages(base_url, token, max_reads, reads_state):
    """Yield each page's includes.media list, paginating until exhausted or max_reads hit.
    reads_state = mutable [posts_read] counter (each returned post = one read)."""
    token_param = None
    while reads_state[0] < max_reads:
        url = base_url
        if token_param:
            url += f"&pagination_token={token_param}" if "/users/" in base_url else f"&next_token={token_param}"
        data = api_get(url, token)
        posts = data.get("data", []) or []
        reads_state[0] += len(posts)
        media = data.get("includes", {}).get("media", []) or []
        yield media
        meta = data.get("meta", {})
        token_param = meta.get("next_token")
        if not token_param or not posts:
            return


def save_media(media_list, out, min_short, saved_state, reads_state):
    for m in media_list:
        if m.get("type") != "photo":
            continue
        w, h = m.get("width", 0), m.get("height", 0)
        if min(w, h) < min_short:
            continue
        key = m.get("media_key")
        url = m.get("url")
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
            reads_state[1] += 1  # counted toward the conservative $ estimate (may overcount; CDN downloads are free)
            saved_state[0] += 1
        except Exception as e:
            sys.stderr.write(f"  download failed {key}: {e!r}\n")


def main():
    ap = argparse.ArgumentParser(description="Download high-res X photos into data/v9_x/.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--handles", help="comma-separated account handles (no @) for timeline mode")
    g.add_argument("--query", help="search query (recent 7d, or --full-archive)")
    ap.add_argument("--full-archive", action="store_true", help="use full-archive search (back to 2006)")
    ap.add_argument("--max-reads", type=int, default=2000, help="cap posts fetched (cost guard)")
    ap.add_argument("--min-short", type=int, default=1536, help="min short side (px) to keep")
    ap.add_argument("--out", default="data/v9_x")
    args = ap.parse_args()

    token = get_token()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved_state = [0]            # images saved
    reads_state = [0, 0]         # [posts_read, media_read]

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
            base = f"{API}/users/{uid}/tweets?exclude=replies,retweets&{_media_params()}"
            for media in iter_pages(base, token, args.max_reads, reads_state):
                save_media(media, out, args.min_short, saved_state, reads_state)
                print(f"  saved={saved_state[0]} posts_read={reads_state[0]} "
                      f"~${reads_state[0]*PRICE_READ + reads_state[1]*PRICE_READ:.2f}")
    else:
        endpoint = "tweets/search/all" if args.full_archive else "tweets/search/recent"
        base = f"{API}/{endpoint}?query={urllib.parse.quote(args.query)}&{_media_params()}"
        for media in iter_pages(base, token, args.max_reads, reads_state):
            save_media(media, out, args.min_short, saved_state, reads_state)
            print(f"  saved={saved_state[0]} posts_read={reads_state[0]} "
                  f"~${reads_state[0]*PRICE_READ + reads_state[1]*PRICE_READ:.2f}")

    cost = reads_state[0] * PRICE_READ + reads_state[1] * PRICE_READ
    print(f"\nDone. Saved {saved_state[0]} images (>= {args.min_short}px) -> {out}/")
    print(f"Reads: {reads_state[0]} posts + {reads_state[1]} media  ~= ${cost:.2f} (estimate)")
