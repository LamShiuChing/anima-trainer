"""v8 high-res sourcing — download ORIGINAL-resolution photos from the Pexels API into
data/v8_raw/<bucket>/. Unlike Open Images (capped at ~1024px), Pexels originals are 3000-6000px,
so this is the real high-fidelity fuel for v8.

Setup (one time, ~1 min):
  1. Get a free API key: https://www.pexels.com/api/  (sign in -> "Your API Key")
  2. Add it to .env in the project root (same file as GEMINI_API_KEY):
        PEXELS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  3. pip install requests   (only if you don't already have it; stdlib urllib is used, no install needed)

Run (one bucket + query at a time):
  python scripts/v8_fetch_pexels.py --bucket detail --query "hands holding smartphone" --max 60
  python scripts/v8_fetch_pexels.py --bucket detail --query "knitted sweater texture" --max 40
  python scripts/v8_fetch_pexels.py --bucket detail --query "bare feet" --max 30
  python scripts/v8_fetch_pexels.py --bucket anchor --query "candid portrait natural light" --max 80
  python scripts/v8_fetch_pexels.py --bucket bg     --query "living room interior" --max 30

Suggested queries per bucket:
  detail: "human hands close up", "hand holding phone", "fingers manicure", "denim texture",
          "knitted wool sweater", "leather bag closeup", "bare feet", "sandals feet"
  anchor: "candid portrait", "casual street style", "natural light woman", "everyday people candid"
  bg:     "living room interior", "bedroom interior", "cafe interior", "kitchen interior"

Notes:
  - Filters min(width,height) >= --min-short (default 1536) using the API's own dimensions BEFORE download.
  - Downloads src.original (full res). Skips IDs already on disk (safe to re-run / paginate further).
  - Pexels skews professional -> for `anchor` curate hard for the AMATEUR look; for `detail` polished+sharp
    is fine (fidelity is the point; aesthetic is handled by captioning).
  - Free tier: 200 requests/hr. The script prints remaining quota from the response headers.
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.pexels.com/v1/search"


def parse_args():
    ap = argparse.ArgumentParser(description="Download high-res Pexels photos into data/v8_raw/<bucket>.")
    ap.add_argument("--bucket", required=True, choices=["detail", "anchor", "bg"])
    ap.add_argument("--query", required=True, help="search term")
    ap.add_argument("--max", type=int, default=60, help="max images to SAVE this run")
    ap.add_argument("--min-short", type=int, default=1536, help="min short side (px) to keep")
    ap.add_argument("--orientation", default=None, choices=["landscape", "portrait", "square"])
    ap.add_argument("--out", default="data/v8_raw")
    return ap.parse_args()


def get_key():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        sys.stderr.write("PEXELS_API_KEY not set. Get a free key at https://www.pexels.com/api/ "
                         "and add it to .env as PEXELS_API_KEY=...\n")
        sys.exit(1)
    return key


def search_page(key, query, page, per_page, orientation):
    params = {"query": query, "per_page": per_page, "page": page}
    if orientation:
        params["orientation"] = orientation
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params),
                                 headers={"Authorization": key})
    with urllib.request.urlopen(req, timeout=30) as r:
        remaining = r.headers.get("X-Ratelimit-Remaining")
        data = json.loads(r.read().decode("utf-8"))
    return data, remaining


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "anima-v8/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def main():
    args = parse_args()
    key = get_key()
    out = Path(args.out) / args.bucket
    out.mkdir(parents=True, exist_ok=True)

    saved, scanned, too_small, page = 0, 0, 0, 1
    while saved < args.max:
        try:
            data, remaining = search_page(key, args.query, page, 80, args.orientation)
        except Exception as e:
            sys.stderr.write(f"API error on page {page}: {e!r}\n")
            break
        photos = data.get("photos", [])
        if not photos:
            print("No more results.")
            break
        for ph in photos:
            if saved >= args.max:
                break
            scanned += 1
            w, h = ph.get("width", 0), ph.get("height", 0)
            if min(w, h) < args.min_short:
                too_small += 1
                continue
            dest = out / f"pexels_{ph['id']}.jpg"
            if dest.exists():
                continue
            try:
                download(ph["src"]["original"], dest)
                saved += 1
            except Exception as e:
                sys.stderr.write(f"download failed for {ph['id']}: {e!r}\n")
        print(f"page {page}: saved={saved}/{args.max} scanned={scanned} "
              f"too_small={too_small} quota_left={remaining}")
        if not data.get("next_page"):
            print("Reached last page of results.")
            break
        page += 1

    print(f"\nDone. Saved {saved} images (>= {args.min_short}px short side) -> {out}/")
    if too_small:
        print(f"Skipped {too_small} below {args.min_short}px. Try a broader query for more high-res hits.")


if __name__ == "__main__":
    main()
