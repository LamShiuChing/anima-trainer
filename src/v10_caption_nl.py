"""Stage 3b (v10): Gemini NL caption (ported from lenstag-ai) appended to the RAM++ tag caption.

RAM++ tags alone are a flat keyword bag (underuses Anima's Qwen3 LLM text encoder). This adds a
detailed natural-language paragraph (subject / composition / lighting / colors / textures / mood /
technical) so the LLM TE gets real sentences. Reuses the existing RAM++ run: reads
data/v10_manifest.csv, snapshots the RAM++ caption into `caption_tags` (once), and rebuilds
`caption` = caption_tags + ", " + NL. Idempotent (always rebuilds from caption_tags + nl, never
double-appends). Concurrent (thread pool) + resumable cache (path -> nl) so reruns never re-bill.
NSFW handled via BLOCK_NONE on the 4 adjustable safety categories (Gemini 3 Flash).

Prompt ported verbatim from lenstag-ai's geminiService.ts (natural style).
Run:  python src/v10_caption_nl.py
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import json
import threading
import time
from pathlib import Path

import common

LOG = common.setup_logging()

# Ported verbatim from lenstag-ai geminiService.ts getSystemInstruction('natural').
SYSTEM_INSTRUCTION = (
    "You are a professional image captioning assistant for machine learning datasets "
    "(Stable Diffusion, Flux, Midjourney).\n"
    "Your goal is to provide a highly detailed, literal, and descriptive caption for the provided image.\n"
    "Focus on: Subject matter, Composition, Lighting, Colors, Textures, Mood, and Technical details.\n"
    "Format: Write a natural language paragraph (50-100 words). Do not use flowery or subjective language.\n"
    "Output ONLY the caption text. No preamble, no conversational fillers."
)
USER_PROMPT = "Generate a training caption for this image."

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


# ---- pure logic (unit-tested) ----

def mime_for(path):
    return MIME.get(Path(path).suffix.lower(), "image/jpeg")


def normalize_nl(text):
    """Collapse all whitespace/newlines to single spaces and strip. None/empty -> ''."""
    return " ".join((text or "").split()).strip()


def assemble_full(caption_tags, nl):
    """Final caption = RAM++ tag caption + ', ' + normalized NL (NL omitted if empty)."""
    base = (caption_tags or "").strip().rstrip(",").strip()
    nl = normalize_nl(nl)
    return f"{base}, {nl}" if nl else base


# ---- cache ----

def load_cache(path):
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path, cache):
    Path(path).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


# ---- Gemini wrapper (not unit-tested: network) ----

class GeminiNL:
    def __init__(self, cfg):
        c = cfg["caption"]["nl"]
        self.model = c.get("model", "gemini-3-flash-preview")
        self.concurrency = int(c.get("concurrency", 8))
        self.max_output_tokens = int(c.get("max_output_tokens", 300))
        self.block_none = bool(c.get("block_none", True))
        self.max_retries = 4
        self._client = None
        self._client_lock = threading.Lock()

    def _client_lazy(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    from google import genai
                    self._client = genai.Client()      # reads GEMINI_API_KEY
        return self._client

    def caption(self, path):
        """One Gemini NL call. Returns normalized text ('' on refusal/empty). Raises on hard error."""
        from google.genai import types
        client = self._client_lazy()
        none = types.HarmBlockThreshold.BLOCK_NONE
        safety = [types.SafetySetting(category=cat, threshold=none) for cat in (
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )] if self.block_none else None
        img = types.Part.from_bytes(data=Path(path).read_bytes(), mime_type=mime_for(path))
        kwargs = dict(
            system_instruction=SYSTEM_INSTRUCTION,
            safety_settings=safety,
            temperature=0.7, top_p=0.95, top_k=40,
            max_output_tokens=self.max_output_tokens,
        )
        # Gemini 3 is a THINKING model: thinking tokens consume max_output_tokens and truncate the
        # caption. Disable thinking so the whole budget goes to the response (no-op on non-thinking models).
        if hasattr(types, "ThinkingConfig"):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        cfg = types.GenerateContentConfig(**kwargs)
        last = None
        for attempt in range(self.max_retries):
            try:
                resp = client.models.generate_content(
                    model=self.model, contents=[img, USER_PROMPT], config=cfg)
                return normalize_nl(resp.text or "")
            except Exception as e:           # rate-limit/5xx -> backoff + retry; final raises -> caller handles
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))
        raise last


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()                  # GEMINI_API_KEY from .env
    except ImportError:
        pass
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    cfg = common.load_config()
    nl_cfg = cfg["caption"]["nl"]
    cache_file = nl_cfg["cache_file"]
    manifest = cfg["paths"]["manifest"]
    rows = common.read_manifest(manifest)
    kept = [r for r in rows if r.get("dropped") == "False" and r.get("caption")]
    LOG.info("Stage 3b (v10): Gemini NL for %d captioned images (model=%s concurrency=%s)",
             len(kept), nl_cfg.get("model"), nl_cfg.get("concurrency"))

    for r in kept:                     # snapshot the RAM++ caption once -> idempotent rebuild base
        if not r.get("caption_tags"):
            r["caption_tags"] = r["caption"]

    cache = load_cache(cache_file)
    gem = GeminiNL(cfg)

    # pre-flight: one real call so a bad key/model aborts LOUDLY before the pool spins up
    todo = [r for r in kept if r["path"] not in cache and Path(r["path"]).exists()]
    if todo:
        try:
            sample = gem.caption(todo[0]["path"])
        except Exception as e:
            raise RuntimeError(f"Gemini NL pre-flight failed: {e!r}. Fix key/model before the full run "
                               f"(no captions changed).") from e
        cache[todo[0]["path"]] = sample
        LOG.info("pre-flight OK. sample NL:\n  %s", (sample or "<refused/empty>")[:200])

    ok = refused = 0
    bar = tqdm(total=len(kept), desc="gemini-nl", unit="img", dynamic_ncols=True)

    def work(r):
        path = r["path"]
        if path in cache:
            return path, cache[path]
        if not Path(path).exists():
            return path, ""
        try:
            return path, gem.caption(path)
        except Exception as e:
            tqdm.write(f"  NL fail {Path(path).name}: {e}")
            return path, None          # None -> don't cache; retried next run

    try:
        with ThreadPoolExecutor(max_workers=max(1, gem.concurrency)) as ex:
            for i, (path, nl) in enumerate(ex.map(work, kept), 1):
                bar.update(1)
                if nl is None:
                    continue
                cache[path] = nl       # main thread only -> safe
                if nl:
                    ok += 1
                else:
                    refused += 1
                if i % 100 == 0:
                    save_cache(cache_file, cache)
    finally:
        bar.close()
        save_cache(cache_file, cache)  # persist even on Ctrl-C (resumable)

    for r in kept:                     # rebuild final captions from caption_tags + cached NL
        r["caption"] = assemble_full(r["caption_tags"], cache.get(r["path"], ""))
    common.write_manifest(manifest, rows)
    LOG.info("Stage 3b done. nl_ok=%d refused/empty=%d (of %d). captions rebuilt -> %s. "
             "Next: python src/04_build_dataset.py", ok, refused, len(kept), manifest)


if __name__ == "__main__":
    main()
