# v8 raw sourcing — drop your curated internet images here

Three buckets. Sort each downloaded image by **what it is**. Source doesn't matter
(Unsplash / Pixabay / Pexels manual downloads AND the FiftyOne Open-Images script all
land here). Filename + format don't matter (jpg/png/webp all fine — webp is converted
to jpg downstream).

```
data/v8_raw/
  detail/   60%  — close-ups where the TARGET DETAIL fills the frame + is tack-sharp:
                   hands, hand+phone, fabric/clothing texture, footwear/feet, held objects.
                   This is the high-frequency fuel. Detail must be LARGE in frame.
  anchor/   35%  — clean-but-CASUAL whole-person shots (phone-style candids shot on a good
                   camera). Holds the amateur look + identity variety + anti-drift.
  bg/        5%  — interiors / scenes / environments. Environment fidelity. Mostly filled
                   by the FiftyOne script + Poly Haven photographic renders.
```

## Automated sourcing (high-res)

- **`scripts/v8_fetch_pexels.py` — USE THIS.** Pexels API, originals 3000–6000px, filtered ≥1536 before
  download. Needs a free `PEXELS_API_KEY` in `.env`. Run per bucket+query (see the script header for
  query ideas). This is the real high-fidelity fuel, esp. for `detail/`.
- **`scripts/v8_fetch_openimages.py` — limited.** Open Images caps at ~1024px longest side, so almost
  nothing clears the ≥1536 gate. Kept for reference / variety only; do **not** rely on it for v8.
- **Manual** Unsplash/Pixabay/Pexels downloads → sort into the buckets by hand (best curation control).

## Hard rules before you drop an image in

1. **Curate at 100% zoom.** Reject anything soft, upscaled, or re-compressed. **Stated
   resolution is NOT enough** — a 4000px image that's actually a soft upscale is garbage
   for this run (this is the exact bug v8 exists to fix). Zoom to 100%, look at edges/pores
   /fabric weave. If it's mushy, skip it.
2. **≥1536 px on the short side.** Smaller = no real detail to learn. (The curation script
   enforces this too, but pre-filtering saves you downloads.)
3. **ALL clean / high-fidelity.** No compressed / JPEG-blocky / screenshotted images. Zero.
   (Compression is the disease we're removing — don't feed it back.)
4. **Amateur, not studio** (for `anchor/` especially): candid, casual framing. Avoid glossy
   stock/studio portraits or the amateur look drifts. Clean ≠ studio.
5. **Feet caveat** (`detail/`): foot/toe close-ups online skew fetish-stock (oily/studio/weird
   angles). Curate hard or keep the count low.

## Don't worry about

- Aspect ratio — the curation step crops to the safe AR 0.66–1.5 (Anima pos-emb limit).
- Exact 60/35/5 — get close; the curation script reports actual counts and warns if off.
- Renaming — keep original names; downstream stages handle collisions.

## What happens next (automated)

`data/v8_raw/{detail,anchor,bg}/` → curation (≥1536 + sharpness + dedup + AR-crop, carries the
bucket label) → `data/v8_clean/` → captioning (WD14 + Gemini) → `data/v8_dataset/` + dataset.toml.
