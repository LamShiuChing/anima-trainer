# v8 frozen eval set — fidelity refiner (compression-erase + detail, keep amateur)

> Run these EXACT prompts + seeds + settings on the **ep17 baseline** AND **every v8 epoch** (1..N).
> Compare side-by-side. v8 uses the **V7 caption vocab** (no `realistic photo` anchor) — these words are
> the trained handles. Goal of the read: **cleaner pixels + sharper fingers/fabric/feet, with the amateur
> look intact.** Loss is blind here — judge ONLY these images.

## Fixed settings (keep constant across ALL model states — constancy > the exact values)
- sampler: `euler` (also try `dpmpp_2m`) · steps: 28 · **CFG: 4.0** (Anima oversaturates high; stay 3.5–4.5) · optional RescaleCFG 0.7
- resolution: 1536×1536 (or portrait ~1152×1536) · VAE: `qwen_image_vae.safetensors`
- seeds (run every prompt at ALL three): **42, 1234, 7777**
- **Negative = the fidelity dial (paste every time):**
  `jpeg artifacts, compressed, blurry, soft focus, lowres, worst quality, watermark, text, anime, illustration, cartoon, 3d, cgi, render, deformed, bad hands, extra fingers`
  *(the `jpeg artifacts / compressed / blurry / soft focus / lowres` tokens are what pull output off the learned compression — confirm they bite.)*

---

## A. DETAIL — the fidelity targets (fingers / fabric / feet)
1. **Hands + phone, macro**
   `extreme close-up, looking at viewer, best quality, casual phone photo, soft window light, sharp focus, macro, shallow depth of field, fair skin, rating:general, 1girl, hands, holding phone, detailed fingers, painted nails`
2. **Fabric texture**
   `extreme close-up, best quality, semi-professional, soft window light, sharp focus, macro, rating:general, knit sweater, wool texture, detailed fabric weave`
3. **Feet / footwear**
   `close-up, from above, best quality, casual phone photo, natural daylight, sharp focus, fair skin, rating:general, 1girl, bare feet, toes, wooden floor`
4. **Generic hands**
   `extreme close-up, best quality, professional photograph, soft window light, sharp focus, 85mm bokeh, rating:general, 1woman, hands, fingers, manicure, holding coffee cup`

**Look for:** crisp fingernails / knuckle creases / fabric weave vs ep17's mush. This is the core win.

## B. YOUR COMPOSITIONS + amateur (de-compression + drift canaries)
5. **Mirror selfie** (your key composition — the soft-output complaint)
   `upper body, looking at viewer, best quality, social media selfie, indoor artificial light, sharp focus, phone camera, slim, fair skin, bedroom, rating:general, 1girl, mirror selfie, phone visible, crop top, blue jeans`
6. **Casual full-body** (drift check — must still read amateur, not stock)
   `full body, front view, high quality, casual phone photo, natural daylight, sharp focus, athletic, fair skin, city street, rating:general, 1girl, white tank top, blue jeans, holding coffee`
7. **Amateur snapshot** (the amateur-look canary — if this goes glossy, you've drifted)
   `full body, high quality, amateur snapshot, direct flash, sharp focus, curvy, fair skin, bedroom, rating:general, 1woman, summer dress, standing`

**Look for:** #5 cleaner + sharper than ep17 (de-compression working) while #6/#7 STILL look candid/amateur (no stock drift). #7 going glossy = stop signal.

## C. BACKGROUND / scene
8. **Interior**
   `wide shot, best quality, professional photograph, soft window light, sharp focus, deep focus, living room, rating:general, sofa, coffee table, houseplant, window`

**Look for:** sharper environment detail; no warping.

---

## Add YOUR real prompts
You know your actual outputs. **Add 3–5 prompts that reproduce the exact soft/compressed images you want
fixed** (mirror selfies, your poses, your settings). Run them on ep17 vs each v8 epoch — that's the truest
read of whether v8 fixed *your* problem.

## How to call it
- **Win:** detail (A) sharper + your compositions (B5) cleaner vs ep17, while B6/B7 stay amateur.
- **Stop signals:** (1) B7/B6 drifting glossy/stock = aesthetic drift → pick an earlier epoch / lower LR;
  (2) textures sharpen but composition variety collapses across seeds = overfit → pick the prior epoch.
- Expected best around **epoch 3–6**. Pick best, DOWNLOAD + verify ~4.18 GB single-file + full `train_v8.log`
  BEFORE destroying the instance.
