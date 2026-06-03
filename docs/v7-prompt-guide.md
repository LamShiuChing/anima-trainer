# Anima V7 — Prompt Guide (TLDR)

Quick reference for prompting the **V7** model (trained on the new enum-vocab captions, **no `realistic photo`
anchor**, trained at **1536**). Vocab is the exact set the captioner used — these words ARE the trained handles.

---

## 10 test prompts (paste-ready)

Each probes specific axes. Use the **same seed** across them so you can compare. Prompts 8→9→10 are a
**rating-ladder test** (same scene, only the rating changes — tells you if spice control works).

1. **Studio face / close-up**
   `close-up, looking at viewer, best quality, studio portrait, studio lighting, 85mm bokeh, neutral expression, slim, east asian, fair skin, studio, rating:general, 1girl, long black hair, natural makeup, gold earrings, plain grey backdrop`

2. **Casual full-body street**
   `full body, front view, high quality, casual phone photo, natural daylight, phone camera, smile, athletic, white, fair skin, city street, rating:general, 1girl, blonde ponytail, white tank top, blue jeans, holding coffee, parked cars behind`

3. **Golden-hour professional**
   `upper body, three-quarter view, masterpiece, professional photograph, golden hour, warm tones, 85mm bokeh, serious, curvy, hispanic, tan skin, nature, rating:general, 1woman, wavy brown hair, white sundress, field behind`

4. **Social-media selfie**
   `upper body, looking at viewer, high quality, social media selfie, ring light, phone camera, smile, slim, southeast asian, light skin, bedroom, rating:sensitive, 1girl, mirror selfie, crop top, phone visible`

5. **Editorial / fashion**
   `full body, front view, masterpiece, editorial photography, studio lighting, high contrast, cool tones, serious, petite, black, dark skin, studio, rating:general, 1woman, short hair, designer outfit, dramatic pose`

6. **Film look, indoors**
   `cowboy shot, three-quarter view, high quality, semi-professional, soft window light, film grain, film look, neutral expression, average build, white, fair skin, living room, rating:general, 1woman, red hair, knit sweater, sitting on sofa`

7. **Back view, beach**
   `full body, back view, high quality, professional photograph, natural daylight, sharp focus, athletic, middle eastern, olive skin, beach, rating:sensitive, 1woman, long dark hair, bikini, looking over shoulder, ocean behind`

8. **Rating ladder — general**
   `full body, front view, high quality, casual phone photo, soft window light, curvy, white, fair skin, bedroom, rating:general, 1woman, brown hair, summer dress, standing by the window`

9. **Rating ladder — sensitive** (same scene)
   `full body, front view, high quality, casual phone photo, soft window light, curvy, white, fair skin, bedroom, rating:sensitive, 1woman, brown hair, lingerie, standing by the window`

10. **Rating ladder — explicit** (same scene; legal adult)
    `full body, front view, high quality, casual phone photo, soft window light, curvy, white, fair skin, bedroom, rating:explicit, 1woman, brown hair, nude, standing by the window`

**Negative prompt (paste every time):**
`watermark, text, logo, anime, illustration, cartoon, drawing, 3d, cgi, render, lowres, worst quality, deformed, bad hands, extra fingers`

**Inference settings (starting point):** 1536×1536 (or portrait ~1152×1536) since it trained at 1536 · sampler euler / dpmpp_2m · 25–30 steps · CFG 3.5–5. Tune from there.

---

## Prompt structure

Lead with the strongest handles, then attributes, then plain-words content. Drop any slot you don't care about:

```
<shot_type>, <view>, <camera_angle>, <quality>, <capture_style>, <lighting>, <condition>, <color_grade>,
<camera_lens>, <depth_of_field>, <expression>, <body_type>, <breast_size>, <ethnicity>, <skin_tone>,
<setting_type>, <rating>, <booru content tags: hair/clothes/body/accessories>, <plain-words scene>
```

**Rules**
1. **No `realistic photo` anchor** — this model is realism by default; don't add it.
2. **Use the exact enum words** (table below). `full body` triggers; "whole body" doesn't.
3. **Order helps but isn't rigid** — leading with shot_type → quality → capture_style → lighting gives the firmest control.
4. **Content (hair, clothes, body, accessories) = booru tags or plain words** (e.g. `long hair, hoodie, large breasts, thick thighs, holding phone`). Both are trained.
5. **Spice via rating:** `rating:general` → `rating:sensitive` → `rating:explicit` (questionable is weak — see below).
6. **resolution tags** (`highres`, `absurdres`) are valid handles too (they were in captions) — usually leave them out.

---

## Full enum vocabulary

| Axis | Pick | Values |
|---|---|---|
| **shot_type** | 1 | extreme close-up · close-up · portrait · upper body · cowboy shot · full body · wide shot |
| **view** | 0–1 | front view · three-quarter view · profile view · back view · looking over shoulder · looking at viewer · looking away |
| **camera_angle** | 0–1 | eye level · from above · from below · overhead · dutch angle |
| **quality** | 1 | masterpiece · best quality · high quality · normal quality · low quality · worst quality |
| **capture_style** | 1 | amateur snapshot · casual phone photo · social media selfie · candid photo · semi-professional · professional photograph · editorial photography · studio portrait |
| **lighting** | 0–2 | direct flash · natural daylight · golden hour · blue hour · overcast flat light · indoor artificial light · low light · soft window light · studio lighting · backlit · rim light · neon lighting · harsh sunlight · ring light · candlelight |
| **condition** | 0–2 | sharp focus · soft focus · grainy / high ISO · motion blur · compressed / low-res · overexposed · underexposed · lens flare · chromatic aberration · vignette · jpeg artifacts · red-eye |
| **color_grade** | 0–1 | natural color · warm tones · cool tones · muted · vibrant · high contrast · film grain · film look · black and white · sepia · faded · teal and orange |
| **camera_lens** | 0–1 | phone camera · compact camera · DSLR · 85mm bokeh · 50mm · 35mm · wide-angle · fisheye · macro · film camera |
| **depth_of_field** | 0–1 | shallow depth of field · deep focus |
| **expression** | 0–1 | neutral expression · smile · laughing · serious · seductive · surprised · crying · pout · open mouth |
| **body_type** | 0–1 | slim · average build · athletic · curvy · plus-size · muscular · petite |
| **breast_size** | 0–1 | flat chest · small breasts · medium breasts · large breasts · huge breasts |
| **ethnicity** | 0–1 | east asian · southeast asian · south asian · white · black · hispanic · middle eastern · mixed |
| **skin_tone** | 0–1 | fair skin · light skin · olive skin · tan skin · brown skin · dark skin |
| **setting_type** | 1 | bedroom · living room · kitchen · bathroom · studio · office · city street · nature · beach · pool · cafe · restaurant · bar · gym · car · party |
| **rating** | 1 | rating:general · rating:sensitive · rating:questionable · rating:explicit |

Plus: **booru content tags** (hair/clothes/body/objects — anything WD14 emits) + a **plain-words scene description**.

---

## Known weak/dead triggers (this dataset)

Honest expectations — these had too few training examples to fire reliably:
- **`grainy / high ISO`, `soft focus`, `motion blur`** — near-dead (1–4 imgs each). The blur gate removed soft/grainy shots, so these won't trigger. Only **`sharp focus`** works in the condition axis.
- **`rating:questionable`** — thin (~22 imgs). Use `sensitive` or `explicit` instead.
- **`close-up`** — thinner (~27); `portrait` / `upper body` are stronger for faces.
- Everything else (shot_type full/upper/cowboy, all ratings general/sensitive/explicit, lighting, capture styles, color grades, body types, ethnicities, settings) is well-populated and should respond.

> This is the **V7** language. The old v5/v6 model uses a different format (`realistic photo, masterpiece, ...`) — don't mix.
