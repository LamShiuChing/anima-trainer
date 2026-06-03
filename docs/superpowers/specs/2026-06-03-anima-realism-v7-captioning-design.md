# Anima Realism V7 — captioning overhaul (design + prompt guide)

> Date: 2026-06-03 · Branch: `v5-build` · Pairs with V7 training (1536 + higher LR + more/originals data).
> Supersedes the v5/v6 caption format. Code: `src/gemini_caption.py`, `src/03_caption.py`, `config/pipeline.yaml`.

## Principle

**Caption == inference prompt.** Every enum value you train on becomes a handle the user types. So the
caption schema *is* the prompt language. Design them together.

## Decisions (locked 2026-06-03)

- **No `realistic photo` anchor** — 100% photo data ⇒ a token on every image carries no discriminative
  signal. (Safe only if V7 **warm-starts from the v6 keeper**; from-base would lose the anti-anime switch.)
- **Enums = photographic/style control. WD14 booru tags + rich NL = content** (person/hair/clothes/
  accessories/setting — too open-ended to enumerate).
- **Populate-or-dead:** a value needs ~50–100+ training images to become a real trigger. Adding *axes* is
  cheap (each axis's values are seen across the whole set); only rare *values within* an axis stay weak.
- **Prefer booru-native vocab** (Anima base has priors → strong, cheap triggers): quality ladder, rating
  ladder, `from above/below`, `looking at viewer`, `breast` ladder, etc.
- **WD14 → `EVA02_Large` @ general_threshold ~0.25** (more detail; key for the ~43% explicit images Gemini
  refuses — booru anatomical tags are the NSFW caption richness). Tradeoff: more noise tags.
- **Rating via Gemini** (booru ladder) replaces binary safe/explicit; Falconsai is the fallback when Gemini
  returns nothing. **NSFW adult: no local block** (BLOCK_NONE + tags-only fallback, already true).
- 🚫 **WD14 underage hard-block KEPT** — non-negotiable. No-op on an all-adult dataset; backstop against a
  single bad file. Not a quality knob.
- **Gemini NL richer** (head-to-toe features + background/objects/materials); `max_output_tokens` 256→450.

## Vocabulary (the control axes)

Single-pick string enums unless noted. `resolution` is **derived from pixel size in stage 1, not Gemini.**

| Slot | Pick | Values |
|---|---|---|
| shot_type | 1 (req) | extreme close-up · close-up · portrait · upper body · cowboy shot · full body · wide shot |
| view | 0–1 | front view · three-quarter view · profile view · back view · looking over shoulder · looking at viewer · looking away |
| camera_angle | 0–1 | eye level · from above · from below · overhead · dutch angle |
| quality | 1 (req) | masterpiece · best quality · high quality · normal quality · low quality · worst quality |
| resolution | auto | absurdres (≥2048 short side) · highres (≥1024) · '' (768–1024) |
| capture_style | 1 (req) | amateur snapshot · casual phone photo · social media selfie · candid photo · semi-professional · professional photograph · editorial photography · studio portrait |
| lighting | 0–2 | direct flash · natural daylight · golden hour · blue hour · overcast flat light · indoor artificial light · low light · soft window light · studio lighting · backlit · rim light · neon lighting · harsh sunlight · ring light · candlelight |
| condition | 0–2 | sharp focus · soft focus · grainy / high ISO · motion blur · compressed / low-res · overexposed · underexposed · lens flare · chromatic aberration · vignette · jpeg artifacts · red-eye |
| color_grade | 0–1 | natural color · warm tones · cool tones · muted · vibrant · high contrast · film grain · film look · black and white · sepia · faded · teal and orange |
| camera_lens | 0–1 | phone camera · compact camera · DSLR · 85mm bokeh · 50mm · 35mm · wide-angle · fisheye · macro · film camera |
| depth_of_field | 0–1 | shallow depth of field · deep focus |
| expression | 0–1 | neutral expression · smile · laughing · serious · seductive · surprised · crying · pout · open mouth |
| body_type | 0–1 | slim · average build · athletic · curvy · plus-size · muscular · petite |
| breast_size | 0–1 | flat chest · small breasts · medium breasts · large breasts · huge breasts |
| ethnicity | 0–1 | east asian · southeast asian · south asian · white · black · hispanic · middle eastern · mixed |
| skin_tone | 0–1 | fair skin · light skin · olive skin · tan skin · brown skin · dark skin |
| setting_type | 1 (req-ish) | bedroom · living room · kitchen · bathroom · studio · office · city street · nature · beach · pool · cafe · restaurant · bar · gym · car · party |
| rating | 1 (req) | rating:general · rating:sensitive · rating:questionable · rating:explicit |

Content (NOT enums): **WD14 EVA02_Large tags** (hair, clothes, body, objects, explicit anatomy) + **Gemini NL**
(detailed head-to-toe + materials + accessories + background).

## Assembled caption order

```
<shot_type>, <view>, <camera_angle>, <quality>, <resolution>, <capture_style>, <lighting..>, <condition..>,
<color_grade>, <camera_lens>, <depth_of_field>, <expression>, <body_type>, <breast_size>, <ethnicity>,
<skin_tone>, <setting_type>, <rating>, <wd14 tags>[, watermark], <NL description>
```
Empty slots are omitted. `rating` = Gemini's, else Falconsai fallback. No anchor.

## Pipeline changes

- `src/gemini_caption.py`: new `VOCAB`, `SINGLE_SLOTS`/`ARRAY_SLOTS`, `build_schema`, `build_prompt`,
  `coerce_response`, `assemble_caption(parts, wd14_tags, resolution, fallback_rating)`, `resolution_tag(w,h)`.
  `ANCHOR` removed.
- `src/03_caption.py`: pass derived resolution + Falconsai-fallback rating into `assemble_caption`; `quality`
  (not `quality_level`) for the gemini-ok count; underage block unchanged.
- `config/pipeline.yaml`: `wd_model_name: EVA02_Large`, `wd_general_threshold: 0.25`,
  `gemini.max_output_tokens: 450`, `nsfw_label_map` → `rating:general`/`rating:explicit`.

---

# TLDR — user prompt guide (V7 model)

**Template** (drop any slot you don't care about; the model fills the rest):
```
<shot type>, <view>, <quality>, <capture style>, <lighting>, [color], [camera/lens], [expression],
[body type], [breast size], [ethnicity/skin], <setting>, <rating>, <plain-words: hair, clothes, accessories, pose, background>
```

**Rules**
1. **Use the exact words** from the tables (they're the trained handles): `full body` works, "whole body" won't.
2. **Strongest triggers first** (shot → quality → capture style → lighting), then attributes, then plain-words detail.
3. **Quality ladder:** `masterpiece` > `best quality` > `high quality` > `normal quality` > `low quality` > `worst quality`.
4. **Spice via rating:** `rating:general` (safe) → `rating:sensitive` → `rating:questionable` → `rating:explicit`.
5. **Negative prompt** (paste every time): `watermark, text, logo, anime, illustration, cartoon, 3d, cgi, render, lowres, worst quality, deformed`.
6. Body/face/clothes detail: type **booru tags** (`large breasts, abs, thick thighs, long hair, hoodie`) or **plain words** — both trained.

**Examples**
- Casual: `full body, front view, high quality, casual phone photo, natural daylight, phone camera, smile, slim, east asian, fair skin, city street, rating:general, a young woman with long black hair in an oversized hoodie and jeans holding an iced coffee, parked cars behind`
- Studio: `close-up, looking at viewer, masterpiece, studio portrait, studio lighting, 85mm bokeh, neutral expression, white, plain grey backdrop, rating:general, freckles, natural makeup, gold hoop earrings`
- Amateur flash: `upper body, front view, low quality, amateur snapshot, direct flash, ring light, athletic, tan skin, bathroom, rating:sensitive, mirror selfie, phone visible, red-eye`

> Note: this is the **V7** language. The current **v5/v6** model only knows the older subset (`<quality>, <capture style>, <lighting>, <condition>, safe, <tags>, <one sentence>`). The full vocab above works only after V7 retraining on the new captions.
