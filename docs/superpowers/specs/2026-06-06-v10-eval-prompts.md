# v10 eval prompts (frozen) — run at every saved epoch (ep5, 10, ... 50)

Inference defaults (HF + prior findings): res <=1536 (AR 0.66-1.5, dims /64), steps 30-50,
**CFG 4-5**, sampler `er_sde` / `euler_a` / `dpmpp_2m_sde_gpu`, scheduler `beta57`,
VAE = qwen_image_vae.safetensors.

Caption format trained in v10 = `<quality>, <rating>, <enums>, <tags>, <paragraph>` where quality is
a single Gemini-judged booru-ladder token (masterpiece/best quality/high quality/...) and rating ∈
{safe, suggestive, explicit} (no `rating:` prefix, no `score_7`). So lead positive prompts with one
quality token (`masterpiece`) + the rating you want, then describe with tags/sentence.

## A. Photoreal set (does realism climb? — judge sharpness, lighting, skin, hands)
1. masterpiece, safe, candid photo, woman, kitchen, soft window light, holding a mug
2. masterpiece, safe, full body, man, city street, overcast flat light, walking
3. masterpiece, safe, close-up, portrait, freckles, soft window light, sharp focus
4. masterpiece, safe, two people, restaurant, evening, shallow depth of field
5. masterpiece, safe, close-up, hands holding a phone, detailed fingers
6. masterpiece, safe, wide shot, living room interior, sofa, natural daylight
neg: worst quality, low quality, blurry, jpeg artifacts, compressed

## B. Concept-retention set (did base knowledge survive?)
> Probe concepts/poses/objects the base knew. Watch for the epoch where these start degrading
> (losing the concept, collapsing to generic) -> pick the epoch JUST BEFORE that.
> (`safe` is a v10-trained rating token, so these prompts match the training distribution.)
1. masterpiece, safe, person in a maid outfit, indoor
2. masterpiece, safe, knight in plate armor, holding a sword, field
3. masterpiece, safe, person in a kimono, garden, cherry blossoms
4. masterpiece, safe, astronaut, space suit, on the moon
5. masterpiece, safe, person playing an electric guitar on stage
6. masterpiece, safe, chef cooking in a professional kitchen
neg: worst quality, low quality

## Decision rule
- Photoreal set should climb across epochs (history: undertrains -> keep going).
- Concept set: pick the LAST epoch where concepts still render correctly AND photoreal is strong.
- If concepts erode before photoreal arrives, reconsider a stronger quality/task token next run.
