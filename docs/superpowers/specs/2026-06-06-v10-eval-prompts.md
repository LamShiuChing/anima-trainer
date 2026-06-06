# v10 eval prompts (frozen) — run at every saved epoch (ep5, 10, ... 50)

Inference defaults (HF + prior findings): res <=1536 (AR 0.66-1.5, dims /64), steps 30-50,
**CFG 4-5**, sampler `er_sde` / `euler_a` / `dpmpp_2m_sde_gpu`, scheduler `beta57`,
VAE = qwen_image_vae.safetensors. Caption format used in training = `masterpiece, best quality,
score_7, <rating>, <tags>`, so lead positive prompts with `masterpiece, best quality, score_7`.

## A. Photoreal set (does realism climb? — judge sharpness, lighting, skin, hands)
1. masterpiece, best quality, score_7, rating:general, woman, kitchen, window light, holding a mug, candid
2. masterpiece, best quality, score_7, rating:general, man, city street, overcast, walking, full body
3. masterpiece, best quality, score_7, rating:general, close-up portrait, freckles, soft window light
4. masterpiece, best quality, score_7, rating:general, two people, restaurant, evening, bokeh background
5. masterpiece, best quality, score_7, rating:general, hands holding a phone, detailed fingers
6. masterpiece, best quality, score_7, rating:general, living room interior, sofa, daylight, wide shot
neg: worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, compressed

## B. Concept-retention set (did base knowledge survive? — written in base-native format)
> Probe concepts/poses/objects the base knew. Watch for the epoch where these start degrading
> (losing the concept, collapsing to generic) -> pick the epoch JUST BEFORE that.
> NOTE: these use the base-native `safe` rating token on purpose — v10 TRAINS `rating:general`/
> `rating:explicit` (Falconsai), not `safe`. `safe` here probes the BASE model's prior, which is
> exactly what this set measures (did base knowledge survive). Not a mismatch.
1. masterpiece, best quality, score_7, safe, person in a maid outfit, indoor
2. masterpiece, best quality, score_7, safe, knight in plate armor, holding a sword, field
3. masterpiece, best quality, score_7, safe, person in a kimono, garden, cherry blossoms
4. masterpiece, best quality, score_7, safe, astronaut, space suit, on the moon
5. masterpiece, best quality, score_7, safe, person playing an electric guitar on stage
6. masterpiece, best quality, score_7, safe, chef cooking in a professional kitchen
neg: worst quality, low quality, score_1, score_2, score_3

## Decision rule
- Photoreal set should climb across epochs (history: undertrains -> keep going).
- Concept set: pick the LAST epoch where concepts still render correctly AND photoreal is strong.
- If concepts erode before photoreal arrives, reconsider re-adding a quality/task token next run.
