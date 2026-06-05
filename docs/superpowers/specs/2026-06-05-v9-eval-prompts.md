# Anima Realism v9 — frozen eval prompts

> Run on EVERY saved epoch + the **`v8_epoch10` baseline** (apples-to-apples: did v9 beat ep10?).
> Loss is BLIND (flat-noise across v5-v8). Judge ONLY by eval images. Fixed seeds across all epochs.

## Inference settings (Anima flow-matching DiT)
- **CFG 3.0-4.5** (high CFG oversaturates — that's a CFG signature, not undertraining), optional RescaleCFG ~0.7.
- Sampler `euler` or `dpmpp_2m` + `simple`/`beta`, steps 20-30.
- VAE = `qwen_image_vae.safetensors`. Generate at ~1536 area (1536x1536 / 1344x1728 / 1856x1280; AR 0.66-1.5, dims ÷64).
- **Fidelity/sharp-bg dial:** prompt append `amateur snapshot, best quality, highres, sharp` +
  negative `jpeg artifacts, compressed, blurry, low quality, bokeh, blurred background`.

## (a) BACKGROUND canaries — the core v9 target
1. `amateur snapshot, a woman taking a selfie in a detailed living room, bookshelf and window and plants behind her, best quality, highres, sharp`
2. `casual phone photo, mirror selfie of a person in a messy bedroom, clothes and posters on the wall in focus, sharp`
3. `amateur snapshot, full body portrait of a person on a busy city street, shops and signs and people in the background, deep focus, sharp`
4. `casual phone photo, a person standing in a kitchen, cabinets and appliances and counter clutter visible and sharp`
Judge: is the BACKGROUND sharp + coherent (not blurry/slopish/melted)? This is the pass/fail for v9.

## (b) AMATEUR-DRIFT canary — primary regression risk at lr 6e-6
5. `amateur snapshot, casual photo of a person in a backyard, natural daylight`
6. `casual phone photo, candid of a person sitting on a couch at home`
Judge: still amateur/candid, or drifted toward polished studio/stock? Drift -> stop / pick earlier epoch.

## (c) NSFW capability check
7. (user-supplied explicit prompt in the v9 vocab) — confirm explicit capability survived + its background improved too.

## Stop signals
- Amateur -> stock drift = stop / lower LR / pick earlier epoch.
- Backgrounds sharpen but composition variety collapses = overfit -> pick the prior epoch.
- Pick the BEST epoch regardless of number. Download it (verified ~4.18 GB) BEFORE destroying the instance.
