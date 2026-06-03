# v6 frozen eval set — climb-vs-plateau

> Run these EXACT prompts + seeds + sampler settings on v5-epoch20 AND every v6 epoch
> (epoch1..5). Compare side-by-side. Plateau = no perceptible overall-realism gain across
> 2 consecutive saved epochs. Keep settings constant across ALL model states — constancy
> matters more than the specific values. Reused by v6b.

## Fixed sampler settings (match what v5 eval used; record actuals here once)
- sampler: euler   | steps: 25   | cfg: 4   | resolution: 1024x1024
- seeds (run every prompt at BOTH): 42, 1234
- negative prompt (constant): `watermark, anime, illustration, cartoon, drawing, 3d, cgi, render`

## Prompts (lead with the `realistic photo` anchor; span framings + vocab)
1. `realistic photo, masterpiece, best quality, studio portrait, studio lighting, sharp focus, safe, 1girl, close-up, face`
2. `realistic photo, high quality, semi-professional, soft window light, sharp focus, safe, 1girl, portrait, freckles`
3. `realistic photo, masterpiece, best quality, professional photograph, golden hour, sharp focus, safe, 1girl, upper body, outdoors, city street`
4. `realistic photo, high quality, professional photograph, natural daylight, sharp focus, safe, 1girl, full body, standing, park`
5. `realistic photo, low quality, amateur snapshot, direct on-camera flash, grainy / high ISO, safe, 1girl, indoors, bedroom`
6. `realistic photo, high quality, casual phone photo, indoor artificial light, sharp focus, safe, 1girl, sitting, cafe, holding phone`
7. `realistic photo, high quality, semi-professional, overcast flat light, sharp focus, safe, 1girl, full body, beach`
8. `realistic photo, high quality, professional photograph, low light, soft focus, safe, 1girl, upper body, nightclub`
9. `realistic photo, high quality, professional photograph, natural daylight, sharp focus, safe, 1boy, upper body, outdoors`
10. `realistic photo, masterpiece, best quality, professional photograph, studio lighting, sharp focus, safe, 1girl, sitting, barefoot, hands visible`

## Why these
- Prompts 1-2: close-up faces (v5 strength — confirm no regression).
- Prompts 3-4,7: medium/full-body (where small faces were weak — watch for overall-realism gain, not detail).
- Prompt 5: `amateur snapshot` (weakly trained control token per v5 blur-gate bias — watch).
- Prompt 6,10: hands / held object / feet (weak areas — for reference vs the inference track, NOT this run's goal).
- Prompt 8: low light + soft focus (lighting/condition vocab).
- Prompt 9: `1boy` (subject diversity).
