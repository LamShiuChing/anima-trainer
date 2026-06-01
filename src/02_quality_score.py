"""Stage 2: CLIP+MLP aesthetic score -> good/medium/bad bucket. Augments manifest."""
import torch
import torch.nn as nn
from PIL import Image

import common

LOG = common.setup_logging()


class AestheticMLP(nn.Module):
    """Architecture from christophschuhmann/improved-aesthetic-predictor (CLIP ViT-L/14, 768-dim)."""
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def score_to_bucket(score, good_min, medium_min):
    if score >= good_min:
        return "good"
    if score >= medium_min:
        return "medium"
    return "bad"


class AestheticScorer:
    def __init__(self, cfg, device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        name = cfg["quality"]["clip_model"]
        self.clip = CLIPModel.from_pretrained(name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.mlp = AestheticMLP(768).to(device).eval()
        weights = common.ensure_aesthetic_weights(cfg)
        # weights_only=True: .pth is a downloaded state_dict (tensors only); blocks arbitrary-code unpickling.
        self.mlp.load_state_dict(torch.load(weights, map_location=device, weights_only=True))

    @torch.no_grad()
    def score(self, path):
        img = Image.open(path).convert("RGB")
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        feats = self.clip.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)  # L2-normalize (predictor was trained on normalized CLIP feats)
        return float(self.mlp(feats).item())


def main():
    cfg = common.load_config()
    q = cfg["quality"]
    rows = common.read_manifest(cfg["paths"]["manifest"])
    kept = [r for r in rows if r.get("dropped") == "False"]
    LOG.info("Stage 2: scoring %d kept images", len(kept))

    scorer = AestheticScorer(cfg)
    updates = {}
    for r in kept:
        s = scorer.score(r["path"])
        updates[r["path"]] = {"aesthetic_score": f"{s:.3f}", "bucket": score_to_bucket(s, q["bucket_good_min"], q["bucket_medium_min"])}

    common.augment_manifest(cfg["paths"]["manifest"], updates)
    LOG.info("Stage 2 done.")


if __name__ == "__main__":
    main()
