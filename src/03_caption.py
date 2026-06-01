"""Stage 3: JoyCaption NL (4-bit) + NSFW safety tag + quality tag -> assembled caption. Augments manifest."""
import re

import torch
from PIL import Image

import common

LOG = common.setup_logging()


# ---- pure logic (unit-tested) ----

def quality_tag_for(bucket, quality_tag_map):
    return quality_tag_map[bucket]


def map_safety(model_label, label_map, default_tag):
    up = model_label.upper()
    for key in sorted(label_map, key=len, reverse=True):
        if key.upper() in up:
            return label_map[key]
    return default_tag


def clean_nl(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".").strip()


def assemble_caption(quality_tag, safety_tag, trigger, nl):
    return f"{quality_tag}, {safety_tag}, {trigger}, {nl}"


# ---- model wrappers (smoke-tested) ----

class JoyCaptioner:
    def __init__(self, cfg, device="cuda"):
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        name = cfg["caption"]["joycaption_model"]
        self.prompt = cfg["caption"]["joycaption_prompt"]
        self.max_new_tokens = cfg["caption"]["max_new_tokens"]
        self.device = device
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            name, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map=device,
        ).eval()

    @torch.no_grad()
    def caption(self, path):
        image = Image.open(path).convert("RGB")
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": self.prompt},
        ]
        convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)[0]
        ids = ids[inputs["input_ids"].shape[1]:]
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return clean_nl(text)


class NSFWTagger:
    def __init__(self, cfg, device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        name = cfg["caption"]["nsfw_model"]
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
        self.label_map = cfg["caption"]["nsfw_label_map"]
        self.default = cfg["caption"]["nsfw_default_tag"]

    @torch.no_grad()
    def tag(self, path):
        image = Image.open(path).convert("RGB")
        inputs = self.proc(images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        idx = int(logits.argmax(-1).item())
        model_label = self.model.config.id2label[idx]
        return map_safety(model_label, self.label_map, self.default)


def main():
    cfg = common.load_config()
    cap_cfg = cfg["caption"]
    rows = common.read_manifest(cfg["paths"]["manifest"])
    kept = [r for r in rows if r.get("dropped") == "False"]
    LOG.info("Stage 3: captioning %d images", len(kept))

    joy = JoyCaptioner(cfg)
    nsfw = NSFWTagger(cfg)
    updates = {}
    for r in kept:
        qtag = quality_tag_for(r["bucket"], cap_cfg["quality_tag_map"])
        stag = nsfw.tag(r["path"])
        nl = joy.caption(r["path"])
        caption = assemble_caption(qtag, stag, cap_cfg["trigger"], nl)
        updates[r["path"]] = {"safety_tag": stag, "quality_tag": qtag, "caption": caption}

    common.augment_manifest(cfg["paths"]["manifest"], updates)
    LOG.info("Stage 3 done.")


if __name__ == "__main__":
    main()
