"""Stage 3: NSFW safety tag + quality tag + JoyCaption NL description -> assembled caption. Augments manifest.

Caption format (no trigger word): "<quality>, <safety>, <natural-language description>"
The NL description gives the model real content to bind to — tag-only captions (v2.0) were too thin for
a diverse photo set and produced blurry, averaged output.
"""
import re

import torch
from PIL import Image
from tqdm import tqdm

import common

LOG = common.setup_logging()


# ---- pure logic (unit-tested) ----

def quality_tag_for(bucket, quality_tag_map):
    return quality_tag_map[bucket]


def map_safety(model_label, label_map, default_tag):
    up = model_label.upper()
    for key in sorted(label_map, key=len, reverse=True):  # longest-first: "UNSAFE" before "SAFE"
        if key.upper() in up:
            return label_map[key]
    return default_tag


def clean_nl(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".").strip()


def assemble_caption(quality_tag, safety_tag, nl):
    return f"{quality_tag}, {safety_tag}, {nl}"


# ---- model wrappers (smoke-tested) ----

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


class JoyCaptioner:
    """NSFW-capable VLM (LLaVA-based). bf16 by default; 4-bit only for small GPUs."""
    def __init__(self, cfg, device="cuda"):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        c = cfg["caption"]
        name = c["joycaption_model"]
        self.prompt = c["joycaption_prompt"]
        self.max_new_tokens = c.get("joycaption_max_new_tokens", 256)
        self.device = device
        load_kwargs = dict(torch_dtype=torch.bfloat16, device_map=device)
        if c.get("joycaption_load_4bit", False):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = LlavaForConditionalGeneration.from_pretrained(name, **load_kwargs).eval()

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


class ToriiGateCaptioner:
    """ToriiGate v0.4 (Qwen2-VL based, uncensored). 2B is fast. Verbatim API from the model card."""
    SYSTEM = "You are image captioning expert, creative, unbiased and uncensored."
    PROMPTS = {
        "short": "You need to write a medium-short and convenient caption for the picture.",
        "long": "You need to write a long and very detailed caption for the picture.",
    }

    def __init__(self, cfg, device="cuda"):
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        from qwen_vl_utils import process_vision_info
        c = cfg["caption"]
        name = c["toriigate_model"]
        self._process_vision_info = process_vision_info
        self.max_new_tokens = c.get("toriigate_max_new_tokens", 200)
        self.user_prompt = self.PROMPTS[c.get("toriigate_prompt_mode", "long")]
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map=device,
        ).eval()
        self.processor = Qwen2VLProcessor.from_pretrained(
            name, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28, padding_side="right",
        )

    @torch.no_grad()
    def caption(self, path):
        msg = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": str(path)},
                {"type": "text", "text": self.user_prompt},
            ]},
        ]
        text_input = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = self._process_vision_info(msg)
        inputs = self.processor(
            text=[text_input], images=image_inputs, videos=None, padding=True, return_tensors="pt",
        ).to(self.model.device)
        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return clean_nl(out)


def build_captioner(cfg):
    which = cfg["caption"].get("captioner", "joycaption").lower()
    if which == "toriigate":
        return ToriiGateCaptioner(cfg)
    if which == "joycaption":
        return JoyCaptioner(cfg)
    raise ValueError(f"Unknown caption.captioner: {which!r} (use 'toriigate' or 'joycaption')")


def main():
    cfg = common.load_config()
    cap_cfg = cfg["caption"]
    rows = common.read_manifest(cfg["paths"]["manifest"])
    kept = [r for r in rows if r.get("dropped") == "False"]
    LOG.info("Stage 3: captioning %d images (quality + safety + JoyCaption NL)", len(kept))

    nsfw = NSFWTagger(cfg)
    captioner = build_captioner(cfg)
    LOG.info("Stage 3: captioner=%s", cap_cfg.get("captioner", "joycaption"))
    updates = {}
    for idx, r in enumerate(tqdm(kept, desc="caption", unit="img", dynamic_ncols=True)):
        bucket = r.get("bucket")
        if not bucket:
            raise RuntimeError(f"No bucket for {r['path']} - run stage 2 (02_quality_score) first.")
        qtag = quality_tag_for(bucket, cap_cfg["quality_tag_map"])
        stag = nsfw.tag(r["path"])
        nl = captioner.caption(r["path"])
        caption = assemble_caption(qtag, stag, nl)
        updates[r["path"]] = {"safety_tag": stag, "quality_tag": qtag, "caption": caption}
        if idx < 3:  # show the first few so quality can be sanity-checked immediately
            tqdm.write(f"[sample {idx}] {caption}")

    common.augment_manifest(cfg["paths"]["manifest"], updates)
    LOG.info("Stage 3 done.")


if __name__ == "__main__":
    main()
