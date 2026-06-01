try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 backport
from pathlib import Path

from conftest import load_stage

stage = load_stage("05_make_train_config.py")


def test_training_toml_valid_and_key_fields(tmp_path):
    out = tmp_path / "train.toml"
    stage.write_training_toml(
        out,
        dit="models/anima-base-v1.0.safetensors",
        te="models/qwen_3_06b_base.safetensors",
        vae="models/qwen_image_vae.safetensors",
        output_dir="outputs",
        output_name="anima_realism_v1",
        sample_prompts="outputs/sample_prompts.txt",
        dim=32, alpha=32, lr=1e-4, epochs=10, seed=42,
    )
    d = tomllib.loads(out.read_text(encoding="utf-8"))
    assert d["network_module"] == "networks.lora_anima"
    assert d["network_train_unet_only"] is True   # freeze Qwen3 TE
    assert d["cache_latents"] is True
    assert d["cache_text_encoder_outputs"] is True
    assert d["network_dim"] == 32
    assert d["learning_rate"] == 1e-4
    assert d["mixed_precision"] == "bf16"
    assert d["sample_every_n_epochs"] == 1


def test_sample_prompts_demonstrate_quality_and_safety_steering():
    prompts = stage.build_sample_prompts(trigger="realistic photo", seed=42)
    text = "\n".join(prompts)
    assert "realistic photo" in text
    assert "masterpiece, best quality" in text
    assert "low quality" in text       # success criterion #4: quality axis
    assert "--d 42" in text             # fixed seed for comparable previews (--d is sd-scripts' seed flag)
