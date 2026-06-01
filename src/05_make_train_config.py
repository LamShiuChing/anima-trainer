"""Stage 5: emit training TOML (sd-scripts/anima schema) + fixed-seed sample_prompts.txt."""
from pathlib import Path

import common

LOG = common.setup_logging()


def build_sample_prompts(trigger, seed):
    """Fixed-seed previews proving the trigger + quality/safety steering (success criteria #3,#4)."""
    common_tail = f"--w 768 --h 768 --seed {seed} --s 24 --l 5.0"
    return [
        f"masterpiece, best quality, safe, {trigger}, a woman sitting on a park bench at golden hour, 35mm {common_tail}",
        f"masterpiece, best quality, safe, {trigger}, a city street in the rain at night, neon reflections {common_tail}",
        f"low quality, safe, {trigger}, a woman sitting on a park bench at golden hour, 35mm {common_tail}",
        f"high quality, safe, {trigger}, portrait of an old man, natural window light {common_tail}",
    ]


def write_training_toml(out_path, dit, te, vae, output_dir, output_name, sample_prompts,
                        dim, alpha, lr, epochs, seed):
    fwd = lambda p: str(p).replace("\\", "/")
    toml = f"""pretrained_model_name_or_path = "{fwd(dit)}"
qwen3 = "{fwd(te)}"
vae = "{fwd(vae)}"
network_module = "networks.lora_anima"
network_dim = {dim}
network_alpha = {alpha}
network_train_unet_only = true
learning_rate = {lr}
optimizer_type = "AdamW8bit"
optimizer_args = ["weight_decay=0.1", "betas=[0.9, 0.99]"]
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 1
lr_warmup_steps = 100
max_train_epochs = {epochs}
train_batch_size = 1
gradient_accumulation_steps = 1
max_grad_norm = 1.0
seed = {seed}
timestep_sampling = "sigmoid"
discrete_flow_shift = 1.0
qwen3_max_token_length = 512
t5_max_token_length = 512
mixed_precision = "bf16"
gradient_checkpointing = true
cache_latents = true
cache_text_encoder_outputs = true
vae_chunk_size = 64
vae_disable_cache = true
output_dir = "{fwd(output_dir)}"
output_name = "{output_name}"
save_model_as = "safetensors"
save_precision = "bf16"
save_every_n_epochs = 1
save_last_n_epochs = 4
shuffle_caption = false
caption_extension = ".txt"
noise_offset = 0.03
multires_noise_discount = 0.3
sample_prompts = "{fwd(sample_prompts)}"
sample_every_n_epochs = 1
sample_at_first = true
sample_sampler = "euler_a"
"""
    Path(out_path).write_text(toml, encoding="utf-8")


def main():
    cfg = common.load_config()
    t = cfg["train"]
    m = cfg["models"]
    models_dir = Path(cfg["paths"]["models_dir"])
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)

    sample_path = out / "sample_prompts.txt"
    sample_path.write_text("\n".join(build_sample_prompts(cfg["caption"]["trigger"], t["seed"])) + "\n", encoding="utf-8")

    toml_path = out / f"{t['project_name']}_training_config.toml"
    write_training_toml(
        toml_path,
        dit=models_dir / Path(m["dit"]).name,
        te=models_dir / Path(m["te"]).name,
        vae=models_dir / Path(m["vae"]).name,
        output_dir=cfg["paths"]["outputs"],
        output_name=t["project_name"],
        sample_prompts=sample_path,
        dim=t["network_dim"], alpha=t["network_alpha"], lr=t["learning_rate"],
        epochs=t["max_train_epochs"], seed=t["seed"],
    )
    LOG.info("Stage 5 done. Training TOML -> %s ; samples -> %s", toml_path, sample_path)


if __name__ == "__main__":
    main()
