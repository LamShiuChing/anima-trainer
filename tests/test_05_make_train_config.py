try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 backport

from conftest import load_stage

stage = load_stage("05_make_train_config.py")


def test_anima_toml_valid_and_full_finetune(tmp_path):
    out = tmp_path / "anima.toml"
    stage.write_anima_toml(
        out,
        base_dir="/workspace/anima",
        project_name="anima_realism_ft_v1",
        dit_file="anima-base-v1.0.safetensors",
        vae_file="qwen_image_vae.safetensors",
        qwen_file="qwen_3_06b_base.safetensors",
        epochs=10, lr=8.0e-6, optimizer="AdamW8bitKahan", warmup_steps=100,
        save_every_n_epochs=1, checkpoint_every_n_minutes=30,
        llm_adapter_lr=0, tag_dropout_percent=0.10, caption_dropout_percent=0.05,
    )
    d = tomllib.loads(out.read_text(encoding="utf-8"))
    # full finetune => NO [adapter] block
    assert "adapter" not in d
    assert d["model"]["type"] == "anima"
    assert d["model"]["transformer_path"] == "/workspace/anima/models/anima-base-v1.0.safetensors"
    assert d["model"]["qwen_path"] == "/workspace/anima/models/qwen_3_06b_base.safetensors"
    assert d["model"]["vae_path"] == "/workspace/anima/models/qwen_image_vae.safetensors"
    assert d["model"]["llm_adapter_lr"] == 0          # freeze Qwen3 adapter
    assert d["model"]["caption_mode"] == "tags"        # tag-only
    assert d["model"]["cache_text_embeddings"] is False
    assert d["epochs"] == 10
    assert d["activation_checkpointing"] is True
    assert d["optimizer"]["type"] == "AdamW8bitKahan"
    assert d["optimizer"]["lr"] == 8.0e-6
    assert d["dataset"].endswith("anima_realism_ft_v1_dataset_config.toml")


def test_anima_toml_paths_use_forward_slashes(tmp_path):
    out = tmp_path / "anima.toml"
    stage.write_anima_toml(
        out, base_dir="/workspace/anima", project_name="p",
        dit_file="a.safetensors", vae_file="v.safetensors", qwen_file="q.safetensors",
        epochs=1, lr=8e-6, optimizer="AdamW8bitKahan", warmup_steps=1,
        save_every_n_epochs=1, checkpoint_every_n_minutes=30,
        llm_adapter_lr=0, tag_dropout_percent=0.1, caption_dropout_percent=0.05,
    )
    text = out.read_text(encoding="utf-8")
    assert "\\" not in text  # no backslashes in any path
