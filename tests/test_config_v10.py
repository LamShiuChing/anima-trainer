# tests/test_config_v10.py
# Validates the real config/pipeline.yaml for the v10 run. Asserts STABLE invariants + the v10
# GEMINI-ONLY caption block. (finetune project/epochs/lr are NOT asserted here — they churn per run
# and live authoritatively in outputs/anima_realism_ft_v10_train_config.toml, not pipeline.yaml.)
import common  # conftest puts src/ on sys.path


def test_pipeline_config_v10():
    cfg = common.load_config()

    # ingest floor + aggressive dedup (stable since v5)
    assert cfg["ingest"]["min_size"] == 1024
    assert cfg["ingest"]["drop_small"] is True
    assert cfg["ingest"]["phash_hamming_threshold"] == 8

    # no separate CLIP/quality stage
    assert "quality" not in cfg

    # v10 captioning = GEMINI-ONLY structured (no RAM++/WD14/Falconsai/underage gate)
    cap = cfg["caption"]
    nl = cap["nl"]
    assert nl["model"] == "gemini-3-flash-preview"
    assert nl["block_none"] is True
    assert nl["max_output_tokens"] >= 600           # JSON: enums + tags + paragraph
    assert nl["cache_file"]
    # the v5/v7 caption machinery is gone
    for removed in ("gemini", "nsfw_model", "nsfw_label_map", "block_tags", "wd_model_name", "ram"):
        assert removed not in cap, f"caption.{removed} should be removed in v10"

    # dataset: train at 1536, v10 floor raised to 1280, AR pos-emb cap, char oversample wired
    ds = cfg["dataset"]
    assert ds["resolutions"] == [1536]
    assert ds["min_resolution"] == 1280
    assert (ds["min_ar"], ds["max_ar"]) == (0.66, 1.5)
    assert ds["char_num_repeats"] >= 1
    assert "buckets_to_keep" not in ds
