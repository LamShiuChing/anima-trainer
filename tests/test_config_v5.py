# tests/test_config_v5.py
# Validates the real config/pipeline.yaml. Asserts STABLE invariants + the v7 caption block.
# (finetune project/epochs/lr are NOT asserted here — they churn per run and now live authoritatively
#  in the pre-staged outputs/anima_realism_ft_v*_train_config.toml, not pipeline.yaml.)
import common  # conftest puts src/ on sys.path


def test_pipeline_config_invariants_and_v7_caption():
    cfg = common.load_config()  # the real config/pipeline.yaml

    # curation: drop sub-1024 + aggressive dedup at ingest
    assert cfg["ingest"]["min_size"] == 1024
    assert cfg["ingest"]["drop_small"] is True
    assert cfg["ingest"]["phash_hamming_threshold"] == 8

    # CLIP aesthetic stage removed entirely (Gemini emits quality)
    assert "quality" not in cfg

    # Gemini captioner
    g = cfg["caption"]["gemini"]
    assert g["model"] == "gemini-2.5-flash-lite"
    assert g["safety_block_none"] is True
    assert g["max_output_tokens"] == 450             # v7: richer NL

    # v7 caption: bigger WD14 tagger + lower threshold + Falconsai rating fallback
    cap = cfg["caption"]
    assert cap["wd_model_name"] == "EVA02_Large"
    assert cap["wd_general_threshold"] == 0.25
    assert cap["nsfw_label_map"]["NORMAL"] == "rating:general"
    assert cap["nsfw_label_map"]["NSFW"] == "rating:explicit"
    # underage hard-block kept (legal boundary, non-negotiable)
    assert {"loli", "shota", "child"}.issubset(set(cap["block_tags"]))

    # dataset: current training res (1024; V7 training will bump to 1536 in its own spec)
    assert cfg["dataset"]["resolutions"] == [1024]
    assert "min_blur_var" in cfg["dataset"]
    assert "buckets_to_keep" not in cfg["dataset"]
