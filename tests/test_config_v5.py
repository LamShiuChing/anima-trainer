# tests/test_config_v5.py
import common  # conftest puts src/ on sys.path


def test_v5_config_values():
    cfg = common.load_config()  # the real config/pipeline.yaml

    # curation: drop sub-1024 + raise dedup aggressiveness at ingest
    assert cfg["ingest"]["min_size"] == 1024
    assert cfg["ingest"]["drop_small"] is True
    assert cfg["ingest"]["phash_hamming_threshold"] == 8

    # CLIP aesthetic stage removed entirely
    assert "quality" not in cfg

    # Gemini captioner block
    g = cfg["caption"]["gemini"]
    assert g["model"] == "gemini-2.5-flash-lite"
    assert g["safety_block_none"] is True

    # dataset: 1024, blur backstop, no aesthetic bucket filter
    assert cfg["dataset"]["resolutions"] == [1024]
    assert cfg["dataset"]["min_resolution"] == 1024
    assert "min_blur_var" in cfg["dataset"]
    assert "buckets_to_keep" not in cfg["dataset"]

    # finetune: from base (empty init_from), v5 project, CFG dropout 0.10
    f = cfg["finetune"]
    assert f["project_name"] == "anima_realism_ft_v5"
    assert f["init_from"] == ""
    assert f["epochs"] == 20
    assert f["caption_dropout_percent"] == 0.10
