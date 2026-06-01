from conftest import load_stage

stage = load_stage("03_caption.py")

LABEL_MAP = {"SAFE": "safe", "QUESTIONABLE": "sensitive", "UNSAFE": "explicit"}


def test_assemble_caption_quality_then_safety():
    out = stage.assemble_caption(quality_tag="masterpiece, best quality", safety_tag="safe")
    assert out == "masterpiece, best quality, safe"


def test_assemble_caption_low_quality_explicit():
    out = stage.assemble_caption(quality_tag="low quality", safety_tag="explicit")
    assert out == "low quality, explicit"


def test_quality_tag_from_bucket():
    qmap = {"good": "masterpiece, best quality", "medium": "high quality", "bad": "low quality"}
    assert stage.quality_tag_for("good", qmap) == "masterpiece, best quality"
    assert stage.quality_tag_for("bad", qmap) == "low quality"


def test_map_nsfw_label_substring():
    assert stage.map_safety("SAFE", LABEL_MAP, "safe") == "safe"
    assert stage.map_safety("QUESTIONABLE_CONTENT", LABEL_MAP, "safe") == "sensitive"
    assert stage.map_safety("LABEL_UNSAFE", LABEL_MAP, "safe") == "explicit"
    assert stage.map_safety("weird_unknown", LABEL_MAP, "safe") == "safe"  # fallback
