from conftest import load_stage

stage = load_stage("03_caption.py")

LABEL_MAP = {"SAFE": "safe", "QUESTIONABLE": "sensitive", "UNSAFE": "explicit"}


def test_assemble_caption_order_and_commas():
    out = stage.assemble_caption(
        quality_tag="masterpiece, best quality",
        safety_tag="safe",
        trigger="realistic photo",
        nl="a woman on a park bench at golden hour, 35mm",
    )
    assert out == "masterpiece, best quality, safe, realistic photo, a woman on a park bench at golden hour, 35mm"


def test_quality_tag_from_bucket():
    qmap = {"good": "masterpiece, best quality", "medium": "high quality", "bad": "low quality"}
    assert stage.quality_tag_for("good", qmap) == "masterpiece, best quality"
    assert stage.quality_tag_for("bad", qmap) == "low quality"


def test_map_nsfw_label_substring():
    assert stage.map_safety("SAFE", LABEL_MAP, "safe") == "safe"
    assert stage.map_safety("QUESTIONABLE_CONTENT", LABEL_MAP, "safe") == "sensitive"
    assert stage.map_safety("LABEL_UNSAFE", LABEL_MAP, "safe") == "explicit"
    assert stage.map_safety("weird_unknown", LABEL_MAP, "safe") == "safe"  # fallback


def test_nl_cleanup_strips_newlines_and_trailing_period():
    assert stage.clean_nl("  A photo of a cat.\nSecond line.  ") == "A photo of a cat. Second line"
