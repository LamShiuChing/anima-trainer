from conftest import load_stage

stage = load_stage("03_caption.py")

LABEL_MAP = {"SAFE": "safe", "QUESTIONABLE": "sensitive", "UNSAFE": "explicit"}


def test_map_nsfw_label_substring():
    assert stage.map_safety("SAFE", LABEL_MAP, "safe") == "safe"
    assert stage.map_safety("QUESTIONABLE_CONTENT", LABEL_MAP, "safe") == "sensitive"
    assert stage.map_safety("LABEL_UNSAFE", LABEL_MAP, "safe") == "explicit"
    assert stage.map_safety("weird_unknown", LABEL_MAP, "safe") == "safe"


def test_underage_hit_fires_on_wd14_comma_tags():
    block = {"child", "baby", "toddler"}
    assert stage.underage_hit("1girl, child, indoor", block) == {"child"}
    assert stage.underage_hit("woman, kitchen, mug", block) == set()


def test_underage_hit_needs_comma_tokens_not_nl_substring():
    # regression: this is WHY WD14 tags (not Gemini NL) must drive the block.
    block = {"child"}
    assert stage.underage_hit("a child sitting on a chair", block) == set()  # NL won't match -> would leak
    assert stage.underage_hit("child, chair", block) == {"child"}            # comma tags match
