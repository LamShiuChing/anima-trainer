from conftest import load_stage

stage = load_stage("02_quality_score.py")


def test_score_to_bucket_thresholds():
    assert stage.score_to_bucket(7.2, good_min=6.0, medium_min=5.0) == "good"
    assert stage.score_to_bucket(6.0, good_min=6.0, medium_min=5.0) == "good"
    assert stage.score_to_bucket(5.4, good_min=6.0, medium_min=5.0) == "medium"
    assert stage.score_to_bucket(5.0, good_min=6.0, medium_min=5.0) == "medium"
    assert stage.score_to_bucket(3.1, good_min=6.0, medium_min=5.0) == "bad"


def test_mlp_shape():
    import torch
    mlp = stage.AestheticMLP(768)
    out = mlp(torch.zeros(1, 768))
    assert out.shape == (1, 1)
