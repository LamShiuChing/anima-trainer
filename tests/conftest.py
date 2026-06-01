import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # so stages can `import common` and tests can `import common`


def load_stage(filename):
    """Import a numbered stage module (e.g. '01_ingest_clean.py') that cannot be imported normally."""
    path = SRC / filename
    name = filename[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def make_image(tmp_path):
    """Factory: write a synthetic JPEG/PNG and return its Path."""
    def _make(name="img.jpg", size=(800, 600), color=(120, 80, 40), noise=False):
        w, h = size
        if noise:
            arr = (np.random.rand(h, w, 3) * 255).astype("uint8")
            img = Image.fromarray(arr)
        else:
            img = Image.new("RGB", size, color)
        p = tmp_path / name
        img.save(p)
        return p
    return _make
