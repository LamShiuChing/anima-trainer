"""Download the RAM++ checkpoint to models/ram_plus_swin_large_14m.pth (run once).
Usage:  python scripts/v10_dl_ram.py
"""
import pathlib
import shutil

from huggingface_hub import hf_hub_download

dest = pathlib.Path("models/ram_plus_swin_large_14m.pth")
dest.parent.mkdir(exist_ok=True)
if dest.exists():
    print(f"already present: {dest} ({dest.stat().st_size} bytes)")
else:
    src = hf_hub_download("xinyu1205/recognize-anything-plus-model", "ram_plus_swin_large_14m.pth")
    shutil.copy(src, dest)
    print(f"downloaded -> {dest} ({dest.stat().st_size} bytes)")
