"""Generates Anima_LoRA_RunMe.ipynb - a zero-knowledge guided runner for the pipeline.
Run:  .venv\\Scripts\\python.exe scripts\\make_runme_notebook.py
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- intro
md(r"""
# 🪄 Anima Realism LoRA — Run Me (step-by-step)

**What is this, in one breath?** You have an AI image model called **Anima** that natively draws *anime*. We are going to teach it to make **realistic photographs** instead. We do that by showing it a few hundred real photos, each paired with a short text description, and training a tiny add-on file called a **LoRA**. When you later load that LoRA into Anima and type `realistic photo`, the output shifts toward photorealism.

**You do not need to understand any of the code.** Just run the cells from top to bottom (click a cell, press **Shift+Enter**). Each step prints what it did and shows you pictures so you can see it working.

**How long?** Steps 1–5 = minutes (plus one-time model downloads). Step 7 (training) = a few hours on your RTX 4080.

---
### ▶️ How to run a cell
Click a grey **code** cell, then press **Shift + Enter**. It runs, prints output below it, and moves to the next cell. Run them **in order, top to bottom.** If a cell shows an error, fix what it says before moving on.
""")

# ---------------------------------------------------------------- what you provide
md(r"""
## 📥 What YOU must provide (read this first)

**One thing: your photos.** Put them in the folder:

```
D:\anima training\data\raw\
```

Guidance:
- **What kind:** realistic photographs — the *look* you want the model to learn. The more consistent the style, the stronger the result.
- **How many:** 500–3000 is the sweet spot for this first run. Fewer than ~300 may be too weak; you can always add more and re-run.
- **Format:** `.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp`. Any size — the pipeline handles resizing and throws out the junk (blurry, tiny, duplicates, memes/screenshots) automatically.
- **Folders are fine** — subfolders inside `data\raw\` are scanned too.
- **⚠️ Hard rule — legal adult content only.** Real, consenting adults. **No minors, nothing non-consensual.** This is non-negotiable.

You don't need to clean, rename, crop, or caption them. That's the pipeline's whole job. Just dump the photos in `data\raw\` and run the cells.

> **Don't have photos yet?** Add them now, then come back. Nothing below works on an empty folder.
""")

# ---------------------------------------------------------------- setup cell
md(r"""
## ⚙️ Step 0 — Setup (run this first, every time you open the notebook)
This loads a few small helpers used by the rest of the notebook. It writes nothing and changes nothing.
""")

code(r"""
import sys, subprocess, csv
from pathlib import Path
from collections import Counter

# Find the project root (the folder containing config/pipeline.yaml).
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "config" / "pipeline.yaml").exists():
    for parent in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        if (parent / "config" / "pipeline.yaml").exists():
            PROJECT_ROOT = parent
            break
print("Project root:", PROJECT_ROOT)

def run_cmd(args, cwd=None):
    "Run a command and stream its output live into the notebook. Raises if it fails."
    cwd = str(cwd or PROJECT_ROOT)
    printable = " ".join(str(a) for a in args)
    print(">", printable, "\n" + "-" * 60)
    proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    print("-" * 60 + f"\n[finished, exit code {proc.returncode}]")
    if proc.returncode != 0:
        raise SystemExit("That step FAILED (see the output above). Fix it before continuing.")

def run_stage(filename):
    "Run one pipeline stage script with this notebook's Python (the .venv)."
    run_cmd([sys.executable, str(PROJECT_ROOT / "src" / filename)])

def run_powershell(script_relpath):
    run_cmd(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(PROJECT_ROOT / script_relpath)])

def read_manifest():
    mp = PROJECT_ROOT / "data" / "manifest.csv"
    if not mp.exists():
        return []
    with open(mp, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def show_images(paths, n=6, size=256):
    "Display up to n images inline as thumbnails."
    from PIL import Image
    from IPython.display import display
    shown = 0
    for p in list(paths)[:n]:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((size, size))
            display(im)
            print(Path(p).name)
            shown += 1
        except Exception as e:
            print("could not show", p, "-", e)
    if shown == 0:
        print("(no images to show yet)")

def assert_venv():
    "Stop with a clear message if the notebook is running on the wrong Python (not the .venv)."
    exe = sys.executable.replace("/", "\\").lower()
    if ".venv" not in exe:
        raise SystemExit(
            "WRONG KERNEL.\n"
            f"  This notebook is running: {sys.executable}\n"
            "  It must run in the project's .venv.\n"
            "  FIX: top menu -> Kernel -> Change kernel -> 'Python (anima .venv)',\n"
            "       then Kernel -> Restart Kernel and Run All Cells."
        )

assert_venv()
print("Helpers loaded, running in the .venv. ✅")
""")

# ---------------------------------------------------------------- preflight
md(r"""
## 🩺 Step 0.5 — Health check
Confirms your GPU is visible, your photos are in place, and the right libraries are installed. **Read the green ✅ / red ❌ line at the bottom.**
""")

code(r"""
assert_venv()   # stops with a clear message if you're on the wrong kernel (before torch loads)
import torch, transformers
issues = []

print("Python     :", sys.version.split()[0])
print("Running in :", sys.executable)
in_venv = ".venv" in sys.executable.lower()
if not in_venv:
    issues.append("This notebook is NOT running in the project's .venv. Pick the .venv kernel (see the launch instructions).")

print("PyTorch    :", torch.__version__)
cuda = torch.cuda.is_available()
print("CUDA (GPU) :", cuda)
if cuda:
    print("GPU        :", torch.cuda.get_device_name(0))
else:
    issues.append("CUDA not available - the GPU isn't visible. Did you install the cu128 torch into .venv?")

print("transformers:", transformers.__version__)
if int(transformers.__version__.split(".")[0]) >= 5:
    issues.append('transformers 5.x found - run in a terminal:  pip install "transformers>=4.45,<5"  then restart the kernel.')

raw = PROJECT_ROOT / "data" / "raw"
exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
imgs = [p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in exts] if raw.exists() else []
print("Photos in data/raw:", len(imgs))
if len(imgs) == 0:
    issues.append("data/raw is EMPTY. Put your photos in  " + str(raw) + "  then re-run this cell.")

print("\n" + "=" * 60)
if issues:
    print("❌ Fix these before going further:")
    for i in issues:
        print("  -", i)
else:
    print("✅ All good - you're ready. Run the steps below in order.")
""")

# ---------------------------------------------------------------- stage 1
md(r"""
## 🧹 Step 1 — Clean the photos
**What happens:** the pipeline scans `data\raw\`, throws out images that are corrupt, too small, blurry, near-duplicate reposts, or are memes/screenshots (lots of text). The survivors are copied to `data\clean\` and recorded in a spreadsheet (`data\manifest.csv`) that the later steps build on.

**You provide:** nothing — just run it. First run downloads a small text-detection model (a few seconds).
""")
code(r"""
run_stage("01_ingest_clean.py")
""")
code(r"""
# See what got kept vs thrown out, and preview a few survivors.
m = read_manifest()
kept = [r for r in m if r.get("dropped") == "False"]
dropped = [r for r in m if r.get("dropped") == "True"]
print(f"Kept {len(kept)} photos, dropped {len(dropped)}.")
print("Why dropped:", dict(Counter(r.get("drop_reason", "") for r in dropped)) or "(none)")
print("\nA few of the photos we kept:")
show_images([r["path"] for r in kept], n=6)
""")

# ---------------------------------------------------------------- stage 2
md(r"""
## 🏅 Step 2 — Score photo quality
**What happens:** every kept photo gets an automatic "aesthetic" score and is sorted into **good / medium / bad** buckets. This is how mixed-quality photos become useful — instead of deleting bad ones, the model later learns the *quality* difference (so your nice prompts produce nice output).

**You provide:** nothing. First run downloads the CLIP vision model (~1.7 GB, one time).
""")
code(r"""
run_stage("02_quality_score.py")
""")
code(r"""
m = read_manifest()
scored = [r for r in m if r.get("bucket")]
print("Buckets:", dict(Counter(r["bucket"] for r in scored)))
for b in ("good", "bad"):
    ex = [r for r in scored if r["bucket"] == b][:2]
    if ex:
        print(f"\n{b.upper()} examples (score shown):")
        for r in ex:
            print("   score", r.get("aesthetic_score"), "-", Path(r["path"]).name)
        show_images([r["path"] for r in ex], n=2)
""")

# ---------------------------------------------------------------- stage 3
md(r"""
## 📝 Step 3 — Write captions
**What happens:** each photo gets a text description written by **JoyCaption** (an AI captioner), plus a quality tag and a safety tag. The final caption looks like:

```
masterpiece, best quality, safe, realistic photo, a woman on a park bench at golden hour, 35mm
```

`realistic photo` is the **trigger phrase** you'll type later to invoke the LoRA. The safety tag (`safe` / `sensitive` / `explicit`) is recorded, never used to delete anything.

**You provide:** nothing. ⚠️ **First run downloads the JoyCaption model (~6 GB)** and an NSFW classifier — be patient. Run the small check cell first.
""")
code(r"""
# One-time check: confirm the NSFW classifier's label names match our config.
from transformers import AutoModelForImageClassification
_m = AutoModelForImageClassification.from_pretrained("MichalMlodawski/nsfw-image-detection-large")
print("NSFW model labels:", _m.config.id2label)
print("\nExpected something like SAFE / QUESTIONABLE / UNSAFE.")
print("If the words are very different, open  config\\pipeline.yaml  and adjust the")
print("'caption: nsfw_label_map:' section so each label maps to safe / sensitive / explicit.")
del _m
""")
code(r"""
# This is the slow one (downloads ~6GB the first time, then captions every photo). Let it run.
run_stage("03_caption.py")
""")
code(r"""
# Look at a few photos next to the captions that were written for them.
m = read_manifest()
capd = [r for r in m if r.get("caption")][:4]
for r in capd:
    show_images([r["path"]], n=1)
    print("CAPTION:", r["caption"], "\n")
""")

# ---------------------------------------------------------------- stage 4
md(r"""
## 📦 Step 4 — Build the training set
**What happens:** the good + medium photos are copied into `data\dataset\`, each with a matching `.txt` caption file next to it, and a dataset config file is written. This is the exact format the trainer expects.

**You provide:** nothing.
""")
code(r"""
run_stage("04_build_dataset.py")
""")
code(r"""
ds = PROJECT_ROOT / "data" / "dataset"
imgs = [p for p in ds.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
txts = list(ds.glob("*.txt"))
print(f"Training set: {len(imgs)} images + {len(txts)} caption files in {ds}")
if txts:
    t = txts[0]
    print(f"\nExample caption file ({t.name}):\n  {t.read_text(encoding='utf-8')}")
""")

# ---------------------------------------------------------------- stage 5
md(r"""
## 🎛️ Step 5 — Write the training settings
**What happens:** writes the training config (`outputs\anima_realism_v1_training_config.toml`) tuned to fit your 16 GB GPU, plus a `sample_prompts.txt` of preview prompts the trainer renders each epoch so you can watch it learn.

**You provide:** nothing.
""")
code(r"""
run_stage("05_make_train_config.py")
""")
code(r"""
out = PROJECT_ROOT / "outputs"
print("=== training config ===")
print((out / "anima_realism_v1_training_config.toml").read_text(encoding="utf-8"))
print("=== preview prompts (rendered every epoch) ===")
print((out / "sample_prompts.txt").read_text(encoding="utf-8"))
""")

# ---------------------------------------------------------------- stage 6 downloads
md(r"""
## ⬇️ Step 6 — Download the Anima model files
**What happens:** downloads the 3 Anima model files (the image model, the text encoder, the VAE) into `models\`. **~5.6 GB, one time.** It skips any file already downloaded, so it's safe to re-run.

**You provide:** nothing. Needs internet.
""")
code(r"""
run_powershell("scripts/download_models.ps1")
""")

# ---------------------------------------------------------------- stage 7 train
md(r"""
## 🚀 Step 7 — Train the LoRA (the big one)
**What happens:** the first run clones the trainer and installs its own environment (**~10–15 min, one time**), then trains. **Training itself takes a few hours** on your 4080.

**What to watch:** as it runs, preview images appear in `outputs\` (and a `sample` subfolder). Compare the `masterpiece, best quality ... realistic photo` previews across epochs — they should drift from anime toward photoreal. That's the LoRA working.

**You provide:** nothing, just patience. This cell streams the trainer's output live. It's a long-running cell — that's normal. (Prefer a terminal? You can instead run `\.\scripts\06_train.ps1` in PowerShell and skip this cell.)

> 💥 **If it crashes with "out of memory":** open `config\pipeline.yaml`, under `train:` set `network_dim: 8`, `network_alpha: 8`, `resolution: 512`, then re-run **Step 5** and this step.
""")
code(r"""
# Long-running: clones + sets up the trainer (first time), then trains for hours.
run_powershell("scripts/06_train.ps1")
""")

# ---------------------------------------------------------------- done
md(r"""
## ✅ Done — where's my LoRA?

Your trained add-on files are in **`outputs\`**: look for `anima_realism_v1*.safetensors` (one per saved epoch, plus the final one). Preview images are alongside them.

**How to use it:** load `anima_realism_v1.safetensors` as a LoRA in your Anima image-generation app, and include **`realistic photo`** in your prompt (with quality tags like `masterpiece, best quality`) to invoke the photoreal shift.

**Want it stronger / different?**
- Add more / better photos to `data\raw\` and re-run from Step 1.
- Not photoreal enough → train more epochs (raise `max_train_epochs` in `config\pipeline.yaml`, re-run Step 5 + 7).
- Out of memory → the `dim 8 / res 512` fallback noted in Step 7.

**Re-running is safe.** Every step is idempotent — it reads the previous step's output and rebuilds cleanly. You can stop after any step and pick up later.

---
*Tip: everything this notebook does is just running the scripts in `src\` and `scripts\`. The notebook only adds the explanations and the picture previews.*
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (anima .venv)", "language": "python", "name": "anima-venv"},
    "language_info": {"name": "python"},
}

out_path = Path(__file__).resolve().parents[1] / "Anima_LoRA_RunMe.ipynb"
nbf.write(nb, str(out_path))
print("Wrote", out_path, "with", len(cells), "cells")
