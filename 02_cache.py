# Outfit Generator — upload a face + a prompt, get that person in your outfit

Built for a single 8 GB laptop GPU (RTX 5050 / 4060 / 3070 class).

**How it works.** Two separate mechanisms, because they solve two different problems:

| problem | solved by | training needed |
|---|---|---|
| any face, supplied at generation time | **InstantID** (identity adapter) | none — zero-shot |
| your one specific outfit | **LoRA** trained on your 50 photos | ~3 hours, once |

A LoRA cannot do "upload any face" — LoRAs bake specific people in at training time. That is what the identity adapter is for. Your 50 photos are spent entirely on the garment.

---

## 0. Install

Python **3.10 or 3.11**. Not 3.12+ (insightface has no wheels there yet).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# torch FIRST, and it must be the cu128 build - your RTX 5050 is Blackwell (sm_120)
# and older torch builds have no kernels for it.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
python setup_models.py
```

`setup_models.py` verifies your GPU actually has kernels compiled for it, then pulls ~10 GB of weights. One time only.

> **If `pip install insightface` fails on Windows** it is trying to compile C++. Install *Microsoft C++ Build Tools* (Desktop development with C++), reopen the terminal, retry. Or `conda install -c conda-forge insightface`.

---

## 1. Your photos

Drop your ~50 outfit photos into `data/raw/`. Any size, any format.

What makes this dataset work: 40-50 **different people** in the **same garment**. The variety of faces, bodies, poses and backgrounds forces the LoRA to learn the clothing and nothing else. That is exactly the right shape — most people get this wrong by using one model in many outfits.

```bash
python scripts/01_prepare.py     # ~3 min
```

Resizes everything to ~768px worth of pixels with aspect ratio preserved (no cropping — batch size is 1 so mixed sizes are fine), then auto-captions with BLIP and writes `data/captions.txt`.

### Then open `data/captions.txt` and read it.

This is the highest-leverage 10 minutes in the whole project. The LoRA learns **whatever you don't describe.**

- **Describe:** the person, their pose, the background, the lighting.
- **Do NOT describe:** the outfit. No colours, no fabric, no "jacket", no "blue".
- Every line ends with `wearing ohwxfit`.
- Format: `filename` TAB `caption`.

```
004.png	a young man standing on a city street, daylight, wearing ohwxfit
011.png	a woman sitting on steps, indoor, soft light, wearing ohwxfit
```

The script already strips clothing words automatically, but BLIP misses things. If a caption still says "red shirt", that phrase competes with `ohwxfit` for the garment and your LoRA comes out weak.

---

## 2. Cache

```bash
python scripts/02_cache.py       # ~5 min
```

Encodes all images through the VAE and all captions through both text encoders, dumps them to disk, then frees those models. This is what buys the ~3 GB of headroom that makes SDXL training fit in 8 GB.

---

## 3. Train

```bash
python scripts/03_train_lora.py
```

Defaults: rank 32, alpha 16, LR 1e-4, 10 repeats, 5 epochs ≈ 2500 optimizer steps.

- **Peak VRAM:** ~6.5–7 GB
- **Wall time on RTX 5050 @ 768px:** roughly **2.5–3.5 hours**
- Saves a checkpoint every epoch to `models/lora/epoch_XX/`

**Test epoch 3 and 4, not just the last one.** Garment LoRAs typically peak around there and start overcooking after — skin goes plasticky, poses get stiff, backgrounds collapse. That is normal, just use an earlier epoch.

Useful flags:

```bash
--epochs 8 --repeats 6      # more, gentler passes
--rank 48                   # more capacity for detailed garments (prints, logos)
--lr 5e-5                   # if it overcooks fast
--snr-gamma 0               # disable Min-SNR loss weighting
--resume models/lora/epoch_03/peft_state.pt
```

If you OOM: lower `TRAIN_PIXELS` in `src/config.py` to `640*640`, re-run step 2, re-run step 3.

---

## 4. Generate

```bash
python app.py
```

→ `http://127.0.0.1:7860`. Upload a face, write a prompt, hit Generate.

Or CLI:

```bash
python scripts/04_generate.py \
  --face faces/someone.jpg \
  --prompt "standing on a rooftop at golden hour, half body, cinematic" \
  -n 4
```

**Write the prompt about the scene, not the clothes.** `wearing ohwxfit` is appended for you.

Speed: **~30–50 s per image** at 896×1152 with CPU offload on. First generation of a session adds ~60 s of model loading.

### Tuning

| symptom | fix |
|---|---|
| face doesn't look like the person | Face likeness → 1.0–1.2 |
| outfit wrong or missing | Outfit strength → 1.0–1.1, or a later epoch |
| stiff, flat, over-baked | IdentityNet → 0.6, Outfit strength → 0.7, or earlier epoch |
| garment right but body distorted | lower CFG to 4.0, raise steps to 40 |

---

## Swapping the base model

`stabilityai/stable-diffusion-xl-base-1.0` is the safe default and the thing your LoRA is trained against. For much better skin and lighting, edit `BASE_MODEL` in `src/config.py` to a photoreal SDXL checkpoint (`SG161222/RealVisXL_V5.0`, `RunDiffusion/Juggernaut-XL-v9`) **for inference only** — train on SDXL base, generate on the photoreal one. The LoRA transfers fine and the results are noticeably better.

---

## If the LoRA isn't enough

A LoRA gives you *"that style of garment"*, not a pixel-exact reproduction. Logos, prints and specific seam lines will drift. If you need the exact garment — say for a shop listing — switch to a two-stage pipeline: use this project to generate the person and scene, then run **CatVTON** or **IDM-VTON** as a second pass to composite the real garment onto the result. Both run in 8 GB. Slower, but the clothing is exact.

Start with the LoRA. For most cases it is enough.

---

## Layout

```
setup_models.py          env check + one-time downloads
app.py                   gradio UI
src/config.py            paths, TRIGGER token, base model, resolution
src/engine.py            InstantID + LoRA inference
scripts/01_prepare.py    resize + auto-caption
scripts/02_cache.py      precompute latents + text embeds
scripts/03_train_lora.py SDXL LoRA training (8 GB)
scripts/04_generate.py   CLI generation
ip_adapter/              downloaded by setup_models.py (not in the zip)
data/raw/                >>> your 50 outfit photos go here
faces/                   face photos to generate from
models/lora/             trained checkpoints
outputs/                 generated images
```

## Manual downloads (only if setup_models.py fails)

Everything below comes from `https://raw.githubusercontent.com/instantX-research/InstantID/main/`:

- `pipeline_stable_diffusion_xl_instantid.py` → save as `src/pipeline_instantid.py`
- `ip_adapter/__init__.py`, `ip_adapter/utils.py`, `ip_adapter/resampler.py`,
  `ip_adapter/attention_processor.py` → save into `ip_adapter/` at the project root.
  (These are not on PyPI — the pipeline imports them from the repo.)

And from HuggingFace:

- `InstantX/InstantID` → `ControlNetModel/` and `ip-adapter.bin` into `models/instantid/`
- antelopev2 `.onnx` files → `models/insightface/models/antelopev2/`
  (note the doubled `models/` — insightface expects exactly that layout)

## Where the weights come from

SDXL, the fp16 VAE and InstantID itself all download from their official
repos. The one exception is **antelopev2** (the face recognition model):
insightface distributes it as a Google Drive zip that can't be scripted, so
`setup_models.py` pulls it from community re-uploads on HuggingFace, same as
most InstantID and ComfyUI installs do. It's the normal path, but it is a
third party — if that matters for you, download antelopev2 from insightface
directly and drop the `.onnx` files into
`models/insightface/models/antelopev2/`.

The InstantID pipeline file is Python that gets imported and run. It's fetched
only from the authors' own GitHub repo and HF Space. Read
`src/pipeline_instantid.py` after setup if you want to see exactly what landed.

## One note on use

This takes arbitrary uploaded faces and puts them in generated photos. Keep it to faces you have permission to use — your own, or customers who have opted in. If this becomes a product feature, make the consent step explicit in the UI.
