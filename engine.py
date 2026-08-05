"""Central config + paths. Edit TRIGGER and BASE_MODEL here if you want."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"          # you drop your ~50 outfit photos here
PREP_DIR = ROOT / "data" / "prepared"    # resized images land here
CACHE_DIR = ROOT / "data" / "cache"      # precomputed latents + text embeds
CAPTION_FILE = ROOT / "data" / "captions.txt"
LORA_DIR = ROOT / "models" / "lora"
INSIGHTFACE_ROOT = ROOT / "models" / "insightface"
INSTANTID_DIR = ROOT / "models" / "instantid"
FACES_DIR = ROOT / "faces"               # you drop face photos here
OUT_DIR = ROOT / "outputs" 

for _d in (RAW_DIR, PREP_DIR, CACHE_DIR, LORA_DIR, INSIGHTFACE_ROOT,
           INSTANTID_DIR, FACES_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# The rare token your outfit gets bound to. Must be a nonsense word that
# SDXL has no prior for. Do not change it after training. 
# ---------------------------------------------------------------------
TRIGGER = "ohwxfit"

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"   # fp16-safe SDXL VAE

# Target pixel budget for training. 768*768 keeps you inside 8GB.
# Aspect ratio is preserved (batch size is 1, so mixed sizes are fine).
TRAIN_PIXELS = 768 * 768
