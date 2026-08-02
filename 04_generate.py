"""
Step 1 — resize your outfit photos and write captions.

    python scripts/01_prepare.py

What it does
------------
* Reads every image in data/raw/
* Resizes each to ~768x768 worth of pixels, KEEPING aspect ratio,
  snapping both sides to a multiple of 64. No cropping, nothing lost.
* Auto-captions with BLIP, strips out clothing words, appends the
  trigger token.
* Writes data/captions.txt  <-- REVIEW THIS FILE BEFORE TRAINING.

Why captions matter more than anything else here
------------------------------------------------
The LoRA learns whatever you DON'T describe. So we describe the person,
the pose, the background, the lighting - and we deliberately do NOT
describe the clothes. Everything about the garment then gets absorbed
into the trigger token.
"""
import argparse
import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DIR, PREP_DIR, CAPTION_FILE, TRIGGER, TRAIN_PIXELS  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}

# Words that describe clothing. If BLIP mentions them, we delete them -
# otherwise they'd compete with the trigger token for the garment.
CLOTHING = r"""
shirt|t-shirt|tshirt|tee|blouse|top|jacket|coat|blazer|hoodie|sweater|
jumper|cardigan|vest|waistcoat|kurta|kurti|saree|sari|lehenga|salwar|
sherwani|dhoti|dress|gown|frock|skirt|pants|trousers|jeans|shorts|
leggings|joggers|suit|uniform|outfit|clothing|clothes|apparel|garment|
attire|robe|tunic|polo|sweatshirt|windbreaker|parka|anorak|overalls|
dungarees|jersey|tracksuit|scarf|tie|necktie
"""
COLOURS = r"""
red|blue|green|yellow|orange|purple|pink|black|white|grey|gray|brown|
beige|cream|navy|maroon|teal|turquoise|olive|tan|khaki|golden|gold|
silver|striped|plaid|checkered|floral|printed|patterned|denim|leather|
cotton|silk|wool|linen|satin|velvet
"""
# one clothing item, e.g. "a striped cotton t-shirt".
# The (?:\w+\s+){0,2} slot catches adjectives that aren't in the COLOURS list
# ("matching", "oversized", "traditional") so they don't get left dangling.
_ITEM = (rf"(?:(?:a|an|the|his|her|their)\s+)?(?:(?:{COLOURS})\s+)*"
         rf"(?:\w+\s+){{0,2}}(?:{CLOTHING})s?")
# a whole phrase, e.g. "wearing a blue denim shirt and black jeans"
CLOTH_PHRASE_RE = re.compile(
    rf"\b(?:wearing|dressed\s+in|clad\s+in|in|with)\s+{_ITEM}"
    rf"(?:\s*(?:,|and|&)\s*{_ITEM})*",
    re.IGNORECASE | re.VERBOSE,
)
# any leftover bare item
CLOTH_ITEM_RE = re.compile(rf"\b{_ITEM}\b", re.IGNORECASE | re.VERBOSE)
# connectors left hanging once an item is deleted
DANGLE_RE = re.compile(
    r"(?:^|(?<=\s))(?:wearing|dressed\s+in|clad\s+in|in|with|and|&|a|an|the|of)"
    r"(?=\s*(?:,|\.|$|and\b|with\b|in\b))",
    re.IGNORECASE,
)


def snap(px: int) -> int:
    return max(512, int(round(px / 64)) * 64)


def resize_keep_ar(img: Image.Image, pixel_budget: int):
    """Scale so w*h ~= pixel_budget, both sides divisible by 64."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = math.sqrt(pixel_budget / (w * h))
    nw, nh = snap(w * scale), snap(h * scale)
    # keep it from becoming absurdly wide/tall
    ratio = nw / nh
    if ratio > 1.8:
        nw = snap(nh * 1.8)
    elif ratio < 1 / 1.8:
        nh = snap(nw * 1.8)
    return img.resize((nw, nh), Image.LANCZOS)


def _tidy(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(r"([,.])\s*(?:[,.]\s*)+", r"\1 ", text)
    return text.strip(" ,.")


def clean_caption(text: str) -> str:
    text = text.strip().rstrip(".")
    # BLIP's signature garbage prefixes
    text = re.sub(r"^(?:arafed|araffe|araffes|there is|there are)\s+", "",
                  text, flags=re.I)
    text = CLOTH_PHRASE_RE.sub(" ", text)
    text = CLOTH_ITEM_RE.sub(" ", text)
    text = _tidy(text)
    for _ in range(3):                      # dangling words can chain
        new = _tidy(DANGLE_RE.sub(" ", text))
        if new == text:
            break
        text = new
    text = re.sub(r"\b(?:and|with|in|of|a|an|the)\s*$", "", text, flags=re.I)
    # "two people wearing matching in a gym" -> drop the orphaned verb phrase
    text = re.sub(r"\b(?:wearing|dressed\s+in|clad\s+in)\s+(?:\w+\s+){0,2}?"
                  r"(?=(?:,|\.|$|in\b|on\b|at\b|near\b|inside\b|outside\b))",
                  " ", text, flags=re.I)
    return _tidy(text)


def build_captioner(device: str):
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    mid = "Salesforce/blip-image-captioning-large"
    print(f"[..] loading captioner {mid}")
    # fp16 matmuls are unimplemented on CPU - only use it when we're on GPU
    dt = torch.float16 if device == "cuda" else torch.float32
    proc = BlipProcessor.from_pretrained(mid)
    model = BlipForConditionalGeneration.from_pretrained(
        mid, torch_dtype=dt
    ).to(device).eval()

    @torch.no_grad()
    def caption(img: Image.Image) -> str:
        inputs = proc(img, return_tensors="pt").to(device, dt)
        out = model.generate(**inputs, max_new_tokens=40, num_beams=3)
        return proc.decode(out[0], skip_special_tokens=True)

    return caption, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-caption", action="store_true",
                    help="skip BLIP, write a generic caption you edit by hand")
    args = ap.parse_args()

    files = sorted(p for p in RAW_DIR.iterdir()
                   if p.suffix.lower() in EXTS and p.is_file())
    if not files:
        sys.exit(f"\n[X] No images found in {RAW_DIR}\n    Put your outfit photos there first.\n")
    print(f"[ok] found {len(files)} images in data/raw/")

    caption_fn = None
    model = None
    if not args.no_caption:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        caption_fn, model = build_captioner(dev)

    lines = []
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(tqdm(files, desc="preparing")):
        try:
            img = Image.open(src)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {src.name}: {e}")
            continue

        img = resize_keep_ar(img, TRAIN_PIXELS)
        out_name = f"{i:03d}.png"
        img.save(PREP_DIR / out_name)

        if caption_fn is not None:
            raw = caption_fn(img)
            body = clean_caption(raw)
        else:
            body = "a person standing"
        if not body:
            body = "a person"
        lines.append(f"{out_name}\t{body}, wearing {TRIGGER}")

    if model is not None:
        del model
        import torch
        torch.cuda.empty_cache()

    CAPTION_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[ok] {len(lines)} images -> data/prepared/")
    print(f"[ok] captions      -> {CAPTION_FILE}")
    print("\n" + "-" * 62)
    print("READ data/captions.txt NOW. It takes 10 minutes and it is the")
    print("single highest-leverage thing you will do in this whole project.")
    print("-" * 62)
    print("Rules:")
    print("  * DESCRIBE: the person, their pose, the background, lighting.")
    print("  * DO NOT DESCRIBE: the outfit. No colours, no fabric, no 'jacket'.")
    print(f"  * Every line must end with 'wearing {TRIGGER}'.")
    print("  * Format is:   filename <TAB> caption")
    print("\nExample of a good line:")
    print(f"  004.png\ta young man standing on a city street, daylight, "
          f"wearing {TRIGGER}")
    print("\nThen: python scripts/02_cache.py")


if __name__ == "__main__":
    main()
