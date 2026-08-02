"""
Step 4 (CLI) — upload a face + a prompt, get images.

    python scripts/04_generate.py --face faces/me.jpg \
        --prompt "standing on a rooftop at golden hour, cinematic" -n 4

For the point-and-click version instead:  python app.py
"""
import argparse
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import OUT_DIR, TRIGGER  # noqa: E402
from src.engine import (  # noqa: E402
    OutfitGenerator, DEFAULT_NEGATIVE, SIZE_PRESETS, find_default_lora,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", required=True, help="path to the face photo")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--lora", default=None, help="dir under models/lora/ (default: newest)")
    ap.add_argument("--lora-weight", type=float, default=0.85)
    ap.add_argument("--size", default="portrait 896x1152", choices=list(SIZE_PRESETS))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--identity", type=float, default=0.8,
                    help="IdentityNet / pose+face structure strength")
    ap.add_argument("--face-strength", type=float, default=0.8,
                    help="how hard to push facial likeness")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("-n", "--num", type=int, default=1)
    ap.add_argument("--no-offload", action="store_true",
                    help="only if you have >=12 GB VRAM")
    args = ap.parse_args()

    face_path = Path(args.face)
    if not face_path.exists():
        sys.exit(f"\n[X] face image not found: {face_path}\n")

    lora = args.lora or find_default_lora()
    if lora is None:
        print("[!] no trained LoRA found - generating without the outfit.")
    gen = OutfitGenerator(lora_path=lora, offload=not args.no_offload)

    face = Image.open(face_path)
    t0 = time.time()
    images, final_prompt = gen.generate(
        face_image=face,
        prompt=args.prompt,
        negative_prompt=args.negative,
        size=SIZE_PRESETS[args.size],
        steps=args.steps,
        guidance=args.guidance,
        identity_strength=args.identity,
        face_strength=args.face_strength,
        lora_weight=args.lora_weight,
        seed=None if args.seed < 0 else args.seed,
        num_images=args.num,
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, im in enumerate(images):
        p = OUT_DIR / f"{stamp}_{i:02d}.png"
        im.save(p)
        print(f"[ok] {p}")
    print(f"\nprompt used : {final_prompt}")
    print(f"trigger     : {TRIGGER}")
    print(f"time        : {time.time()-t0:.1f}s for {len(images)} image(s)")


if __name__ == "__main__":
    main()
