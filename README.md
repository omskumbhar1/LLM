"""
Inference engine: SDXL + InstantID (face from your upload) + outfit LoRA.

Built for an 8 GB card. The pipeline never fully sits on the GPU -
`enable_model_cpu_offload` streams each submodule in as it is needed.
Costs a few seconds per image, but it is the difference between running
and OOM-ing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    BASE_MODEL, VAE_MODEL, INSIGHTFACE_ROOT, INSTANTID_DIR, LORA_DIR, TRIGGER,
)

DTYPE = torch.float16

DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, extra fingers, deformed, distorted, "
    "disfigured, blurry, out of focus, jpeg artifacts, watermark, text, "
    "signature, cropped, worst quality, low quality, cartoon, anime, 3d render, "
    "cgi, painting, illustration, plastic skin, oversaturated"
)

SIZE_PRESETS = {
    "portrait 832x1216": (832, 1216),
    "portrait 896x1152": (896, 1152),
    "square 1024x1024": (1024, 1024),
    "landscape 1152x896": (1152, 896),
}


# ---------------------------------------------------------------------------
# face analysis
# ---------------------------------------------------------------------------
_face_app = None


def get_face_app():
    """insightface antelopev2, forced onto CPU so it doesn't fight for VRAM."""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name="antelopev2",
            root=str(INSIGHTFACE_ROOT),
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def analyse_face(pil_img: Image.Image):
    """Return (512-d identity embedding, 5x2 keypoints) for the largest face."""
    app = get_face_app()
    img = pil_img.convert("RGB")
    # detection is more reliable at a sane size
    if max(img.size) > 1280:
        s = 1280 / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        raise ValueError(
            "No face detected. Use a clear, front-facing photo where the face "
            "is at least ~200px wide and not heavily shadowed or angled."
        )
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    f = faces[0]
    return f.normed_embedding, np.array(f.kps), img.size


def kps_control_image(kps, src_size, dst_size):
    """Re-project the face keypoints onto a blank canvas of the target size."""
    from src.pipeline_instantid import draw_kps

    sw, sh = src_size
    dw, dh = dst_size
    scale = min(dw / sw, dh / sh)
    ox = (dw - sw * scale) / 2.0
    oy = (dh - sh * scale) / 2.0
    moved = kps.astype(np.float32) * scale + np.array([ox, oy], dtype=np.float32)
    return draw_kps(Image.new("RGB", (dw, dh), (0, 0, 0)), moved)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
class OutfitGenerator:
    def __init__(self, lora_path: str | None = None, base_model: str = BASE_MODEL,
                 offload: bool = True):
        from diffusers import ControlNetModel, AutoencoderKL
        from src.pipeline_instantid import StableDiffusionXLInstantIDPipeline

        cn_dir = INSTANTID_DIR / "ControlNetModel"
        adapter = INSTANTID_DIR / "ip-adapter.bin"
        if not adapter.exists():
            raise FileNotFoundError(
                f"{adapter} missing. Run:  python setup_models.py"
            )

        print("[..] loading IdentityNet controlnet")
        controlnet = ControlNetModel.from_pretrained(str(cn_dir), torch_dtype=DTYPE)

        print(f"[..] loading base model {base_model}")
        vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=DTYPE)
        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model, controlnet=controlnet, vae=vae,
            torch_dtype=DTYPE, variant="fp16", use_safetensors=True,
        )
        self.pipe.load_ip_adapter_instantid(str(adapter))

        self.lora_loaded = False
        if lora_path:
            self.load_lora(lora_path)

        self.pipe.set_progress_bar_config(disable=False)
        if offload:
            print("[..] enabling model cpu offload (8 GB mode)")
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.enable_vae_tiling()

        # the identity projector is not part of the offload graph - pin it
        if hasattr(self.pipe, "image_proj_model"):
            try:
                self.pipe.image_proj_model.to("cuda", dtype=DTYPE)
            except Exception:  # noqa: BLE001
                pass

    # -- lora ---------------------------------------------------------------
    def load_lora(self, path: str):
        p = Path(path)
        if p.is_dir():
            weights = p / "pytorch_lora_weights.safetensors"
        else:
            weights = p
        if not weights.exists():
            raise FileNotFoundError(f"LoRA not found: {weights}")
        try:
            self.pipe.unload_lora_weights()
        except Exception:  # noqa: BLE001
            pass
        self.pipe.load_lora_weights(str(weights.parent), weight_name=weights.name,
                                    adapter_name="outfit")
        self.lora_loaded = True
        print(f"[ok] LoRA loaded: {weights}")

    def _set_lora_scale(self, w: float):
        if not self.lora_loaded:
            return None
        try:
            self.pipe.set_adapters(["outfit"], adapter_weights=[float(w)])
            return None
        except Exception:  # noqa: BLE001
            return {"scale": float(w)}

    # -- generate -----------------------------------------------------------
    def generate(
        self,
        face_image: Image.Image,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE,
        size=(896, 1152),
        steps: int = 30,
        guidance: float = 5.0,
        identity_strength: float = 0.8,
        face_strength: float = 0.8,
        lora_weight: float = 0.85,
        seed: int | None = None,
        num_images: int = 1,
        add_trigger: bool = True,
    ):
        if add_trigger and TRIGGER not in prompt:
            prompt = f"{prompt.rstrip(', ')}, wearing {TRIGGER}"

        emb, kps, src_size = analyse_face(face_image)
        control = kps_control_image(kps, src_size, size)

        self.pipe.set_ip_adapter_scale(float(face_strength))
        xattn = self._set_lora_scale(lora_weight)

        gen = None
        if seed is not None and seed >= 0:
            gen = torch.Generator(device="cuda").manual_seed(int(seed))

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=emb,
            image=control,
            controlnet_conditioning_scale=float(identity_strength),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=size[0],
            height=size[1],
            num_images_per_prompt=int(num_images),
            generator=gen,
        )
        if xattn:
            kwargs["cross_attention_kwargs"] = xattn

        out = self.pipe(**kwargs)
        return out.images, prompt


def find_default_lora() -> str | None:
    final = LORA_DIR / "final" / "pytorch_lora_weights.safetensors"
    if final.exists():
        return str(final.parent)
    eps = sorted(LORA_DIR.glob("epoch_*/pytorch_lora_weights.safetensors"))
    return str(eps[-1].parent) if eps else None


def list_loras():
    out = []
    for d in sorted(LORA_DIR.iterdir()) if LORA_DIR.exists() else []:
        if (d / "pytorch_lora_weights.safetensors").exists():
            out.append(str(d))
    return out
