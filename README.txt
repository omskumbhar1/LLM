"""
Step 3 — train the outfit LoRA.

    python scripts/03_train_lora.py

Defaults are tuned for an 8 GB card. Only the UNet is on the GPU
(latents and text embeds were cached in step 2), LoRA weights are the
only trainable params, the optimizer is 8-bit, and gradient checkpointing
is on. Peak VRAM lands around 6.5-7 GB.

Expected wall time on an RTX 5050 laptop @ 768px:
    ~2500 steps  ->  roughly 2.5 - 3.5 hours

A checkpoint is saved every epoch to models/lora/epoch_XX/. Test a few.
Garment LoRAs usually peak at epoch 3-4 and start overcooking after that
(skin turns plasticky, poses get stiff) - that is normal, just use an
earlier epoch.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CACHE_DIR, LORA_DIR, BASE_MODEL  # noqa: E402

DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--repeats", type=int, default=10,
                   help="times each image is seen per epoch")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--snr-gamma", type=float, default=5.0,
                   help="Min-SNR loss weighting. 0 disables it.")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


class CachedData(torch.utils.data.Dataset):
    def __init__(self, repeats: int):
        meta_path = CACHE_DIR / "meta.json"
        if not meta_path.exists():
            sys.exit("\n[X] data/cache/meta.json missing. Run scripts/02_cache.py first.\n")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.index = [m for m in self.meta for _ in range(repeats)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        m = self.index[i]
        lat = torch.load(CACHE_DIR / f"{m['idx']:03d}.latent.pt", map_location="cpu")
        txt = torch.load(CACHE_DIR / f"{m['idx']:03d}.text.pt", map_location="cpu")
        # SDXL micro-conditioning: (orig_h, orig_w, crop_top, crop_left, tgt_h, tgt_w)
        time_ids = torch.tensor(
            [m["h"], m["w"], 0, 0, m["h"], m["w"]], dtype=torch.float32
        )
        return lat, txt["prompt_embeds"], txt["pooled"], time_ids


def snr_weights(scheduler, timesteps, gamma):
    ac = scheduler.alphas_cumprod.to(timesteps.device)
    a = ac[timesteps] ** 0.5
    s = (1 - ac[timesteps]) ** 0.5
    snr = (a / s) ** 2
    return (torch.stack([snr, gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from diffusers import DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

    print("[..] loading UNet (bf16)")
    unet = UNet2DConditionModel.from_pretrained(
        BASE_MODEL, subfolder="unet", torch_dtype=torch.bfloat16, variant="fp16"
    )
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg)
    unet.to(DEVICE)

    # LoRA params must be fp32 for a stable optimizer
    params = []
    for n, p in unet.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
            params.append(p)
    n_trainable = sum(p.numel() for p in params)
    print(f"[ok] trainable LoRA params: {n_trainable/1e6:.2f} M")

    if args.resume:
        sd = torch.load(args.resume, map_location="cpu")
        set_peft_model_state_dict(unet, sd)
        print(f"[ok] resumed from {args.resume}")

    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=1e-2, eps=1e-8)
        print("[ok] optimizer: AdamW8bit")
    except Exception as e:  # noqa: BLE001
        print(f"[!] bitsandbytes unavailable ({e}); falling back to AdamW fp32")
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    ds = CachedData(args.repeats)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    total_steps = math.ceil(len(ds) * args.epochs / args.grad_accum)
    print(f"[ok] {len(ds)//args.repeats} images x {args.repeats} repeats "
          f"x {args.epochs} epochs = {len(ds)*args.epochs} samples "
          f"-> {total_steps} optimizer steps")

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(total_steps, 2),
        pct_start=0.05, anneal_strategy="cos", div_factor=10, final_div_factor=10,
    )

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gstep = 0

    for epoch in range(1, args.epochs + 1):
        unet.train()
        running = 0.0
        bar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}")
        for i, (lat, emb, pooled, time_ids) in enumerate(bar):
            # noise schedule in fp32 - bf16 has only 8 mantissa bits and the
            # sqrt(alpha_bar) terms get badly quantised at high timesteps
            lat = lat.to(DEVICE, torch.float32)
            emb = emb.to(DEVICE, torch.bfloat16)
            pooled = pooled.to(DEVICE, torch.bfloat16)
            time_ids = time_ids.to(DEVICE, torch.bfloat16)

            noise = torch.randn_like(lat)
            t = torch.randint(0, scheduler.config.num_train_timesteps,
                              (lat.shape[0],), device=DEVICE).long()
            noisy = scheduler.add_noise(lat, noise, t).to(torch.bfloat16)

            pred = unet(
                noisy, t, emb,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
            ).sample

            if args.snr_gamma > 0:
                w = snr_weights(scheduler, t, args.snr_gamma)
                loss = F.mse_loss(pred.float(), noise.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, loss.ndim))) * w
                loss = loss.mean()
            else:
                loss = F.mse_loss(pred.float(), noise.float())

            (loss / args.grad_accum).backward()
            running += loss.item()

            # Gradient checkpointing on a frozen base can silently produce zero
            # grads on some torch/peft combos - you'd train for 3 hours and get
            # a LoRA full of noise. Catch it on the very first backward instead.
            if epoch == 1 and i == 0:
                gnorm = sum(p.grad.norm().item() ** 2
                            for p in params if p.grad is not None) ** 0.5
                if gnorm == 0.0:
                    sys.exit(
                        "\n[X] LoRA gradients are all zero after the first step.\n"
                        "    Gradient checkpointing isn't propagating grads on this\n"
                        "    torch/peft build. Fix: open scripts/03_train_lora.py and\n"
                        "    comment out the `unet.enable_gradient_checkpointing()`\n"
                        "    line, then lower TRAIN_PIXELS to 640*640 in src/config.py\n"
                        "    and re-run scripts/02_cache.py.\n"
                    )
                print(f"[ok] gradient check passed (norm {gnorm:.4f})")

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if gstep < total_steps - 1:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1

            if i % 10 == 0:
                vram = torch.cuda.max_memory_allocated() / 1024**3
                bar.set_postfix(loss=f"{running/(i+1):.4f}",
                                lr=f"{sched.get_last_lr()[0]:.2e}",
                                vram=f"{vram:.1f}G")

        out = LORA_DIR / f"epoch_{epoch:02d}"
        out.mkdir(parents=True, exist_ok=True)
        lora_sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionXLPipeline.save_lora_weights(
            str(out), unet_lora_layers=lora_sd, safe_serialization=True
        )
        torch.save(get_peft_model_state_dict(unet), out / "peft_state.pt")
        mins = (time.time() - t0) / 60
        print(f"[ok] saved {out}  |  avg loss {running/len(dl):.4f}  |  {mins:.1f} min elapsed")

    # convenience copy of the last epoch
    import shutil
    final = LORA_DIR / "final"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(LORA_DIR / f"epoch_{args.epochs:02d}", final)

    print(f"\n[ok] done in {(time.time()-t0)/60:.1f} min")
    print(f"[ok] peak VRAM {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print("\nNext: python app.py     (or scripts/04_generate.py)")
    print("Try epoch_03 and epoch_04 first, not just final.")


if __name__ == "__main__":
    main()
