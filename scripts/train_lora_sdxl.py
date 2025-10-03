import argparse, json, math, os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model


class CaptionDataset(Dataset):
    def __init__(self, root: Path, size: int = 1024):
        self.root = Path(root)
        self.size = size
        self.images = sorted((self.root / "images").glob("*.png"))
        meta = json.loads((self.root / "captions.json").read_text("utf-8"))
        self.captions = {k: v for k, v in meta.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        path = self.images[idx]
        caption = self.captions.get(path.name, "cozyStickerV1Style")
        image = Image.open(path).convert("RGB").resize((self.size, self.size), Image.LANCZOS)
        return image, caption


def collate(batch, pipe: StableDiffusionXLPipeline):
    imgs, caps = zip(*batch)
    pixel_values = pipe.image_processor.preprocess(list(imgs), do_normalize=True)
    tokens = pipe.tokenizer(
        list(caps),
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    tokens2 = pipe.tokenizer_2(
        list(caps),
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer_2.model_max_length,
        return_tensors="pt",
    )
    return pixel_values, tokens.input_ids, tokens2.input_ids


def main():
    ap = argparse.ArgumentParser(description="Minimal SDXL LoRA style trainer")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train_steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--image_size", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    unet: UNet2DConditionModel = pipe.unet
    lora_cfg = LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=["to_q", "to_v"], lora_dropout=0.0)
    unet_lora = get_peft_model(unet, lora_cfg)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder_2.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    ds = CaptionDataset(Path(args.data_root), size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=lambda b: collate(b, pipe))

    opt = torch.optim.AdamW(unet_lora.parameters(), lr=args.lr)
    total_steps = args.train_steps
    warmup = max(100, int(0.03 * total_steps))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)

    unet.train()
    global_step = 0
    pbar = tqdm(total=total_steps)
    while global_step < total_steps:
        for pixel_values, ids1, ids2 in dl:
            if global_step >= total_steps:
                break
            pixel_values = pixel_values.to(device)
            ids1 = ids1.to(device)
            ids2 = ids2.to(device)

            with torch.no_grad():
                latent_dist = pipe.vae.encode(pixel_values).latent_dist
                latents = latent_dist.sample() * pipe.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy = pipe.scheduler.add_noise(latents, noise, timesteps)
                emb1 = pipe.text_encoder(ids1)[0]
                emb2 = pipe.text_encoder_2(ids2)[0]

            model_pred = unet_lora(noisy, timesteps, encoder_hidden_states=emb1, added_cond_kwargs={"text_embeds": pipe.text_encoder_2.text_model.embeddings.position_embedding.weight[:1], "time_ids": pipe.add_time_ids})
            # Simplified MSE against noise
            loss = torch.nn.functional.mse_loss(model_pred.sample, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            global_step += 1
            pbar.set_description(f"loss={loss.item():.4f}")
            pbar.update(1)
            if global_step >= total_steps:
                break

    pbar.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save only the LoRA params
    unet_lora.save_pretrained(out.parent)
    # Emit a pointer file for convenience
    Path(args.output).write_text("saved with peft in same dir\n")
    print(f"[ok] LoRA trained to {out.parent}")


if __name__ == "__main__":
    main()

