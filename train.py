import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from vton.model import create_model
from vton.train_dataset import VTONDataset


def masked_l1(pred, target, mask):
    mask = mask.expand_as(pred)
    diff = (pred - target).abs() * mask
    return diff.sum() / (mask.sum() + 1e-6)


def run_epoch(model, loader, optimizer, scaler, device, train=True):
    model.train(train)
    total = 0.0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for batch in tqdm(loader, leave=False):
        person = batch["person"].to(device, non_blocking=True)
        garment = batch["garment"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        clothing = batch["clothing_mask"].to(device, non_blocking=True)
        face = batch["face_mask"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
            pred = model(person, garment, clothing, face)
            recon = F.smooth_l1_loss(pred, target)
            cloth_loss = masked_l1(pred, target, clothing)
            face_loss = masked_l1(pred, person, face)
            loss = recon + cloth_loss + 2.0 * face_loss

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        total += loss.item()

    return total / max(1, len(loader))


def load_training_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", -1) + 1, ckpt.get("best_val", float("inf"))


def main():
    p = argparse.ArgumentParser(description="Train the Vasundhara VTON model.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--output-dir", default="checkpoints/vasundhara-vton")
    p.add_argument("--resume", default="")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU. Use CPU only for code smoke tests.")

    device = torch.device("cuda")
    ds = VTONDataset(args.data_root, size=(args.width, args.height))
    if len(ds) < 2:
        raise RuntimeError("Need at least 2 paired samples for train/validation split")

    val_n = max(1, int(len(ds) * args.val_ratio))
    train_n = len(ds) - val_n
    if train_n < 1:
        raise RuntimeError("Validation split leaves no training samples")

    train_ds, val_ds = random_split(
        ds,
        [train_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = create_model(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    start = 0
    best = float("inf")
    if args.resume:
        start, best = load_training_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(start, args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, train=True)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, optimizer, scaler, device, train=False)

        scheduler.step()
        best = min(best, val_loss)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best,
        }
        torch.save(state, out / "last.pt")
        if val_loss <= best:
            torch.save(state, out / "best.pt")

        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train={train_loss:.5f} val={val_loss:.5f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
