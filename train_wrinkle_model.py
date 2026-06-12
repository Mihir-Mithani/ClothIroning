"""
Training Script — Wrinkle Detection CNN
Run this locally (GPU recommended) to train the WrinkleCNN on your dataset,
then upload the weights to HuggingFace Hub.

Dataset expected layout
-----------------------
data/
  train/
    images/   *.jpg / *.png
    masks/    *.png   (binary: 255 = wrinkle, 0 = smooth)
  val/
    images/
    masks/

Usage
-----
python train_wrinkle_model.py --data_dir data --epochs 50 --batch_size 8
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models.wrinkle_detector import WrinkleCNN


# ── Dataset ───────────────────────────────────────────────────────────────────
class WrinkleDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, img_size: int = 256):
        self.img_paths  = sorted(Path(img_dir).glob("*.jpg")) + \
                          sorted(Path(img_dir).glob("*.png"))
        self.mask_dir   = Path(mask_dir)
        self.img_size   = img_size
        self.transform  = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST
        )

        img_t  = self.transform(image)
        mask_t = torch.from_numpy(np.array(mask) / 255.0).float().unsqueeze(0)

        return img_t, mask_t


# ── Loss ──────────────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss  = self.bce(pred, target)
        pred_flat = pred.view(-1)
        tgt_flat  = target.view(-1)
        inter     = (pred_flat * tgt_flat).sum()
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / \
                          (pred_flat.sum() + tgt_flat.sum() + self.smooth)
        return 0.5 * bce_loss + 0.5 * dice_loss


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    train_ds = WrinkleDataset(
        f"{args.data_dir}/train/images",
        f"{args.data_dir}/train/masks",
        args.img_size,
    )
    val_ds = WrinkleDataset(
        f"{args.data_dir}/val/images",
        f"{args.data_dir}/val/masks",
        args.img_size,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model     = WrinkleCNN().to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, "wrinkle_cnn.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to: {args.save_dir}/wrinkle_cnn.pth")
    print("\nTo upload to HuggingFace Hub:")
    print("  huggingface-cli upload your-username/wrinkle-detector-weights "
          f"{args.save_dir}/wrinkle_cnn.pth wrinkle_cnn.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="data",       help="Root data directory")
    parser.add_argument("--save_dir",  default="checkpoints",help="Where to save weights")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--batch_size",type=int, default=8)
    parser.add_argument("--img_size",  type=int, default=256)
    parser.add_argument("--lr",        type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
