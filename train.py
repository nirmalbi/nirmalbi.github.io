"""
train.py — Optimised training script for Age Classification (Phase I).

Usage
-----
1.  Extract the dataset so the structure looks like:

        dataset/
          train/
            0/   # Young
            1/   # Old
          valid/
          valid_labels.csv

2.  Run:
        python train.py                     # train on train/ only, evaluate on valid/
        python train.py --include_valid     # final submission: train on train+valid
        python train.py --epochs 30         # override number of epochs

The script saves the **full model** as  saved_model.pth  (compatible with
the provided evaluation script).
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from model_class import build_model

# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────
class TrainDataset(Dataset):
    """Load images from class sub-folders (train/0/, train/1/)."""

    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        for label in [0, 1]:
            cls_dir = os.path.join(root, str(label))
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class ValidDataset(Dataset):
    """Load labelled validation images from a flat directory + CSV."""

    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.samples.append((row["image"], int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────
IMG_SIZE = 224


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train(args):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = TrainDataset(
        os.path.join(args.data_root, "train"),
        transform=get_train_transform(),
    )

    valid_csv = os.path.join(args.data_root, "valid_labels.csv")
    valid_dir = os.path.join(args.data_root, "valid")

    if args.include_valid and os.path.isfile(valid_csv):
        print("Including validation data in training (final submission mode).")
        valid_train_ds = ValidDataset(valid_dir, valid_csv,
                                      transform=get_train_transform())
        combined_ds = ConcatDataset([train_ds, valid_train_ds])
    else:
        combined_ds = train_ds

    train_loader = DataLoader(
        combined_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation loader (always build for monitoring, even in final mode)
    val_loader = None
    if os.path.isfile(valid_csv) and not args.include_valid:
        val_ds = ValidDataset(valid_dir, valid_csv,
                              transform=get_eval_transform())
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    print(f"Training samples: {len(combined_ds)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_acc = correct / total
        avg_loss = running_loss / total
        lr_now = optimizer.param_groups[0]["lr"]

        msg = (f"Epoch {epoch:02d}/{args.epochs}  "
               f"Loss: {avg_loss:.4f}  "
               f"Train Acc: {train_acc:.4f}  "
               f"LR: {lr_now:.6f}")

        if val_loader:
            val_acc = evaluate(model, val_loader, device)
            msg += f"  Val Acc: {val_acc:.4f}"

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model, "best_model.pth")
                msg += " *"

        print(msg)

    # ── Save final model ──────────────────────────────────────────────
    torch.save(model, "saved_model.pth")
    size_mb = os.path.getsize("saved_model.pth") / 1e6
    print(f"\nSaved to saved_model.pth ({size_mb:.1f} MB)")

    if val_loader and best_val_acc > 0:
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print("Best checkpoint: best_model.pth")

    print("\nDone! For submission rename:")
    print("  model_class.py  ->  roll_no.py")
    print("  saved_model.pth ->  roll_no.pth")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet-18 for age classification")
    p.add_argument("--data_root", type=str, default="dataset",
                   help="Path to dataset root (containing train/, valid/)")
    p.add_argument("--epochs", type=int, default=30,
                   help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Max learning rate for OneCycleLR")
    p.add_argument("--weight_decay", type=float, default=1e-2,
                   help="Weight decay for AdamW")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing for CrossEntropyLoss")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers")
    p.add_argument("--include_valid", action="store_true",
                   help="Include validation data in training (for final submission)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
