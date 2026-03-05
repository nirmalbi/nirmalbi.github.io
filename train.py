"""
Phase 1 — Age Classification Training Script
==============================================
Trains a ResNet-18 (from scratch) for binary age classification (Young=0, Old=1).

Modes
-----
  Quick run  (sanity check):   python train.py --mode quick
  Full run   (real training):  python train.py --mode full
  Final run  (train+valid):    python train.py --mode final

All results (plots, metrics CSV) are saved to ``results/phase1/``.
"""

import argparse
import csv
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from model_class import build_model

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────
class TrainDataset(Dataset):
    """Load images from class sub-folders (train/0/, train/1/)."""

    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        for label in [0, 1]:
            cls_dir = os.path.join(root, str(label))
            if not os.path.isdir(cls_dir):
                continue
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


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.long().to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────
# Plotting / logging helpers
# ──────────────────────────────────────────────
def save_metrics_csv(history, path):
    """Save training history to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr"])
        for row in history:
            writer.writerow(row)


def save_plots(history, out_dir):
    """Generate and save loss / accuracy plots."""
    epochs = [r[0] for r in history]
    train_loss = [r[1] for r in history]
    train_acc = [r[2] for r in history]
    val_acc = [r[3] for r in history]

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, "o-", label="Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, "o-", label="Train Acc")
    if any(v is not None for v in val_acc):
        ax.plot(epochs, [v if v is not None else float("nan") for v in val_acc],
                "s-", label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train / Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────
def train(args):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Mode-dependent hyper-parameters ──
    if args.mode == "quick":
        num_epochs = 2
        batch_size = 16
        lr = 1e-3
        include_valid = False
    elif args.mode == "final":
        num_epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        include_valid = True
    else:  # full
        num_epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        include_valid = False

    weight_decay = args.weight_decay
    label_smoothing = args.label_smoothing
    num_workers = args.num_workers

    # ── Datasets ──
    train_transform = get_train_transform()
    eval_transform = get_eval_transform()

    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")
    valid_csv = os.path.join(args.data_dir, "valid_labels.csv")

    train_dataset = TrainDataset(train_dir, transform=train_transform)
    if len(train_dataset) == 0:
        print(f"ERROR: No training images found in {train_dir}")
        print("       Make sure the dataset is extracted at: dataset/")
        return

    if include_valid and os.path.isfile(valid_csv):
        valid_train_ds = ValidDataset(valid_dir, valid_csv, transform=train_transform)
        combined = ConcatDataset([train_dataset, valid_train_ds])
        print(f"Training on train + valid: {len(combined)} images")
    else:
        combined = train_dataset
        print(f"Training on train only: {len(combined)} images")

    train_loader = DataLoader(combined, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)

    val_loader = None
    if not include_valid and os.path.isfile(valid_csv):
        val_dataset = ValidDataset(valid_dir, valid_csv, transform=eval_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        print(f"Validation: {len(val_dataset)} images")

    # ── Model ──
    model = build_model(num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimizer / Scheduler / Loss ──
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ── Training loop ──
    history = []
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_acc = correct / total
        avg_loss = running_loss / total
        lr_now = optimizer.param_groups[0]["lr"]

        val_acc = None
        msg = (f"Epoch {epoch:02d}/{num_epochs}  "
               f"Loss: {avg_loss:.4f}  Train Acc: {train_acc:.4f}  LR: {lr_now:.6f}")

        if val_loader:
            val_acc = evaluate(model, val_loader, device)
            msg += f"  Val Acc: {val_acc:.4f}"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model, os.path.join(out_dir, "best_model.pth"))
                msg += " *"

        history.append((epoch, avg_loss, train_acc, val_acc, lr_now))
        print(msg)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} min")
    if val_loader and best_val_acc > 0:
        print(f"Best validation accuracy: {best_val_acc:.4f}")

    # ── Save artefacts ──
    model_save_path = os.path.join(out_dir, "saved_model.pth")
    torch.save(model, model_save_path)
    print(f"Saved model to {model_save_path} "
          f"({os.path.getsize(model_save_path) / 1e6:.1f} MB)")

    save_metrics_csv(history, os.path.join(out_dir, "metrics.csv"))
    save_plots(history, out_dir)
    print(f"Metrics CSV and plots saved to {out_dir}/")

    # Save run config for reproducibility
    config = {
        "mode": args.mode,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "include_valid": include_valid,
        "seed": SEED,
        "device": str(device),
        "num_params": n_params,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nFor submission rename:")
    print("  model_class.py          -> roll_no.py")
    print(f"  {model_save_path}  -> roll_no.pth")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 — Age Classification Training")
    p.add_argument("--mode", choices=["quick", "full", "final"], default="full",
                   help="quick = 2-epoch sanity check; full = normal training; "
                        "final = retrain on train+valid")
    p.add_argument("--data_dir", default="dataset",
                   help="Root directory of the dataset")
    p.add_argument("--results_dir", default="results/phase1",
                   help="Where to save outputs")
    p.add_argument("--epochs", type=int, default=30,
                   help="Number of epochs (ignored in quick mode)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
