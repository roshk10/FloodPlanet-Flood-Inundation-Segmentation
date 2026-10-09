import os
import sys
import time
import argparse
import csv
from pathlib import Path

# Add project root and unetmamba root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
UNETMAMBA_DIR = SCRIPT_DIR.parent
PROJECT_DIR = UNETMAMBA_DIR.parent

if "" in sys.path:
    sys.path.remove("")
if str(PROJECT_DIR) in sys.path:
    sys.path.remove(str(PROJECT_DIR))
sys.path.insert(0, str(UNETMAMBA_DIR / "src"))
sys.path.insert(0, str(UNETMAMBA_DIR))
sys.path.append(str(PROJECT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import FloodPlanetDataset, build_records
from src.models.baseline_unet import UNet
from src.losses.useful_loss import UnetMambaLoss
import configs.unetmamba_config as cfg


def calculate_metrics(preds, targets, num_classes=2, ignore_index=-1):
    mask = (targets != ignore_index)
    preds = preds[mask]
    targets = targets[mask]

    iou_list = []
    f1_list = []

    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()

        union = tp + fp + fn
        iou = tp / (union + 1e-7)
        iou_list.append(iou)

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        f1_list.append(f1)

    miou = sum(iou_list) / num_classes
    accuracy = (preds == targets).float().mean().item() if len(targets) > 0 else 0.0

    return {
        "miou": miou,
        "background_iou": iou_list[0],
        "flood_iou": iou_list[1] if num_classes > 1 else 0.0,
        "f1_score": sum(f1_list) / num_classes,
        "accuracy": accuracy,
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    val_loss /= len(loader.dataset)
    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)

    metrics = calculate_metrics(cat_preds, cat_targets, num_classes=2, ignore_index=-1)
    metrics["val_loss"] = val_loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Baseline U-Net on FloodPlanet")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--save_dir", type=str, default=r"D:\Basemodel\FloodPlanet-Flood-Inundation-Segmentation\unetmamba\checkpoints\baseline_unet", help="Checkpoint directory")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Baseline U-Net on {device} ({torch.cuda.get_device_name(0)}) ===")

    data_root = cfg.DATASET_ROOT if cfg.DATASET_ROOT.exists() else Path(r"D:\Basemodel\Datasets")
    all_records, event_names = build_records(data_root, expected_records=366)

    train_recs = [r for r in all_records if r["event"] in cfg.TRAIN_EVENTS]
    val_recs = [r for r in all_records if r["event"] in cfg.VALIDATION_EVENTS]

    train_dataset = FloodPlanetDataset(train_recs, dataset_root=data_root, patch_size=cfg.PATCH_SIZE, stride=cfg.STRIDE, ignore_index=cfg.IGNORE_INDEX, augment=True)
    val_dataset = FloodPlanetDataset(val_recs, dataset_root=data_root, patch_size=cfg.PATCH_SIZE, stride=cfg.STRIDE, ignore_index=cfg.IGNORE_INDEX, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    model = UNet(in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES).to(device)
    criterion = UnetMambaLoss(ignore_index=-1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "baseline_unet_metrics.csv"

    fieldnames = ["epoch", "train_loss", "val_loss", "miou", "flood_iou", "background_iou", "f1_score", "accuracy", "epoch_sec"]
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    best_miou = 0.0
    total_batches = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

            step_loss = loss.item()
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Step [{i:03d}/{total_batches:03d}] Loss: {step_loss:.4f}", flush=True)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)

        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_sec = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
              f"mIoU: {val_metrics['miou']:.4f} | Flood IoU: {val_metrics['flood_iou']:.4f} | F1: {val_metrics['f1_score']:.4f}")

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "miou": val_metrics["miou"],
            "flood_iou": val_metrics["flood_iou"],
            "background_iou": val_metrics["background_iou"],
            "f1_score": val_metrics["f1_score"],
            "accuracy": val_metrics["accuracy"],
            "epoch_sec": epoch_sec,
        }
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(log_row)

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, save_dir / "best.pth")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }, save_dir / "last.pth")

    print(f"\nBaseline U-Net Training Complete! Best Val mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
