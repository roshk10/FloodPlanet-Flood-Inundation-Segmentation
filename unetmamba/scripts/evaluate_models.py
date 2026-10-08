import os
import sys
import argparse
from pathlib import Path

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
from torch.utils.data import DataLoader

from src.dataset import FloodPlanetDataset, build_records
from src.models.UNetMamba import UNetMamba
from src.models.baseline_unet import UNet
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


def evaluate_model_on_test(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(images)

            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)

    return calculate_metrics(cat_preds, cat_targets, num_classes=2, ignore_index=-1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate UNetMamba & Baseline U-Net on Held-out Test Set")
    parser.add_argument("--unetmamba_ckpt", type=str, default=r"D:\Basemodel\FloodPlanet-Flood-Inundation-Segmentation\unetmamba\checkpoints\unetmamba\best.pth", help="UNetMamba checkpoint")
    parser.add_argument("--baseline_ckpt", type=str, default=r"D:\Basemodel\FloodPlanet-Flood-Inundation-Segmentation\unetmamba\checkpoints\baseline_unet\best.pth", help="Baseline U-Net checkpoint")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"HELD-OUT TEST EVALUATION (BANGLADESH, CAMBODIA, NEPAL)")
    print("=" * 70)

    data_root = cfg.DATASET_ROOT if cfg.DATASET_ROOT.exists() else Path(r"D:\Basemodel\Datasets")
    all_records, event_names = build_records(data_root, expected_records=366)
    test_recs = [r for r in all_records if r["event"] in cfg.TEST_EVENTS]

    test_dataset = FloodPlanetDataset(test_recs, dataset_root=data_root, patch_size=cfg.PATCH_SIZE, stride=cfg.STRIDE, ignore_index=cfg.IGNORE_INDEX, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)

    test_events = set(r['event'] for r in test_recs)
    print(f"Held-out Test Events ({len(test_events)}): {sorted(list(test_events))}")
    print(f"Total Test Patches (512x512): {len(test_dataset)}")

    # 1. UNetMamba Evaluation
    unetmamba_path = Path(args.unetmamba_ckpt)
    if unetmamba_path.exists():
        print(f"\nLoading UNetMamba from: {unetmamba_path}")
        model_mamba = UNetMamba(in_chans=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES).to(device)
        ckpt = torch.load(unetmamba_path, map_location=device)
        model_mamba.load_state_dict(ckpt["model_state"])
        mamba_res = evaluate_model_on_test(model_mamba, test_loader, device)
    else:
        print(f"\n[WARNING] UNetMamba checkpoint not found at: {unetmamba_path}")
        mamba_res = None

    # 2. Baseline U-Net Evaluation
    baseline_path = Path(args.baseline_ckpt)
    if baseline_path.exists():
        print(f"\nLoading Baseline U-Net from: {baseline_path}")
        model_unet = UNet(in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES).to(device)
        ckpt = torch.load(baseline_path, map_location=device)
        model_unet.load_state_dict(ckpt["model_state"])
        unet_res = evaluate_model_on_test(model_unet, test_loader, device)
    else:
        print(f"\n[WARNING] Baseline U-Net checkpoint not found at: {baseline_path}")
        unet_res = None

    # Summary Table
    print("\n" + "=" * 75)
    print("HELD-OUT TEST SET COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Model Architecture':<20} | {'mIoU':<8} | {'Flood IoU':<10} | {'Bkg IoU':<8} | {'F1-Score':<8} | {'Accuracy':<8}")
    print("-" * 75)
    if mamba_res:
        print(f"{'UNetMamba':<20} | {mamba_res['miou']:.4f}  | {mamba_res['flood_iou']:.4f}     | {mamba_res['background_iou']:.4f}   | {mamba_res['f1_score']:.4f}   | {mamba_res['accuracy']:.4f}")
    if unet_res:
        print(f"{'Baseline 2D U-Net':<20} | {unet_res['miou']:.4f}  | {unet_res['flood_iou']:.4f}     | {unet_res['background_iou']:.4f}   | {unet_res['f1_score']:.4f}   | {unet_res['accuracy']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
