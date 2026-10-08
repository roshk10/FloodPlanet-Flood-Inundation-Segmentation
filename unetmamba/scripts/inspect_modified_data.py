import os
import sys
import json
from pathlib import Path
import tifffile
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
UNETMAMBA_DIR = SCRIPT_DIR.parent
PROJECT_DIR = UNETMAMBA_DIR.parent
sys.path.insert(0, str(UNETMAMBA_DIR))

import configs.unetmamba_config as cfg


def get_exact_crop_boxes(height=1024, width=1024, patch_size=512, stride=256):
    """Matches exact patching logic in src/dataset.py"""
    crop_boxes = []
    num_h = 0
    while num_h * stride + patch_size <= height:
        num_h += 1
    num_w = 0
    while num_w * stride + patch_size <= width:
        num_w += 1

    # Full patches
    for i in range(num_h):
        for j in range(num_w):
            crop_boxes.append((i * stride, j * stride, patch_size, patch_size))

    # Right edge
    rem_w = width - num_w * stride
    if rem_w != 0:
        for i in range(num_h):
            crop_boxes.append((i * stride, num_w * stride, patch_size, rem_w))

    # Bottom edge
    rem_h = height - num_h * stride
    if rem_h != 0:
        for j in range(num_w):
            crop_boxes.append((num_h * stride, j * stride, rem_h, patch_size))

    # Bottom-right corner
    if rem_h != 0 and rem_w != 0:
        crop_boxes.append((num_h * stride, num_w * stride, rem_h, rem_w))

    return crop_boxes


def inspect_modified():
    print("=" * 80)
    print("             MODIFIED / PREPROCESSED DATASET PIPELINE INSPECTION")
    print("=" * 80)

    data_root = cfg.DATASET_ROOT
    if not data_root.exists():
        print(f"ERROR: Dataset directory not found at {data_root}")
        return

    # 1. Load Metadata & Split
    split_file = UNETMAMBA_DIR / "configs" / "event_split.json"
    with open(split_file) as f:
        split = json.load(f)

    meta_csv = PROJECT_DIR / "data" / "metadata" / "planetscope_index.csv"

    print(f"\n[1] CONFIGURATION SPECIFICATION:")
    print(f"    Dataset Root:         {data_root}")
    print(f"    Event Split File:     {split_file.name} (Seed: {split.get('seed', 2026)})")
    print(f"    Metadata CSV Index:   {meta_csv.name} ({meta_csv.exists()})")
    print(f"    Input Channels:       {cfg.IN_CHANNELS} (RGB + NIR)")
    print(f"    Number of Classes:    {cfg.NUM_CLASSES} (0 = Background, 1 = Flood)")
    print(f"    Patch Size:           {cfg.PATCH_SIZE} x {cfg.PATCH_SIZE}")
    print(f"    Stride:               {cfg.STRIDE}")
    print(f"    Ignore Index:         {cfg.IGNORE_INDEX}")
    print(f"    Augmentations:        Flips (H+V), Continuous Rotation (0-360 deg)")

    # 2. Preprocessing Demonstration on Real Sample
    sample_file_ps = next((data_root / "Bangladesh" / "PS").glob("*.tif"))
    sample_file_lbl = next((data_root / "Bangladesh" / "labels").glob("*.tif"))

    raw_image = tifffile.imread(sample_file_ps)  # (1024, 1024, 4) uint16
    raw_label = tifffile.imread(sample_file_lbl) # (1024, 1024) uint8

    # Convert Channels to first: (4, 1024, 1024)
    if raw_image.shape[-1] == 4:
        image_ch_first = np.transpose(raw_image, (2, 0, 1))
    else:
        image_ch_first = raw_image

    # Apply Normalization
    norm_image = (image_ch_first.astype(np.float32) / 65536.0).clip(0.0, 1.0)

    # Apply Label Re-encoding
    binary_label = np.full(raw_label.shape, cfg.IGNORE_INDEX, dtype=np.int64)
    binary_label[raw_label == 1] = 0  # Background
    binary_label[raw_label == 2] = 1  # Flood

    # Extract 512x512 patch
    boxes = get_exact_crop_boxes(1024, 1024, cfg.PATCH_SIZE, cfg.STRIDE)
    sample_box = boxes[0]  # (0, 0, 512, 512)
    top, left, h, w = sample_box
    sample_patch_img = norm_image[:, top:top+h, left:left+w]
    sample_patch_lbl = binary_label[top:top+h, left:left+w]

    print("\n" + "=" * 80)
    print("                     DATA TRANSFORMATION COMPARISON")
    print("=" * 80)
    print(f"{'Attribute':<24} {'Raw / Unmodified Data':<28} {'Modified / Preprocessed Data'}")
    print("-" * 80)
    print(f"{'Image Array Shape':<24} {str(raw_image.shape):<28} {str(sample_patch_img.shape)} (C, H, W patch)")
    print(f"{'Image Data Type':<24} {str(raw_image.dtype):<28} {str(sample_patch_img.dtype)}")
    print(f"{'Image Value Range':<24} {f'[{raw_image.min()}, {raw_image.max()}]':<28} {f'[{sample_patch_img.min():.4f}, {sample_patch_img.max():.4f}] (Normalized 0..1)'}")
    print(f"{'Mask Array Shape':<24} {str(raw_label.shape):<28} {str(sample_patch_lbl.shape)} (H, W patch)")
    print(f"{'Mask Data Type':<24} {str(raw_label.dtype):<28} {str(sample_patch_lbl.dtype)}")
    print(f"{'Mask Encoding':<24} {str(sorted(list(np.unique(raw_label)))) + ' (1=dry, 2=flood)':<28} {str(sorted(list(np.unique(sample_patch_lbl)))) + ' (0=bg, 1=flood, -1=ignore)'}")
    print("-" * 80)

    # 3. Patching Analysis
    patches_per_chip = len(boxes)
    print(f"\n[2] SPATIAL PATCHING BREAKDOWN:")
    print(f"    Raw Chip Size:        1024 x 1024")
    print(f"    Crop Box Positions:   {patches_per_chip} patches per chip")
    for idx, (t, l, ph, pw) in enumerate(boxes[:4]):
        print(f"      Patch {idx+1:>2}: (top={t:>4}, left={l:>4}, height={ph:>3}, width={pw:>3})")
    print(f"      ... and {patches_per_chip - 4} more patches covering full chip overlap.")

    # 4. Event Split Inventory
    def count_split_chips(ev_list):
        count = 0
        for ev in ev_list:
            count += len(list((data_root / ev / "PS").glob("*.tif")))
        return count

    train_chips = count_split_chips(split["train"])
    val_chips = count_split_chips(split["validation"])
    test_chips = count_split_chips(split["test"])
    total_chips = train_chips + val_chips + test_chips

    print("\n" + "=" * 80)
    print("             DATASET SPLIT & TENSOR COUNTS FOR MODEL TRAINING")
    print("=" * 80)
    print(f"{'Split Name':<15} {'Events':<8} {'Chips (1024x1024)':<20} {'Patches (512x512)':<20} {'Isolation Status'}")
    print("-" * 80)
    print(f"{'TRAIN':<15} {len(split['train']):<8} {train_chips:<20} {train_chips * patches_per_chip:<20} Active Training")
    print(f"{'VALIDATION':<15} {len(split['validation']):<8} {val_chips:<20} {val_chips * patches_per_chip:<20} Model Selection / Tuning")
    print(f"{'TEST':<15} {len(split['test']):<8} {test_chips:<20} {test_chips * patches_per_chip:<20} STRICTLY UNSEEN")
    print("-" * 80)
    print(f"{'TOTAL':<15} {len(split['train'])+len(split['validation'])+len(split['test']):<8} {total_chips:<20} {total_chips * patches_per_chip:<20}")
    print("=" * 80)

    print("\n[3] EVENT ALLOCATION:")
    print(f"    TRAIN ({len(split['train'])} events):")
    print(f"      {', '.join(split['train'])}")
    print(f"    VALIDATION ({len(split['validation'])} events):")
    print(f"      {', '.join(split['validation'])}")
    print(f"    TEST ({len(split['test'])} events):")
    print(f"      {', '.join(split['test'])}")
    print("=" * 80)


if __name__ == "__main__":
    inspect_modified()
