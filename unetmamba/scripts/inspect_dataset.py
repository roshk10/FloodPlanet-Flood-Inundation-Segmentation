import os
import sys
import glob
from pathlib import Path
import tifffile
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
UNETMAMBA_DIR = SCRIPT_DIR.parent
PROJECT_DIR = UNETMAMBA_DIR.parent
sys.path.insert(0, str(UNETMAMBA_DIR))

import configs.unetmamba_config as cfg


def inspect():
    print("=" * 75)
    print("FLOODPLANET DATASET & ENVIRONMENT INSPECTION")
    print("=" * 75)

    data_root = cfg.DATASET_ROOT
    print(f"\n[1] DATASET LOCATION:")
    print(f"    Path:   {data_root}")
    print(f"    Exists: {data_root.exists()}")

    if not data_root.exists():
        print("ERROR: Dataset directory not found!")
        return

    # [2] Event-level inventory
    events = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    print(f"\n[2] EVENT INVENTORY ({len(events)} events found):")
    print(f"    {'Event':<16} {'PS (Chips)':<12} {'Labels':<10} {'L8':<6} {'S1':<6} {'S2':<6} {'Status'}")
    print("    " + "-" * 65)

    total_ps = 0
    total_lbl = 0
    total_l8 = 0
    total_s1 = 0
    total_s2 = 0

    per_event_counts = {}

    for ev in events:
        ev_dir = data_root / ev
        ps_files = list((ev_dir / "PS").glob("*.tif"))
        lbl_files = list((ev_dir / "labels").glob("*.tif"))
        l8_files = list((ev_dir / "L8").glob("*.tif")) if (ev_dir / "L8").exists() else []
        s1_files = list((ev_dir / "S1").glob("*.tif")) if (ev_dir / "S1").exists() else []
        s2_files = list((ev_dir / "S2").glob("*.tif")) if (ev_dir / "S2").exists() else []

        n_ps = len(ps_files)
        n_lbl = len(lbl_files)
        total_ps += n_ps
        total_lbl += n_lbl
        total_l8 += len(l8_files)
        total_s1 += len(s1_files)
        total_s2 += len(s2_files)

        per_event_counts[ev] = n_ps
        status = "MATCH" if n_ps == n_lbl and n_ps > 0 else "MISMATCH"
        print(f"    {ev:<16} {n_ps:<12} {n_lbl:<10} {len(l8_files):<6} {len(s1_files):<6} {len(s2_files):<6} {status}")

    print("    " + "-" * 65)
    print(f"    {'TOTAL':<16} {total_ps:<12} {total_lbl:<10} {total_l8:<6} {total_s1:<6} {total_s2:<6}")

    # [3] Raw vs Modified Data Properties
    sample_ps_file = next((data_root / events[0] / "PS").glob("*.tif"))
    sample_lbl_file = next((data_root / events[0] / "labels").glob("*.tif"))

    raw_ps = tifffile.imread(sample_ps_file)
    raw_lbl = tifffile.imread(sample_lbl_file)

    # Check unique label values across several events
    all_raw_labels = set()
    for ev in events[:5]:
        for lf in (data_root / ev / "labels").glob("*.tif"):
            arr = tifffile.imread(lf)
            all_raw_labels.update(np.unique(arr).tolist())

    print(f"\n[3] UNMODIFIED (RAW) DATA PROPERTIES:")
    print(f"    PlanetScope sample:    {sample_ps_file.name}")
    print(f"      - Shape:             {raw_ps.shape} (Height, Width, Bands: RGB + NIR)")
    print(f"      - Data Type:         {raw_ps.dtype}")
    print(f"      - Value Range:       min={raw_ps.min()}, max={raw_ps.max()}")
    print(f"    Ground-Truth sample:   {sample_lbl_file.name}")
    print(f"      - Shape:             {raw_lbl.shape}")
    print(f"      - Data Type:         {raw_lbl.dtype}")
    print(f"      - Raw Unique Values: {sorted(list(all_raw_labels))}")
    print(f"        (Meaning: 1 = Background/Dry Land, 2 = Flood Inundation, 0 = NoData/Ignore)")

    # Simulate Preprocessed Output
    norm_ps = (raw_ps.astype(np.float32) / 65536.0).clip(0.0, 1.0)
    proc_lbl = np.full(raw_lbl.shape, cfg.IGNORE_INDEX, dtype=np.int64)
    proc_lbl[raw_lbl == 1] = 0  # Background
    proc_lbl[raw_lbl == 2] = 1  # Flood

    print(f"\n[4] MODIFIED / PREPROCESSED DATA (ON-THE-FLY):")
    print(f"    Normalized Imagery:")
    print(f"      - Scaling Formula:   image / 65536.0, clipped to [0.0, 1.0]")
    print(f"      - Output dtype:      {norm_ps.dtype}")
    print(f"      - Value Range:       min={norm_ps.min():.4f}, max={norm_ps.max():.4f}")
    print(f"    Re-encoded Binary Mask:")
    print(f"      - Mapping:           raw 1 -> 0 (Background)")
    print(f"                           raw 2 -> 1 (Flood)")
    print(f"                           other -> -1 (Ignore Index)")
    print(f"      - Output dtype:      {proc_lbl.dtype}")
    print(f"      - Processed Values:  {np.unique(proc_lbl).tolist()}")

    # [5] Spatial Patching Protocol
    # 1024x1024 chips with patch_size=512, stride=256 yield 16 patches
    patches_per_chip = 16
    print(f"\n[5] SPATIAL PATCHING SPECIFICATION:")
    print(f"    Full Chip Resolution: 1024 x 1024")
    print(f"    Training Patch Size:  {cfg.PATCH_SIZE} x {cfg.PATCH_SIZE}")
    print(f"    Sliding Stride:       {cfg.STRIDE}")
    print(f"    Patches per Chip:     {patches_per_chip} patches")

    # [6] Fixed Event-Level Split Breakdown
    train_chips = sum(per_event_counts[e] for e in cfg.TRAIN_EVENTS)
    val_chips = sum(per_event_counts[e] for e in cfg.VALIDATION_EVENTS)
    test_chips = sum(per_event_counts[e] for e in cfg.TEST_EVENTS)

    print(f"\n[6] FIXED EVENT-LEVEL SPLIT BREAKDOWN (Seed {cfg.SEED}):")
    print(f"    TRAIN ({len(cfg.TRAIN_EVENTS)} events):")
    print(f"      Events:  {', '.join(cfg.TRAIN_EVENTS)}")
    print(f"      Chips:   {train_chips} chips")
    print(f"      Patches: {train_chips * patches_per_chip} patches (512x512)")

    print(f"\n    VALIDATION ({len(cfg.VALIDATION_EVENTS)} events):")
    print(f"      Events:  {', '.join(cfg.VALIDATION_EVENTS)}")
    print(f"      Chips:   {val_chips} chips")
    print(f"      Patches: {val_chips * patches_per_chip} patches (512x512)")

    print(f"\n    TEST ({len(cfg.TEST_EVENTS)} events - STRICTLY UNSEEN):")
    print(f"      Events:  {', '.join(cfg.TEST_EVENTS)}")
    print(f"      Chips:   {test_chips} chips")
    print(f"      Patches: {test_chips * patches_per_chip} patches (512x512)")

    print(f"\n    TOTAL:")
    print(f"      Chips:   {train_chips + val_chips + test_chips} / 366 chips")
    print(f"      Patches: {(train_chips + val_chips + test_chips) * patches_per_chip} / 5,856 patches")
    print("=" * 75)


if __name__ == "__main__":
    inspect()
