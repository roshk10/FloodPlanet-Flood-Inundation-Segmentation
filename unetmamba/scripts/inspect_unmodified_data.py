import os
import sys
from pathlib import Path
import tifffile
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
UNETMAMBA_DIR = SCRIPT_DIR.parent
PROJECT_DIR = UNETMAMBA_DIR.parent
sys.path.insert(0, str(UNETMAMBA_DIR))

import configs.unetmamba_config as cfg


def inspect_unmodified():
    print("=" * 80)
    print("                  UNMODIFIED (RAW) FLOODPLANET DATASET INSPECTION")
    print("=" * 80)

    data_root = cfg.DATASET_ROOT
    print(f"\n[RAW DATASET ROOT]: {data_root}")
    print(f"Status: {'EXISTS' if data_root.exists() else 'MISSING'}\n")

    if not data_root.exists():
        print("ERROR: Dataset directory not found!")
        return

    events = sorted([d.name for d in data_root.iterdir() if d.is_dir()])

    print(f"Total Flood Events: {len(events)}")
    print("-" * 80)
    print(f"{'Event Name':<16} {'PS (Chips)':<12} {'Labels':<10} {'L8':<8} {'S1':<8} {'S2':<8} {'Match Status'}")
    print("-" * 80)

    total_ps = 0
    total_lbl = 0
    total_l8 = 0
    total_s1 = 0
    total_s2 = 0
    total_bytes = 0

    per_event_details = {}

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

        # Calculate bytes
        ev_bytes = sum(f.stat().st_size for f in ps_files + lbl_files + l8_files + s1_files + s2_files)
        total_bytes += ev_bytes

        status = "OK (MATCH)" if n_ps == n_lbl and n_ps > 0 else "MISMATCH"
        print(f"{ev:<16} {n_ps:<12} {n_lbl:<10} {len(l8_files):<8} {len(s1_files):<8} {len(s2_files):<8} {status}")
        per_event_details[ev] = {
            "ps": n_ps, "lbl": n_lbl, "l8": len(l8_files), "s1": len(s1_files), "s2": len(s2_files), "bytes": ev_bytes
        }

    print("-" * 80)
    print(f"{'TOTALS':<16} {total_ps:<12} {total_lbl:<10} {total_l8:<8} {total_s1:<8} {total_s2:<8}")
    print(f"Total Dataset Disk Size: {total_bytes / (1024**3):.2f} GB\n")

    # Sample Inspection
    sample_ev = events[0]
    sample_ps_path = next((data_root / sample_ev / "PS").glob("*.tif"))
    sample_lbl_path = next((data_root / sample_ev / "labels").glob("*.tif"))

    raw_ps = tifffile.imread(sample_ps_path)
    raw_lbl = tifffile.imread(sample_lbl_path)

    # Check unique label values across all 19 events
    all_raw_labels = set()
    for ev in events:
        for lf in (data_root / ev / "labels").glob("*.tif"):
            arr = tifffile.imread(lf)
            all_raw_labels.update(np.unique(arr).tolist())

    print("=" * 80)
    print("                   RAW FILE STRUCTURE & METRICS")
    print("=" * 80)
    print("1. PLANETSCOPE IMAGERY (PS):")
    print(f"   - Sample File:          {sample_ps_path.name}")
    print(f"   - File Format:          GeoTIFF (.tif)")
    print(f"   - Raw Dimensions:       {raw_ps.shape[0]} x {raw_ps.shape[1]} (Height x Width)")
    print(f"   - Spectral Bands:       {raw_ps.shape[2]} channels (Band 1: Red, Band 2: Green, Band 3: Blue, Band 4: NIR)")
    print(f"   - Data Type (dtype):    {raw_ps.dtype} (unsigned 16-bit integer)")
    print(f"   - Raw Radiance Range:   min={raw_ps.min()}, max={raw_ps.max()}")
    print(f"   - Total Raw PS Chips:   {total_ps} chips globally distributed")

    print("\n2. GROUND-TRUTH FLOOD MASKS (labels):")
    print(f"   - Sample File:          {sample_lbl_path.name}")
    print(f"   - File Format:          GeoTIFF (.tif)")
    print(f"   - Raw Dimensions:       {raw_lbl.shape[0]} x {raw_lbl.shape[1]} (Single channel)")
    print(f"   - Data Type (dtype):    {raw_lbl.dtype} (unsigned 8-bit integer)")
    print(f"   - Raw Unique Values:    {sorted(list(all_raw_labels))}")
    print(f"   - Raw Value Meanings:")
    print(f"       Value 1: Background / Dry land")
    print(f"       Value 2: Flood Inundation")
    print(f"       Value 0: NoData / Ignore boundary")
    print(f"   - Total Raw Masks:      {total_lbl} masks matching 1:1 with PS chips")

    print("\n3. SUPPLEMENTARY MULTI-SENSOR MODALITIES (Untouched):")
    print(f"   - Landsat 8 (L8):       {total_l8} chips (30m spatial resolution)")
    print(f"   - Sentinel-1 (S1):      {total_s1} chips (SAR dual-pol VV/VH backscatter)")
    print(f"   - Sentinel-2 (S2):      {total_s2} chips (10m/20m multi-spectral)")
    print("=" * 80)


if __name__ == "__main__":
    inspect_unmodified()
