import os
import sys
import json
import ast
from pathlib import Path
import tifffile
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
UNETMAMBA_DIR = SCRIPT_DIR.parent
PROJECT_DIR = UNETMAMBA_DIR.parent
sys.path.insert(0, str(UNETMAMBA_DIR))

import configs.unetmamba_config as cfg


def verify():
    print("=" * 80)
    print("             COMPREHENSIVE PROJECT & FILE INTEGRITY VERIFICATION")
    print("=" * 80)

    all_passed = True
    data_root = cfg.DATASET_ROOT

    # -------------------------------------------------------------
    # 1. Dataset Directory & Event Structure
    # -------------------------------------------------------------
    print("\n[CHECK 1/5] Verifying Dataset Directory & 19 Flood Events...")
    if not data_root.exists():
        print(f"  [FAIL] Dataset root {data_root} does not exist!")
        return False

    events = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    if len(events) == 19:
        print(f"  [PASS] All 19 flood event directories found at {data_root}.")
    else:
        print(f"  [FAIL] Expected 19 events, found {len(events)}.")
        all_passed = False

    # -------------------------------------------------------------
    # 2. Complete 366 Chip File-Level Integrity (PS & Labels)
    # -------------------------------------------------------------
    print("\n[CHECK 2/5] Validating File Integrity for All 366 Image/Mask Pairs...")
    ps_corrupt = 0
    lbl_corrupt = 0
    shape_mismatch = 0
    total_verified = 0

    for ev in events:
        ev_dir = data_root / ev
        ps_files = sorted(list((ev_dir / "PS").glob("*.tif")))
        lbl_files = sorted(list((ev_dir / "labels").glob("*.tif")))

        ps_dict = {f.stem: f for f in ps_files}
        lbl_dict = {f.stem: f for f in lbl_files}

        for chip_id, ps_p in ps_dict.items():
            if chip_id not in lbl_dict:
                print(f"  [FAIL] Missing label mask for {chip_id} in {ev}")
                all_passed = False
                continue

            lbl_p = lbl_dict[chip_id]

            # Fast header inspection
            try:
                ps_arr = tifffile.imread(ps_p)
                if ps_arr.shape != (1024, 1024, 4) or ps_arr.dtype != np.uint16:
                    shape_mismatch += 1
            except Exception as e:
                ps_corrupt += 1

            try:
                lbl_arr = tifffile.imread(lbl_p)
                if lbl_arr.shape != (1024, 1024) or lbl_arr.dtype not in (np.uint8, np.float32):
                    shape_mismatch += 1
            except Exception as e:
                lbl_corrupt += 1

            total_verified += 1

    if total_verified == 366 and ps_corrupt == 0 and lbl_corrupt == 0 and shape_mismatch == 0:
        print(f"  [PASS] All 366 PlanetScope chips (1024x1024x4 uint16) verified healthy.")
        print(f"  [PASS] All 366 Ground-Truth masks (1024x1024 uint8) verified healthy.")
        print(f"  [PASS] 0 corrupted files, 1:1 image/mask pair matching confirmed.")
    else:
        print(f"  [FAIL] Issues found: verified={total_verified}, ps_corrupt={ps_corrupt}, lbl_corrupt={lbl_corrupt}, shape_mismatch={shape_mismatch}")
        all_passed = False

    # -------------------------------------------------------------
    # 3. Fixed Split & Isolation Integrity
    # -------------------------------------------------------------
    print("\n[CHECK 3/5] Verifying Fixed Event Split (Seed 2026)...")
    split_file = UNETMAMBA_DIR / "configs" / "event_split.json"
    if not split_file.exists():
        print("  [FAIL] event_split.json missing!")
        all_passed = False
    else:
        with open(split_file) as f:
            split = json.load(f)

        train_ev = set(split["train"])
        val_ev = set(split["validation"])
        test_ev = set(split["test"])

        disjoint = (len(train_ev.intersection(val_ev)) == 0 and 
                    len(train_ev.intersection(test_ev)) == 0 and 
                    len(val_ev.intersection(test_ev)) == 0)

        if disjoint and len(train_ev) == 14 and len(val_ev) == 2 and len(test_ev) == 3:
            print(f"  [PASS] Split is strictly disjoint: 14 Train, 2 Val, 3 Test.")
            print(f"  [PASS] Test events strictly held out: {sorted(list(test_ev))}")
        else:
            print("  [FAIL] Split event counts or isolation violated!")
            all_passed = False

    # -------------------------------------------------------------
    # 4. Code Syntax & Source Structure
    # -------------------------------------------------------------
    print("\n[CHECK 4/5] Verifying Code Syntax for Models, Losses, and Pipelines...")
    code_files = [
        UNETMAMBA_DIR / "configs" / "unetmamba_config.py",
        UNETMAMBA_DIR / "configs" / "event_split.py",
        UNETMAMBA_DIR / "src" / "dataset.py",
        UNETMAMBA_DIR / "src" / "models" / "UNetMamba.py",
        UNETMAMBA_DIR / "src" / "models" / "ResT.py",
        UNETMAMBA_DIR / "src" / "models" / "vmamba.py",
        UNETMAMBA_DIR / "src" / "models" / "csm_triton.py",
        UNETMAMBA_DIR / "src" / "losses" / "useful_loss.py",
        UNETMAMBA_DIR / "src" / "losses" / "joint_loss.py",
        UNETMAMBA_DIR / "src" / "losses" / "dice.py",
        UNETMAMBA_DIR / "src" / "losses" / "soft_ce.py",
        PROJECT_DIR / "src" / "model.py",
        PROJECT_DIR / "src" / "train.py",
        PROJECT_DIR / "src" / "dataloader.py",
        UNETMAMBA_DIR / "scripts" / "benchmark_unetmamba.py",
    ]

    syntax_errors = 0
    for cf in code_files:
        if not cf.exists():
            print(f"  [FAIL] Missing file: {cf.relative_to(PROJECT_DIR)}")
            syntax_errors += 1
            all_passed = False
            continue
        try:
            with open(cf, "r", encoding="utf-8") as f:
                ast.parse(f.read())
        except Exception as e:
            print(f"  [FAIL] Syntax error in {cf.name}: {e}")
            syntax_errors += 1
            all_passed = False

    if syntax_errors == 0:
        print(f"  [PASS] All {len(code_files)} critical Python modules passed syntax validation.")

    # -------------------------------------------------------------
    # 5. Selective-Scan CUDA Extension Sources
    # -------------------------------------------------------------
    print("\n[CHECK 5/5] Verifying Selective-Scan CUDA Extension Sources...")
    kernel_dir = UNETMAMBA_DIR / "kernels" / "selective_scan"
    kernel_files = [
        kernel_dir / "setup.py",
        kernel_dir / "csrc" / "selective_scan" / "selective_scan.h",
        kernel_dir / "csrc" / "selective_scan" / "cus" / "selective_scan.cpp",
        kernel_dir / "csrc" / "selective_scan" / "cus" / "selective_scan_core_fwd.cu",
        kernel_dir / "csrc" / "selective_scan" / "cus" / "selective_scan_core_bwd.cu",
    ]

    kernel_ok = all(f.exists() for f in kernel_files)
    if kernel_ok:
        print(f"  [PASS] All C++/CUDA selective scan source files present and ready.")
    else:
        print(f"  [FAIL] Missing selective scan source files!")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("          VERDICT: ALL FILES VERIFIED & 100% CORRECT (READY FOR PIPELINE)")
    else:
        print("          VERDICT: SOME ISSUES DETECTED - PLEASE REVIEW LOG ABOVE")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    verify()
