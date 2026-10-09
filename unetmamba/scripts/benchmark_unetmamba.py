import os
import sys
import time
from pathlib import Path

# Add project root and unetmamba root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
UNETMAMBA_DIR = SCRIPT_DIR.parent
PROJECT_DIR = UNETMAMBA_DIR.parent

# Ensure unetmamba root is prioritized over repo root
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
from src.models.UNetMamba import UNetMamba
from src.losses.useful_loss import UnetMambaLoss
import configs.unetmamba_config as cfg


def run_benchmark():
    print("=" * 70)
    print("UNETMAMBA REAL FLOODPLANET BENCHMARK (RTX 5060 Ti 16 GB)")
    print("=" * 70)

    # 1. Device Diagnostics
    assert torch.cuda.is_available(), "CUDA is not available in PyTorch!"
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_cap = torch.cuda.get_device_capability(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {gpu_cap[0]}.{gpu_cap[1]}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")

    # Check selective-scan extension
    try:
        import selective_scan_cuda_core
        print("Selective-Scan Extension: CUDA core extension loaded successfully.")
    except ImportError:
        print("Selective-Scan Extension: Fallback / Not yet compiled as binary module.")

    # 2. Build Dataset Records
    data_root = cfg.DATASET_ROOT
    if not data_root.exists():
        data_root = Path(r"D:\Basemodel\FloodPlanet")

    print(f"\nLoading FloodPlanet from: {data_root}")
    all_records, event_names = build_records(data_root, expected_records=366)

    # Filter for Train Events
    train_records = [r for r in all_records if r["event"] in cfg.TRAIN_EVENTS]
    print(f"Train Events ({len(cfg.TRAIN_EVENTS)}): {len(train_records)} chips")

    # Create Dataset and DataLoader
    train_dataset = FloodPlanetDataset(
        records=train_records,
        dataset_root=data_root,
        patch_size=cfg.PATCH_SIZE,
        stride=cfg.STRIDE,
        ignore_index=cfg.IGNORE_INDEX,
        augment=True,
    )
    print(f"Total Train 512x512 Patches: {len(train_dataset)}")

    batch_size = cfg.BATCH_SIZE
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 for initial synchronous benchmark
        pin_memory=True,
    )

    # 3. Instantiate UNetMamba Model
    print("\nInstantiating UNetMamba model...")
    model = UNetMamba(
        pretrained=cfg.PRETRAINED,
        decode_channels=cfg.DECODE_CHANNELS,
        embed_dim=cfg.EMBED_DIM,
        in_chans=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f}M (Trainable: {trainable_params / 1e6:.2f}M)")

    criterion = UnetMambaLoss(ignore_index=cfg.IGNORE_INDEX).to(device)

    # Separate backbone and head parameters
    backbone_params = [p for n, p in model.named_parameters() if "encoder" in n]
    head_params = [p for n, p in model.named_parameters() if "encoder" not in n]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.BACKBONE_LEARNING_RATE},
        {"params": head_params, "lr": cfg.LEARNING_RATE},
    ], weight_decay=cfg.WEIGHT_DECAY)

    # 4. SINGLE SAMPLE FORWARD PASS (512x512)
    print("\n" + "-" * 50)
    print("STEP 1: Single Real 512x512 Sample Forward Pass")
    print("-" * 50)

    single_image, single_label = train_dataset[0]
    single_image = single_image.unsqueeze(0).to(device)  # [1, 4, 512, 512]
    
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_eval = model(single_image)
    torch.cuda.synchronize(device)
    single_fwd_time_ms = (time.perf_counter() - t0) * 1000

    print(f"Input Shape:  {list(single_image.shape)} (dtype: {single_image.dtype})")
    print(f"Output Shape: {list(out_eval.shape)} (dtype: {out_eval.dtype})")
    print(f"Single sample forward latency: {single_fwd_time_ms:.2f} ms")

    # 5. REAL TRAINING BATCH (FORWARD + LOSS + BACKWARD + OPTIMIZER)
    print("\n" + "-" * 50)
    print(f"STEP 2: Real Training Batch (Batch Size = {batch_size})")
    print("-" * 50)

    batch_size = 4
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats(device)

    batch_images, batch_labels = next(iter(loader))
    batch_images = batch_images.to(device, non_blocking=True)
    batch_labels = batch_labels.to(device, non_blocking=True)

    print(f"Batch Images Shape: {list(batch_images.shape)}")
    print(f"Batch Labels Shape: {list(batch_labels.shape)}")

    # Time Forward Pass & Loss with AMP
    torch.cuda.synchronize(device)
    t_start = time.perf_counter()
    
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)

    torch.cuda.synchronize(device)
    t_fwd = time.perf_counter()
    t_loss = t_fwd

    # Time Backward Pass
    scaler.scale(loss).backward()
    torch.cuda.synchronize(device)
    t_bwd = time.perf_counter()

    # Time Optimizer Step
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    t_opt = time.perf_counter()


    fwd_time_ms = (t_fwd - t_start) * 1000
    loss_time_ms = (t_loss - t_fwd) * 1000
    bwd_time_ms = (t_bwd - t_loss) * 1000
    opt_time_ms = (t_opt - t_bwd) * 1000
    total_iter_ms = (t_opt - t_start) * 1000

    peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_res_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    vram_headroom_gb = total_vram_gb - (peak_res_mb / 1024)

    # 6. ESTIMATE EPOCH TIME
    total_patches = len(train_dataset)
    num_batches = len(loader)
    est_epoch_sec = (total_iter_ms / 1000.0) * num_batches
    est_epoch_min = est_epoch_sec / 60.0

    print("\n" + "=" * 70)
    print("BENCHMARK REPORT")
    print("=" * 70)
    print(f"Loss Value:             {loss.item():.4f}")
    print(f"Batch Size:             {batch_size}")
    print(f"Forward Pass Time:      {fwd_time_ms:.2f} ms")
    print(f"Loss Compute Time:      {loss_time_ms:.2f} ms")
    print(f"Backward Pass Time:     {bwd_time_ms:.2f} ms")
    print(f"Optimizer Step Time:    {opt_time_ms:.2f} ms")
    print(f"Total Iteration Time:   {total_iter_ms:.2f} ms ({1000.0 / total_iter_ms:.2f} iter/sec)")
    print(f"Peak VRAM Allocated:    {peak_alloc_mb:.2f} MB ({peak_alloc_mb / 1024:.2f} GB)")
    print(f"Peak VRAM Reserved:     {peak_res_mb:.2f} MB ({peak_res_mb / 1024:.2f} GB)")
    print(f"VRAM Headroom:          {vram_headroom_gb:.2f} GB / {total_vram_gb:.2f} GB ({vram_headroom_gb / total_vram_gb * 100:.1f}% free)")
    print(f"Total Training Batches: {num_batches} (from {total_patches} patches)")
    print(f"Estimated Epoch Time:   {est_epoch_min:.2f} minutes ({est_epoch_sec:.1f} seconds)")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
