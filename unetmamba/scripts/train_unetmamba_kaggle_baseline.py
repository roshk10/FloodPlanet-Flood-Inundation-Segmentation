
from pathlib import Path
import csv
import json
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    "/kaggle/working/FloodPlanet-Flood-Inundation-Segmentation"
)

UNETMAMBA_ROOT = PROJECT_ROOT / "unetmamba"

DATASET_ROOT = Path(
    "/kaggle/input/datasets/leviosan/floodplanet/FloodPlanet"
)

CHECKPOINT_ROOT = UNETMAMBA_ROOT / "results" / "checkpoints"
METRICS_ROOT = UNETMAMBA_ROOT / "results" / "metrics"

CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
METRICS_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# IMPORT PATH
# ------------------------------------------------------------

import sys

sys.path.insert(0, str(UNETMAMBA_ROOT))

# Compiled selective-scan CUDA extension.
# This must be visible before importing UNetMamba/vmamba.py.
SELECTIVE_SCAN_ROOT = (
    UNETMAMBA_ROOT
    / "kernels"
    / "selective_scan"
)

sys.path.insert(
    0,
    str(SELECTIVE_SCAN_ROOT),
)

from src.dataset import FloodPlanetDataset, build_records
try:
    import selective_scan_cuda_core
except Exception as exc:
    raise RuntimeError(
        "selective_scan_cuda_core could not be imported. "
        "The compiled CUDA kernel is required for training."
    ) from exc

from src.models.UNetMamba import UNetMamba
from src.losses.useful_loss import UnetMambaLoss


# ============================================================
# FIXED PROJECT CONFIGURATION
# ============================================================

SEED = 2026

TRAIN_EVENTS = [
    "Colombia",
    "Ghana",
    "Nigeria",
    "Paraguay",
    "Somalia",
    "Spain",
    "US-Alabama",
    "US-Arkansas",
    "US-Carolina",
    "US-Kansas",
    "US-Nebraska",
    "US-Oklahoma",
    "US-Texas",
    "Uzbekistan",
]

VALIDATION_EVENTS = [
    "Bolivia",
    "US-Dakota",
]

TEST_EVENTS = [
    "Bangladesh",
    "Cambodia",
    "Nepal",
]

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

IN_CHANNELS = 4
NUM_CLASSES = 2
PATCH_SIZE = 512
STRIDE = 256
IGNORE_INDEX = -1

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 4
NUM_WORKERS = 2

# Extended from the original 20-epoch configuration because
# the real T4 benchmark gives ~2.95 h for 30 epochs.
MAX_EPOCHS = 14

# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

LEARNING_RATE = 6e-4
BACKBONE_LEARNING_RATE = 6e-5
WEIGHT_DECAY = 2.5e-4

# ------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------

SCHEDULER_T0 = 15
SCHEDULER_T_MULT = 2

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

PRETRAINED = False
EMBED_DIM = 64
DECODE_CHANNELS = 64

# ------------------------------------------------------------
# AMP
# ------------------------------------------------------------

AMP_ENABLED = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# WORKER SEEDING
# ============================================================

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# METRICS
# ============================================================

def update_confusion(
    confusion,
    prediction,
    target,
    ignore_index=-1,
):
    """
    Update 2x2 confusion matrix for binary flood segmentation.

    confusion:
        [ [TN, FP],
          [FN, TP] ]
    """

    prediction = prediction.detach().view(-1).cpu()
    target = target.detach().view(-1).cpu()

    valid = target != ignore_index

    prediction = prediction[valid]
    target = target[valid]

    if prediction.numel() == 0:
        return

    tn = ((prediction == 0) & (target == 0)).sum().item()
    fp = ((prediction == 1) & (target == 0)).sum().item()
    fn = ((prediction == 0) & (target == 1)).sum().item()
    tp = ((prediction == 1) & (target == 1)).sum().item()

    confusion[0][0] += tn
    confusion[0][1] += fp
    confusion[1][0] += fn
    confusion[1][1] += tp


def metrics_from_confusion(confusion):
    tn = confusion[0][0]
    fp = confusion[0][1]
    fn = confusion[1][0]
    tp = confusion[1][1]

    eps = 1e-12

    background_iou = tn / (
        tn + fp + fn + eps
    )

    flood_iou = tp / (
        tp + fp + fn + eps
    )

    miou = (
        background_iou + flood_iou
    ) / 2.0

    f1 = (
        2.0 * tp
        / (
            2.0 * tp
            + fp
            + fn
            + eps
        )
    )

    precision = tp / (
        tp + fp + eps
    )

    recall = tp / (
        tp + fn + eps
    )

    accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn + eps
    )

    specificity = tn / (
        tn + fp + eps
    )

    return {
        "mIoU": miou,
        "IoU_background": background_iou,
        "IoU_flood": flood_iou,
        "F1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "specificity": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ============================================================
# DATA SPLIT
# ============================================================

def build_fixed_split():
    records, event_names = build_records(
        DATASET_ROOT,
        expected_records=366,
    )

    train_records = [
        r
        for r in records
        if r["event"] in TRAIN_EVENTS
    ]

    val_records = [
        r
        for r in records
        if r["event"] in VALIDATION_EVENTS
    ]

    test_records = [
        r
        for r in records
        if r["event"] in TEST_EVENTS
    ]

    train_found = sorted(
        {r["event"] for r in train_records}
    )

    val_found = sorted(
        {r["event"] for r in val_records}
    )

    test_found = sorted(
        {r["event"] for r in test_records}
    )

    if set(train_found) != set(TRAIN_EVENTS):
        raise RuntimeError(
            f"Training event mismatch:\n"
            f"Expected: {TRAIN_EVENTS}\n"
            f"Found: {train_found}"
        )

    if set(val_found) != set(VALIDATION_EVENTS):
        raise RuntimeError(
            f"Validation event mismatch:\n"
            f"Expected: {VALIDATION_EVENTS}\n"
            f"Found: {val_found}"
        )

    if set(test_found) != set(TEST_EVENTS):
        raise RuntimeError(
            f"Test event mismatch:\n"
            f"Expected: {TEST_EVENTS}\n"
            f"Found: {test_found}"
        )

    if (
        set(TRAIN_EVENTS)
        & set(VALIDATION_EVENTS)
    ):
        raise RuntimeError(
            "Train/validation event leakage detected."
        )

    if (
        set(TRAIN_EVENTS)
        & set(TEST_EVENTS)
    ):
        raise RuntimeError(
            "Train/test event leakage detected."
        )

    if (
        set(VALIDATION_EVENTS)
        & set(TEST_EVENTS)
    ):
        raise RuntimeError(
            "Validation/test event leakage detected."
        )

    return (
        records,
        event_names,
        train_records,
        val_records,
        test_records,
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    scheduler,
    device,
    epoch,
):
    model.train()

    running_loss = 0.0
    batches = 0

    epoch_start = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=AMP_ENABLED,
        ):
            outputs = model(images)

        # ----------------------------------------------------
        # Numerical-stability rule established by the real
        # FloodPlanet smoke test:
        # compute the loss from FP32 logits.
        # ----------------------------------------------------

        outputs_fp32 = tuple(
            output.float()
            for output in outputs
        )

        loss = criterion(
            outputs_fp32,
            labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at "
                f"epoch={epoch + 1}, "
                f"batch={batch_idx + 1}: "
                f"{loss.item()}"
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # ----------------------------------------------------
        # Warm restart scheduler is expressed in epochs.
        # Fractional epoch stepping keeps T0=15 in epoch units.
        # ----------------------------------------------------

        progress = (
            epoch
            + (batch_idx + 1) / len(loader)
        )

        scheduler.step(progress)

        running_loss += loss.item()
        batches += 1

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - epoch_start
    )

    return (
        running_loss / max(batches, 1),
        elapsed,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    batches = 0

    confusion = [
        [0, 0],
        [0, 0],
    ]

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=AMP_ENABLED,
        ):
            logits = model(images)

        # Evaluation returns only the main logits.
        logits_fp32 = logits.float()

        loss = criterion(
            logits_fp32,
            labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss: "
                f"{loss.item()}"
            )

        prediction = (
            torch.argmax(
                logits_fp32,
                dim=1,
            )
            .to(torch.int64)
        )

        update_confusion(
            confusion,
            prediction,
            labels,
            IGNORE_INDEX,
        )

        running_loss += loss.item()
        batches += 1

    metrics = metrics_from_confusion(
        confusion
    )

    metrics["val_loss"] = (
        running_loss / max(batches, 1)
    )

    return metrics


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_miou,
    history,
):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_mIoU": best_val_miou,
        "history": history,
        "config": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "patch_size": PATCH_SIZE,
            "stride": STRIDE,
            "max_epochs": MAX_EPOCHS,
            "num_workers": NUM_WORKERS,
            "in_channels": IN_CHANNELS,
            "num_classes": NUM_CLASSES,
            "embed_dim": EMBED_DIM,
            "decode_channels": DECODE_CHANNELS,
            "pretrained": PRETRAINED,
            "learning_rate": LEARNING_RATE,
            "backbone_learning_rate": BACKBONE_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler_t0": SCHEDULER_T0,
            "scheduler_t_mult": SCHEDULER_T_MULT,
            "ignore_index": IGNORE_INDEX,
            "amp_enabled": AMP_ENABLED,
            "train_events": TRAIN_EVENTS,
            "validation_events": VALIDATION_EVENTS,
            "test_events": TEST_EVENTS,
            "monitor_metric": "val_mIoU",
        },
    }

    torch.save(
        state,
        path,
    )


# ============================================================
# CSV HISTORY
# ============================================================

def save_history_csv(history, path):
    if not history:
        return

    fieldnames = list(history[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(history)


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    device = torch.device("cuda:0")

    print("=" * 90)
    print("UNETMAMBA — FLOODPLANET TRAINING")
    print("=" * 90)

    print()
    print("[1] Hardware")
    print("-" * 90)

    gpu_name = torch.cuda.get_device_name(
        device
    )

    vram_gb = (
        torch.cuda.get_device_properties(
            device
        ).total_memory
        / (1024 ** 3)
    )

    print(f"GPU: {gpu_name}")
    print(
        f"VRAM (GB): {vram_gb:.3f}"
    )

    # --------------------------------------------------------
    # Records
    # --------------------------------------------------------

    print()
    print("[2] Building FloodPlanet split")
    print("-" * 90)

    (
        records,
        event_names,
        train_records,
        val_records,
        test_records,
    ) = build_fixed_split()

    print(
        f"Total records: {len(records)}"
    )

    print(
        f"Train records: {len(train_records)}"
    )

    print(
        f"Validation records: {len(val_records)}"
    )

    print(
        f"Test records: {len(test_records)}"
    )

    print(
        f"Training events: {TRAIN_EVENTS}"
    )

    print(
        f"Validation events: {VALIDATION_EVENTS}"
    )

    print(
        f"Test events: {TEST_EVENTS}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print()
    print("[3] Creating datasets")
    print("-" * 90)

    train_dataset = FloodPlanetDataset(
        records=train_records,
        dataset_root=DATASET_ROOT,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        ignore_index=IGNORE_INDEX,
        augment=True,
    )

    val_dataset = FloodPlanetDataset(
        records=val_records,
        dataset_root=DATASET_ROOT,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        ignore_index=IGNORE_INDEX,
        augment=False,
    )

    print(
        f"Training patches: {len(train_dataset)}"
    )

    print(
        f"Validation patches: {len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=True,
    )

    print(
        f"Training batches / epoch: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches / epoch: "
        f"{len(val_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("[4] Creating UNetMamba")
    print("-" * 90)

    model = UNetMamba(
        pretrained=PRETRAINED,
        decode_channels=DECODE_CHANNELS,
        embed_dim=EMBED_DIM,
        in_chans=IN_CHANNELS,
        num_classes=NUM_CLASSES,
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{total_params:,} "
        f"({total_params / 1e6:.3f}M)"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = UnetMambaLoss(
        ignore_index=IGNORE_INDEX
    )

    print(
        "Loss: UnetMambaLoss"
    )

    print(
        "Loss computation: FP32"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    model.encoder.parameters(),
                "lr":
                    BACKBONE_LEARNING_RATE,
            },
            {
                "params":
                    model.decoder.parameters(),
                "lr":
                    LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "Optimizer: AdamW"
    )

    print(
        f"Backbone LR: "
        f"{BACKBONE_LEARNING_RATE}"
    )

    print(
        f"Decoder LR: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingWarmRestarts(
            optimizer,
            T_0=SCHEDULER_T0,
            T_mult=SCHEDULER_T_MULT,
        )
    )

    # --------------------------------------------------------
    # AMP scaler
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=AMP_ENABLED,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    last_checkpoint = (
        CHECKPOINT_ROOT / "last.pt"
    )

    start_epoch = 0
    best_val_miou = -math.inf
    history = []

    if last_checkpoint.exists():

        print()
        print("[5] Resuming existing checkpoint")
        print("-" * 90)

        checkpoint = torch.load(
            last_checkpoint,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_val_miou = checkpoint[
            "best_val_mIoU"
        ]

        history = checkpoint.get(
            "history",
            [],
        )

        print(
            f"Resuming from epoch "
            f"{start_epoch + 1}"
        )

        print(
            f"Previous best val_mIoU: "
            f"{best_val_miou:.6f}"
        )

    else:

        print()
        print("[5] Starting fresh training")
        print("-" * 90)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        MAX_EPOCHS,
    ):

        epoch_start = time.perf_counter()

        train_loss, train_time = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                device=device,
                epoch=epoch,
            )
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        current_lr = (
            optimizer.param_groups[1]["lr"]
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics[
                "val_loss"
            ],
            "val_mIoU": val_metrics[
                "mIoU"
            ],
            "val_F1": val_metrics[
                "F1"
            ],
            "val_precision": val_metrics[
                "precision"
            ],
            "val_recall": val_metrics[
                "recall"
            ],
            "val_accuracy": val_metrics[
                "accuracy"
            ],
            "val_specificity": val_metrics[
                "specificity"
            ],
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
            "train_time_sec": train_time,
        }

        history.append(row)

        # ----------------------------------------------------
        # Checkpoint: every epoch
        # ----------------------------------------------------

        epoch_checkpoint = (
            CHECKPOINT_ROOT
            / f"epoch_{epoch + 1:02d}.pt"
        )

        save_checkpoint(
            path=epoch_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_miou=best_val_miou,
            history=history,
        )

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            path=last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_miou=best_val_miou,
            history=history,
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            val_metrics["mIoU"]
            > best_val_miou
        ):

            best_val_miou = (
                val_metrics["mIoU"]
            )

            best_checkpoint = (
                CHECKPOINT_ROOT
                / "best.pt"
            )

            save_checkpoint(
                path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_miou=best_val_miou,
                history=history,
            )

            best_marker = "  <-- BEST"

        else:
            best_marker = ""

        save_history_csv(
            history,
            METRICS_ROOT
            / "training_history.csv",
        )

        print()
        print(
            "=" * 90
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{MAX_EPOCHS}"
            f"{best_marker}"
        )

        print(
            f"Train loss      : "
            f"{train_loss:.6f}"
        )

        print(
            f"Val loss        : "
            f"{val_metrics['val_loss']:.6f}"
        )

        print(
            f"Val mIoU        : "
            f"{val_metrics['mIoU']:.6f}"
        )

        print(
            f"Val F1          : "
            f"{val_metrics['F1']:.6f}"
        )

        print(
            f"Val precision   : "
            f"{val_metrics['precision']:.6f}"
        )

        print(
            f"Val recall      : "
            f"{val_metrics['recall']:.6f}"
        )

        print(
            f"Val accuracy    : "
            f"{val_metrics['accuracy']:.6f}"
        )

        print(
            f"Val specificity : "
            f"{val_metrics['specificity']:.6f}"
        )

        print(
            f"LR              : "
            f"{current_lr:.8f}"
        )

        print(
            f"Epoch time      : "
            f"{epoch_time / 60.0:.2f} min"
        )

        print(
            f"Best val mIoU   : "
            f"{best_val_miou:.6f}"
        )

        print(
            "=" * 90
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("UNETMAMBA TRAINING COMPLETE")
    print("=" * 90)

    print(
        f"Completed epochs: "
        f"{MAX_EPOCHS}"
    )

    print(
        f"Best validation mIoU: "
        f"{best_val_miou:.6f}"
    )

    print()
    print(
        f"Best checkpoint: "
        f"{CHECKPOINT_ROOT / 'best.pt'}"
    )

    print(
        f"Last checkpoint: "
        f"{CHECKPOINT_ROOT / 'last.pt'}"
    )

    print(
        f"History CSV: "
        f"{METRICS_ROOT / 'training_history.csv'}"
    )


if __name__ == "__main__":
    main()
