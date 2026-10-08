import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
UNETMAMBA_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(UNETMAMBA_DIR / "src"))
sys.path.insert(0, str(UNETMAMBA_DIR))

from src.dataset import FloodPlanetDataset, build_records
from src.models.UNetMamba import UNetMamba
from src.losses.useful_loss import UnetMambaLoss
import configs.unetmamba_config as cfg

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_root = cfg.DATASET_ROOT if cfg.DATASET_ROOT.exists() else Path(r"D:\Basemodel\Datasets")
    all_records, event_names = build_records(data_root, expected_records=366)
    train_recs = [r for r in all_records if r["event"] in cfg.TRAIN_EVENTS]
    val_recs = [r for r in all_records if r["event"] in cfg.VALIDATION_EVENTS]

    train_dataset = FloodPlanetDataset(train_recs, dataset_root=data_root, patch_size=cfg.PATCH_SIZE, stride=cfg.STRIDE, ignore_index=cfg.IGNORE_INDEX, augment=True)
    val_dataset = FloodPlanetDataset(val_recs, dataset_root=data_root, patch_size=cfg.PATCH_SIZE, stride=cfg.STRIDE, ignore_index=cfg.IGNORE_INDEX, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    model = UNetMamba(in_chans=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES).to(device)
    criterion = UnetMambaLoss(ignore_index=-1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    t0 = time.time()
    model.train()
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
        if i % 10 == 0:
            print(f"Step {i}/{len(train_loader)} - Elapsed: {time.time()-t0:.2f}s", flush=True)
    epoch_time = time.time() - t0
    print(f"1 Training Epoch with batch_size=16, num_workers=2 took: {epoch_time:.2f} seconds")

if __name__ == '__main__':
    main()
