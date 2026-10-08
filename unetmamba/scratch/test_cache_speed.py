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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data_root = cfg.DATASET_ROOT if cfg.DATASET_ROOT.exists() else Path(r"D:\Basemodel\Datasets")
all_records, event_names = build_records(data_root, expected_records=366)
train_recs = [r for r in all_records if r["event"] in cfg.TRAIN_EVENTS]

train_dataset = FloodPlanetDataset(train_recs, dataset_root=data_root, patch_size=cfg.PATCH_SIZE, stride=cfg.STRIDE, ignore_index=cfg.IGNORE_INDEX, augment=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

model = UNetMamba(in_chans=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES).to(device)
criterion = UnetMambaLoss(ignore_index=-1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda')

print("Starting Epoch 1 (Disk load + Cache filling)...", flush=True)
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
epoch1_time = time.time() - t0
print(f"Epoch 1 finished in {epoch1_time:.2f} seconds!", flush=True)

print("Starting Epoch 2 (In-RAM cached)...", flush=True)
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
epoch2_time = time.time() - t0
print(f"Epoch 2 finished in {epoch2_time:.2f} seconds!", flush=True)
