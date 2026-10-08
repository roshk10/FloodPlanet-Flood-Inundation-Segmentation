import sys
import time
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
UNETMAMBA_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(UNETMAMBA_DIR / "src"))
sys.path.insert(0, str(UNETMAMBA_DIR))

from src.dataset import FloodPlanetDataset, build_records
from src.models.UNetMamba import UNetMamba
import configs.unetmamba_config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

t0 = time.time()
data_root = cfg.DATASET_ROOT if cfg.DATASET_ROOT.exists() else Path(r"D:\Basemodel\Datasets")
all_records, event_names = build_records(data_root, expected_records=366)
train_recs = [r for r in all_records if r["event"] in cfg.TRAIN_EVENTS]
print(f"Records loaded in {time.time()-t0:.2f}s, train_recs count: {len(train_recs)}")

t0 = time.time()
train_dataset = FloodPlanetDataset(train_recs, dataset_root=data_root, patch_size=cfg.PATCH_SIZE, stride=cfg.STRIDE, ignore_index=cfg.IGNORE_INDEX, augment=True)
print(f"Dataset created (samples: {len(train_dataset)}) in {time.time()-t0:.2f}s")

t0 = time.time()
item0 = train_dataset[0]
print(f"Dataset __getitem__(0) loaded in {time.time()-t0:.2f}s. Image shape: {item0[0].shape}, Label shape: {item0[1].shape}")

t0 = time.time()
model = UNetMamba(in_chans=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES).to(device)
print(f"Model created & moved to GPU in {time.time()-t0:.2f}s")

x = torch.randn(16, 4, 512, 512, device=device)
t0 = time.time()
with torch.amp.autocast('cuda', dtype=torch.float16):
    out = model(x)
torch.cuda.synchronize()
print(f"Dummy batch forward pass (batch=16, 512x512) took {time.time()-t0:.2f}s")
