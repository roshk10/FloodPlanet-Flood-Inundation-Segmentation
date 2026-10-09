# FloodPlanet Project

Semantic Segmentation for Flood Inundation Mapping Using Multi-Sensor Satellite Imagery.

## Workflow
1. Project environment setup
2. FloodPlanet dataset inspection and audit
3. Dataset loader and preprocessing
4. Base-paper U-Net reproduction
5. Two established baseline models
6. Research gap identification
7. Proposed extension
8. Controlled experiments and analysis
9. Final demo and documentation

## Repository rules
- Keep raw FloodPlanet data outside GitHub.
- Keep source code in `src/` and reusable scripts in `scripts/`.
- Keep environment-specific paths in `configs/paths.yaml`.
- Do not fabricate or manually edit experiment results.

## UNetMamba Training Progress

**Model:** UNetMamba (ResT-Lite encoder + MambaSegDecoder)
**GPU:** NVIDIA GeForce RTX 5060 Ti
**Config:** batch_size=16, lr=3e-4, AMP FP16, CosineAnnealingLR
**Dataset:** FloodPlanet — 832 train patches, 456 val patches (512×512)
**Checkpoints:** `unetmamba/checkpoints/unetmamba/`

| Epoch | Train Loss | Val Loss | mIoU   | Flood IoU | F1 Score | Accuracy |
|-------|-----------|----------|--------|-----------|----------|----------|
| 1     | 1.1715    | 1.3275   | 0.3072 | 0.0000    | 0.3805   | 0.6143   |
| 2     | 1.0265    | 1.0523   | 0.5202 | 0.3650    | 0.6705   | 0.7264   |
| 3     | 0.9482    | 0.8497   | 0.6207 | 0.5359    | 0.7626   | 0.7803   |
| 4     | 0.9014    | 0.8592   | 0.6101 | 0.5172    | 0.7537   | 0.7747   |
| 5     | 0.9229    | 0.9930   | 0.5536 | 0.5073    | 0.7116   | 0.7167   |
| 6     | 0.9040    | 0.8631   | 0.6242 | 0.5437    | 0.7656   | 0.7816   |
| **7** | **0.8913**| **0.8366**| **0.6338** | **0.5574** | **0.7732** | **0.7877** |

**Best so far:** Epoch 7 — mIoU: **0.6338**, Flood IoU: **0.5574** ← saved to `best.pth`
**Last saved:** Epoch 7 — `last.pth`
**Status:** Paused at Epoch 7. Resume with 11 more epochs using command below.

### Resume Command
```powershell
cd unetmamba
python scripts/train_unetmamba.py --resume last --add_epochs 11 --batch_size 16 --lr 3e-4
```
