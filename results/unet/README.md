# U-Net Matched Benchmark Results

## Dataset

FloodPlanet only.

## Fixed event split

### Training — 14 events

Colombia, Ghana, Nigeria, Paraguay, Somalia, Spain,
US-Alabama, US-Arkansas, US-Carolina, US-Kansas,
US-Nebraska, US-Oklahoma, US-Texas, Uzbekistan

### Validation — 2 events

Bolivia, US-Dakota

### Test — 3 events

Bangladesh, Cambodia, Nepal

## Model

Standard repository U-Net.

Input channels: 4
Output classes: 2
Bilinear upsampling: True

## Training

Seed: 2026
Patch size: 300 x 300
Stride: 150
Batch size: 8
Optimizer: Adam
Learning rate: 1e-4
Loss: CrossEntropyLoss(ignore_index=-1)
Epochs: 10
Augmentation: False

## Model selection

Best checkpoint selected using validation IoU.

Selected epoch: 6
Best validation IoU: 0.60003056

The test events were not used for model selection.

## Final test results

Mean IoU: 0.7326
Mean F1: 0.8417
Mean Precision: 0.8539
Mean Recall: 0.8345
Mean Accuracy: 0.8491
Mean Specificity: 0.8338

## Event-level results

Bangladesh:
IoU 0.591460
F1 0.743292

Cambodia:
IoU 0.803563
F1 0.891084

Nepal:
IoU 0.802772
F1 0.890598

## Important

The historical LOEO U-Net checkpoints were audited separately and are
not used for the primary matched benchmark comparison.

The frozen best checkpoint is stored separately because large model
weights are intentionally not committed to the Git repository.
