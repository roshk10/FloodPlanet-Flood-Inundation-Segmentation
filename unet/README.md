# FloodPlanet U-Net

This directory contains the U-Net baseline implementation and inference
utilities for FloodPlanet flood-inundation segmentation.

## Components

- `src/model.py`: U-Net architecture.
- `src/dataset.py`: FloodPlanet dataset loader.
- `src/dataloader.py`: DataLoader utilities.
- `src/train.py`: Training helpers.
- `scripts/predict_floodplanet.py`: Full-scene inference.
- `requirements-inference.txt`: Inference dependencies.

## Requirements

Use Python 3.13 with a compatible PyTorch installation.

Install dependencies from the repository root using:

`python -m pip install -r unet/requirements-inference.txt`

## Trained checkpoint

The approximately 198 MB trained checkpoint is intentionally not committed
to the Git repository.

Place it at:

`unet/checkpoints/best_val_iou.pth`

Alternatively, provide its location through the `--checkpoint` argument.

The checkpoint must contain a `model_state_dict` compatible with the
four-channel, two-class U-Net configured with bilinear upsampling.

## Dataset

The FloodPlanet dataset is not included in this repository. Obtain it
separately and provide a four-band PlanetScope TIFF.

For uint16 imagery, inference divides pixel values by 65536, matching
the preprocessing used by the training dataset.

## Run inference

From the repository root, run:

`python unet/scripts/predict_floodplanet.py --image "path/to/image.tif" --label "path/to/label.tif" --checkpoint "unet/checkpoints/best_val_iou.pth" --output-dir "outputs/unet_demo"`

Replace the input paths with files available on your computer.

The `--label` argument is optional. When supplied, the script reports
single-image metrics while excluding raw label value 0.

## Outputs

The inference script generates:

- A georeferenced binary prediction GeoTIFF.
- A binary flood-mask PNG.
- A visualization showing the image, prediction, and overlay.
- The ground-truth mask in the visualization when a label is supplied.

Predicted mask encoding:

- 0 = non-flood.
- 1 = flood.

Original FloodPlanet label encoding:

- 0 = no-data / ignored.
- 1 = non-flood.
- 2 = flood.

The model uses overlapping 300 x 300 patches with a stride of 150.
Overlapping flood probabilities are averaged before thresholding at 0.5.

## Evaluation notes

Single-image metrics are diagnostic examples and must not be reported as
the complete test-set benchmark.

See `results/unet/README.md` for the matched benchmark results.
