FloodPlanet — Flood Inundation Semantic Segmentation

Semantic segmentation for flood inundation mapping using multi-sensor satellite imagery.

Problem Statement

Flood inundation mapping is an important task for disaster monitoring and response. The objective of this project is to perform pixel-level semantic segmentation of flood inundation, where each image pixel is classified as either flood water or non-flood.

This project uses the FloodPlanet dataset and reproduces the base U-Net segmentation methodology described in the 2025 FloodPlanet research paper.

Research Objective

The current objective is to reproduce and evaluate the FloodPlanet base-paper U-Net methodology using the FloodPlanet dataset.

The reproduction focuses on:

PlanetScope imagery and corresponding FloodPlanet labels
4-channel image input
2-class flood segmentation
300 × 300 image patches
Leave-One-Event-Out (LOEO) evaluation across the 19 FloodPlanet flood events
reproducible training and evaluation

Further model comparisons and the final research extension will be added only after the base-model reproduction and evaluation are completed.

Dataset

The project uses the FloodPlanet dataset, which contains manually annotated flood inundation labels associated with globally distributed flood events.

The dataset used for this project contains:

19 flood events
366 PlanetScope image-label pairs
PlanetScope imagery with 4 input bands
corresponding flood labels
high-resolution flood annotations

The actual FloodPlanet imagery is not stored in this repository.

Only a small metadata/index file is included under:

data/metadata/planetscope_index.csv

The raw dataset must be obtained separately and placed in the local dataset location specified through the project configuration.

Base Paper

The current implementation is based on the following FloodPlanet research article:

Assessing Inundation Semantic Segmentation Models Trained on High- versus Low-Resolution Labels using FloodPlanet, a Manually Labeled Multi-Sourced High-Resolution Flood Dataset

Journal of Remote Sensing, 2025.

DOI:

https://doi.org/10.34133/remotesensing.0575

The paper establishes a U-Net-based segmentation baseline for FloodPlanet.

Methodology

The current project implementation uses a U-Net architecture for binary flood inundation segmentation.

Input
PlanetScope image
    ↓
4 input channels
Preprocessing
FloodPlanet image
    ↓
PlanetScope normalization
    ↓
300 × 300 patches
    ↓
paired flood label

Boundary patches are handled so that the model receives fixed-size 300 × 300 inputs.

Training augmentation applies the same geometric transformation to the image and its corresponding label.

Model

The current base model is U-Net with:

4 input channels
2 output classes
convolutional encoder
max-pooling downsampling
mirrored decoder
skip connections
bilinear upsampling
batch normalization
ReLU activations

The implementation follows the base-paper methodology and the referenced U-Net implementation used for the FloodPlanet baseline. Project-specific implementation details and modifications are documented separately.

LOEO Evaluation Protocol

The current evaluation protocol follows Leave-One-Event-Out (LOEO).

FloodPlanet contains 19 flood events. For each fold:

19 flood events
       │
       ├── 1 event → held out for evaluation
       │
       └── remaining 18 events → training

The process is repeated so that each flood event serves as the held-out evaluation event.

This provides an event-level evaluation of how well the model generalizes to a flood event that was not used during training.

Current Training Status

Base U-Net training is currently being executed through distributed Kaggle and Google Colab environments because of compute and runtime constraints.

Completed fold checkpoints and training artifacts are backed up outside this repository.

The repository currently contains the clean project foundation and base U-Net source components. The full distributed training implementation will be consolidated into the repository after the ongoing experiments are completed and verified.

Current Project Status
Base model:              U-Net
Dataset:                 FloodPlanet only
Evaluation:              19-event LOEO
Training:                In progress
Final centralized
evaluation:              In progress
Advanced model:          Not yet implemented
Final research
extension:               Not yet implemented

No final project metrics are reported in this repository yet.

Results will be added only after they have been generated and verified from the completed experiments.

Repository Structure
FloodPlanet_Project/
│
├── README.md
├── requirements.txt
├── .gitignore
├── setup_windows.bat
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── dataloader.py
│   ├── model.py
│   └── train.py
│
├── configs/
│   ├── base_unet.yaml
│   └── paths.yaml
│
├── docs/
│   ├── project_scope.md
│   ├── dataset_structure.md
│   ├── preprocessing.md
│   ├── evaluation_protocol.md
│   ├── training_setup.md
│   └── experiment_log.md
│
├── data/
│   ├── metadata/
│   ├── raw/
│   └── processed/
│
├── checkpoints/
├── outputs/
└── notebooks/
Installation

The repository uses Python and PyTorch-based components.

Dependencies will be finalized and pinned after the current verified training environment has been consolidated into the repository.

A local virtual environment should be created outside version-controlled source content:

py -m venv .venv

Activate it with:

.\.venv\Scripts\Activate.ps1

Then install the finalized dependencies:

pip install -r requirements.txt
Training and Evaluation

The complete production training and evaluation workflow is currently being executed through the project’s Kaggle and Google Colab environments.

The repository will be updated with the verified centralized training and evaluation entry points after the current distributed base-model experiments are completed.

Until then, this README intentionally does not provide unverified commands for final training or evaluation.

Research Provenance

The base U-Net methodology is derived from the FloodPlanet 2025 research paper and the implementation approach referenced by the authors.

Externally sourced methodology and implementation references will be identified explicitly.

Project-specific modifications, experiment orchestration, dataset handling, checkpoint management, and other implementation changes will be documented separately rather than presented as entirely original work.

Reproducibility and Data Policy

The following are intentionally excluded from GitHub:

FloodPlanet imagery and labels
trained model checkpoints
large ZIP/archive files
temporary Kaggle/Colab runtime files
generated caches
secrets and API keys
unverified experimental outputs

GitHub is used as the source of truth for the project's clean source code and documentation.

Project Status Notice

Final results are still in progress.

The repository currently represents the clean project foundation and base U-Net implementation. The ongoing distributed training experiments and centralized final evaluation will be incorporated after verification.