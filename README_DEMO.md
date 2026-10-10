# UNetMamba prediction demo in VS Code

This is an inference-only script. It does **not** train the model or modify the checkpoint.

## Place the script

Copy `predict_floodplanet.py` to:

`C:\Project\FloodPlanet_Project\unetmamba\scripts\predict_floodplanet.py`

Required existing file:

`C:\Project\FloodPlanet_Project\unetmamba\checkpoints\best.pt`

The corresponding `checkpoint_manifest.json` is used for SHA-256/size integrity checking when present.

## Run it in VS Code PowerShell

No notebook is required. Use the Python environment that has compatible PyTorch, Rasterio, Pillow, NumPy, `timm`, `einops`, `fvcore`, and a working selective-scan backend installed.

Example (replace the image path with a real 4-band FloodPlanet PlanetScope GeoTIFF on your machine):

```powershell
cd C:\Project\FloodPlanet_Project
python unetmamba\scripts\predict_floodplanet.py --image "D:\FloodPlanet\Bolivia\PS\BOL_1016.tif"
```

If the matching label file is available and you want optional metrics:

```powershell
python unetmamba\scripts\predict_floodplanet.py --image "D:\FloodPlanet\Bolivia\PS\BOL_1016.tif" --label "D:\FloodPlanet\Bolivia\labels\BOL_1016.tif"
```

Output files are created under `unetmamba\results\demo\`:

- `*_flood_prediction.tif`: georeferenced mask (`0=background`, `1=flood`, `255=NoData`)
- `*_flood_mask.png`: quick-look segmentation
- `*_flood_overlay.png`: display composite with predicted flood in red
- `*_summary.json`: checkpoint, runtime, valid-pixel and flood-area summary

## Important environment limitation

The UNetMamba implementation depends on a Mamba selective-scan backend. A local Windows/Intel-UHD setup may not have a compatible CUDA extension, and a reference CPU backend can be extremely slow. If the script reports that no backend is available, do not change the model architecture or disable strict checkpoint loading to work around it. Run actual inference in the known compatible Kaggle GPU environment and copy the generated result files to `unetmamba\results\demo\` for the VS Code presentation.

For a convincing demo, use one 4-band `.tif` input—not a screenshot or RGB export—so that inference preserves the model's trained input contract.
