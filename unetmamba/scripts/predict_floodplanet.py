#!/usr/bin/env python3
"""Run the trained UNetMamba baseline on one 4-band FloodPlanet GeoTIFF.

Run from the repository root, for example:
  python unetmamba/scripts/predict_floodplanet.py --image "D:/FloodPlanet/Bolivia/PS/BOL_1016.tif"

Outputs under unetmamba/results/demo unless --output-dir is supplied:
  *_flood_prediction.tif  georeferenced mask (0=background, 1=flood, 255=NoData)
  *_flood_mask.png        quick-look mask
  *_flood_overlay.png     first-three-band display composite with predicted flood overlay
  *_summary.json          run metadata and optional metrics

This script does not train or modify model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_PATH = Path(__file__).resolve()
UNETMAMBA_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CHECKPOINT = UNETMAMBA_ROOT / "checkpoints" / "best.pt"
DEFAULT_OUTPUT_DIR = UNETMAMBA_ROOT / "results" / "demo"
MANIFEST_PATH = UNETMAMBA_ROOT / "checkpoints" / "checkpoint_manifest.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_checkpoint(checkpoint_path: Path, skip_hash: bool) -> str:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Place the supplied best.pt in unetmamba/checkpoints/best.pt."
        )
    actual_hash = sha256_file(checkpoint_path) if not skip_hash else "SKIPPED"
    if skip_hash or not MANIFEST_PATH.is_file():
        return actual_hash

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    # The tracked manifest keys correspond to the canonical local checkpoint names.
    key = "best_pt" if checkpoint_path.name.lower() == "best.pt" else "last_pt"
    expected = manifest.get(key, {})
    expected_hash = str(expected.get("sha256", "")).upper()
    expected_bytes = expected.get("bytes")
    if expected_bytes is not None and checkpoint_path.stat().st_size != int(expected_bytes):
        raise RuntimeError(
            f"Checkpoint size does not match the manifest.\n"
            f"Expected: {expected_bytes:,} bytes\n"
            f"Actual:   {checkpoint_path.stat().st_size:,} bytes\n"
            "Use the matching checkpoint; do not retrain the baseline for this demo."
        )
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            "Checkpoint SHA-256 does not match checkpoint_manifest.json.\n"
            f"Expected: {expected_hash}\nActual:   {actual_hash}"
        )
    print(f"Checkpoint integrity: verified ({actual_hash})")
    return actual_hash


def load_model(checkpoint_path: Path, device: torch.device, allow_slow_cpu: bool) -> torch.nn.Module:
    # Make both the package root and optional selective-scan extension importable.
    sys.path.insert(0, str(UNETMAMBA_ROOT))
    selective_scan_root = UNETMAMBA_ROOT / "kernels" / "selective_scan"
    if selective_scan_root.exists():
        sys.path.insert(0, str(selective_scan_root))

    try:
        from src.models.UNetMamba import UNetMamba
        import src.models.vmamba as vmamba
    except Exception as exc:
        raise RuntimeError(
            "Could not import the project's UNetMamba implementation.\n"
            "Check that unetmamba/src/models/UNetMamba.py and its source dependencies exist, "
            "and install the project's compatible dependencies.\n"
            f"Import error: {type(exc).__name__}: {exc}"
        ) from exc

    cuda_backend = any(
        getattr(vmamba, name, None) is not None
        for name in ("selective_scan_cuda_core", "selective_scan_cuda_oflex", "selective_scan_cuda")
    )
    reference_backend = getattr(vmamba, "selective_scan_ref", None) is not None

    if cuda_backend and device.type != "cuda":
        raise RuntimeError(
            "A CUDA selective-scan extension is imported, but the selected device is not CUDA. "
            "Use --device cuda in a compatible NVIDIA CUDA environment (for example, the verified Kaggle T4 setup)."
        )
    if not cuda_backend and not reference_backend:
        raise RuntimeError(
            "No selective-scan backend is available. The model requires a compatible compiled CUDA extension "
            "or the Mamba reference implementation. On Windows/Intel UHD, the baseline may not run locally; "
            "run inference in the verified Kaggle GPU environment and bring the outputs into VS Code for the demo."
        )
    if device.type == "cpu" and reference_backend and not cuda_backend and not allow_slow_cpu:
        raise RuntimeError(
            "Only the slow reference selective-scan backend is available on CPU. "
            "Use --allow-slow-cpu for a small demo image, or run the model on a compatible CUDA GPU."
        )
    if device.type == "cpu" and reference_backend and not cuda_backend:
        print("WARNING: using reference selective scan on CPU; inference may be very slow.")

    model = UNetMamba(
        pretrained=False,
        decode_channels=64,
        embed_dim=64,
        in_chans=4,
        num_classes=2,
    )

    try:
        # This is the trusted project checkpoint generated by this training code.
        try:
            checkpoint: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # older PyTorch releases do not accept weights_only
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Unable to read checkpoint: {checkpoint_path}\n{exc}") from exc

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a recognizable model state dictionary.")

    # Permit checkpoints saved by DataParallel without silently dropping other keys.
    if state_dict and all(str(k).startswith("module.") for k in state_dict):
        state_dict = {str(k)[7:]: v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint parameters do not match this UNetMamba source. "
            "Do not use strict=False to hide this mismatch.\n"
            f"Load error: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    return model


def tile_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def stretch_to_uint8(image_chw: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Make a display-only composite from the first three raster bands."""
    display = np.moveaxis(image_chw[:3], 0, -1)
    out = np.zeros(display.shape, dtype=np.uint8)
    for channel in range(3):
        values = display[..., channel][valid_mask & np.isfinite(display[..., channel])]
        if values.size == 0:
            continue
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            high = low + 1.0
        stretched = (display[..., channel] - low) / (high - low)
        out[..., channel] = np.clip(stretched * 255.0, 0, 255).astype(np.uint8)
    return out


def compute_metrics(pred: np.ndarray, label_path: Path, valid_image: np.ndarray) -> dict[str, float | int]:
    with rasterio.open(label_path) as label_src:
        raw = label_src.read(1)
        label_valid = label_src.read_masks(1) > 0
        if raw.shape != pred.shape:
            raise ValueError(
                f"Label dimensions {raw.shape} do not match image/prediction dimensions {pred.shape}."
            )
    # FloodPlanet source-label convention used by this project: 1=background, 2=flood.
    gt = np.full(raw.shape, 255, dtype=np.uint8)
    gt[raw == 1] = 0
    gt[raw == 2] = 1
    valid = valid_image & label_valid & (gt != 255)
    if not np.any(valid):
        raise ValueError("No valid overlapping pixels remain for optional label metrics.")

    y_true = gt[valid].astype(np.uint8)
    y_pred = pred[valid].astype(np.uint8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return {
        "metric_pixel_count": int(valid.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flood_iou": float(iou), "flood_f1": float(f1),
        "precision": float(precision), "recall": float(recall),
        "accuracy": float(accuracy),
    }


def run_inference(
    model: torch.nn.Module,
    image_path: Path,
    checkpoint_path: Path,
    checkpoint_hash: str,
    output_dir: Path,
    device: torch.device,
    patch_size: int,
    stride: int,
    label_path: Path | None,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input GeoTIFF not found: {image_path}")
    if patch_size < 32 or stride < 1 or stride > patch_size:
        raise ValueError("Use --patch-size >= 32 and 1 <= --stride <= --patch-size.")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with rasterio.open(image_path) as src:
        if src.count != 4:
            raise ValueError(
                f"Expected a 4-band PlanetScope raster, got {src.count} bands. "
                "Provide the original 4-band PS GeoTIFF, not a rendered RGB image."
            )
        image_raw = src.read().astype(np.float32)
        profile = src.profile.copy()
        height, width = src.height, src.width
        # Use the raster's per-band validity masks so NoData is not rendered as valid output.
        valid_mask = np.all(src.read_masks() > 0, axis=0)

    # Match the trained FloodPlanet preprocessing: integer reflectance scaled by 65536.
    image = np.clip(image_raw / 65536.0, 0.0, 1.0)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    prediction_count = np.zeros((height, width), dtype=np.uint16)
    y_starts = tile_starts(height, patch_size, stride)
    x_starts = tile_starts(width, patch_size, stride)
    tile_total = len(y_starts) * len(x_starts)

    print(f"Image: {image_path}")
    print(f"Raster shape: {height} x {width} | bands: 4 | tiles: {tile_total}")
    print(f"Device: {device} | patch: {patch_size} | stride: {stride}")
    print("Running inference; no training or checkpoint modification will occur...")

    tile_index = 0
    with torch.inference_mode():
        for top in y_starts:
            for left in x_starts:
                tile_index += 1
                tile = np.zeros((4, patch_size, patch_size), dtype=np.float32)
                crop_h = min(patch_size, height - top)
                crop_w = min(patch_size, width - left)
                tile[:, :crop_h, :crop_w] = image[:, top:top + crop_h, left:left + crop_w]
                tensor = torch.from_numpy(tile[None]).to(device=device, dtype=torch.float32)
                logits = model(tensor)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if logits.ndim != 4 or logits.shape[1] != 2:
                    raise RuntimeError(f"Expected logits shaped (1, 2, H, W), received {tuple(logits.shape)}")
                if logits.shape[-2:] != (patch_size, patch_size):
                    logits = F.interpolate(logits, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
                flood_prob = torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()
                probability_sum[top:top + crop_h, left:left + crop_w] += flood_prob[:crop_h, :crop_w]
                prediction_count[top:top + crop_h, left:left + crop_w] += 1
                if tile_index == 1 or tile_index == tile_total or tile_index % 10 == 0:
                    print(f"  tile {tile_index}/{tile_total}")

    averaged_prob = probability_sum / np.maximum(prediction_count, 1)
    pred = (averaged_prob >= 0.5).astype(np.uint8)
    pred[~valid_mask] = 255

    stem = image_path.stem
    mask_png = output_dir / f"{stem}_flood_mask.png"
    overlay_png = output_dir / f"{stem}_flood_overlay.png"
    prediction_tif = output_dir / f"{stem}_flood_prediction.tif"
    summary_path = output_dir / f"{stem}_summary.json"

    # A single-band georeferenced raster: 0 background, 1 flood, 255 invalid/NoData.
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(prediction_tif, "w", **profile) as dst:
        dst.write(pred, 1)
        dst.set_band_description(1, "UNetMamba flood prediction: 0=background, 1=flood, 255=NoData")

    mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mask_rgb[pred == 1] = (255, 55, 55)
    mask_rgb[pred == 255] = (100, 100, 100)
    Image.fromarray(mask_rgb, mode="RGB").save(mask_png)

    display = stretch_to_uint8(image, valid_mask)
    overlay = display.copy()
    flood = pred == 1
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    red[..., 1] = 45
    red[..., 2] = 45
    overlay[flood] = (0.52 * overlay[flood] + 0.48 * red[flood]).astype(np.uint8)
    overlay[~valid_mask] = (95, 95, 95)
    Image.fromarray(overlay, mode="RGB").save(overlay_png)

    valid_count = int(valid_mask.sum())
    flood_count = int(np.sum(pred == 1))
    summary: dict[str, Any] = {
        "model": "UNetMamba",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "input_geotiff": str(image_path.resolve()),
        "output_prediction_geotiff": str(prediction_tif.resolve()),
        "output_mask_png": str(mask_png.resolve()),
        "output_overlay_png": str(overlay_png.resolve()),
        "device": str(device),
        "raster_width": width,
        "raster_height": height,
        "input_bands": 4,
        "patch_size": patch_size,
        "stride": stride,
        "tiles_inferred": tile_total,
        "valid_pixels": valid_count,
        "predicted_flood_pixels": flood_count,
        "predicted_flood_percent_of_valid_pixels": round(100.0 * flood_count / valid_count, 4) if valid_count else None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "note": "PNG composite uses the first three raster bands for display; source band identities are not inferred from band metadata.",
    }
    if label_path is not None:
        summary["optional_label_metrics"] = compute_metrics(pred, label_path, valid_mask)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSUCCESS: inference completed")
    print(f"Flood pixels among valid pixels: {flood_count:,}/{valid_count:,}")
    print(f"Prediction GeoTIFF: {prediction_tif}")
    print(f"Mask PNG:           {mask_png}")
    print(f"Overlay PNG:        {overlay_png}")
    print(f"Run summary:        {summary_path}")
    if label_path is not None:
        print("Optional label metrics:")
        for key, value in summary["optional_label_metrics"].items():
            print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained UNetMamba on a 4-band FloodPlanet GeoTIFF.")
    parser.add_argument("--image", required=True, type=Path, help="Path to a 4-band PlanetScope .tif/.tiff chip")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Defaults to unetmamba/checkpoints/best.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for prediction artifacts")
    parser.add_argument("--label", type=Path, default=None, help="Optional matching raw FloodPlanet label GeoTIFF for metrics")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--skip-checkpoint-hash", action="store_true", help="Skip SHA-256 verification only if you intentionally use a different checkpoint")
    parser.add_argument("--allow-slow-cpu", action="store_true", help="Allow CPU inference with the slow reference selective-scan backend")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    label_path = args.label.expanduser().resolve() if args.label else None

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_hash = verify_checkpoint(checkpoint_path, args.skip_checkpoint_hash)
    model = load_model(checkpoint_path, device, args.allow_slow_cpu)
    run_inference(
        model=model,
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        output_dir=output_dir,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        label_path=label_path,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nDEMO FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
