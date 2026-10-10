from pathlib import Path
import argparse
import os
import sys

import numpy as np
import rasterio
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SOURCE = PROJECT_ROOT / "unet" / "src"

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "unet" / "checkpoints" / "best_val_iou.pth"
)

PATCH_SIZE = 300
STRIDE = 150
NUM_CHANNELS = 4
NUM_CLASSES = 2


def read_planetscope_image(image_path):
    """Read and normalize a four-band PlanetScope TIFF."""
    with rasterio.open(image_path) as src:
        raw_image = src.read()
        profile = src.profile.copy()

    if raw_image.ndim != 3 or raw_image.shape[0] != NUM_CHANNELS:
        raise ValueError(
            f"Expected four-band [C,H,W] image; got {raw_image.shape}"
        )

    original_dtype = raw_image.dtype
    image = raw_image.astype(np.float32, copy=False)

    # Match the normalization used by the FloodPlanet dataset loader.
    if original_dtype == np.uint16:
        image = image / (2 ** 16)

    image = np.clip(image, 0.0, 1.0)

    if not np.isfinite(image).all():
        raise ValueError("Input image contains NaN or infinite values.")

    return raw_image, image, profile


def get_axis_crops(length, patch_size=PATCH_SIZE, stride=STRIDE):
    """
    Generate patch starts using the crop layout of the project's
    FloodPlanet Dataset implementation. Edge patches can be smaller.
    """
    if length <= 0:
        raise ValueError("Image dimensions must be positive.")

    number_of_full_crops = 0

    while number_of_full_crops * stride + patch_size <= length:
        number_of_full_crops += 1

    crops = [
        (i * stride, patch_size)
        for i in range(number_of_full_crops)
    ]

    remaining = length - number_of_full_crops * stride

    if remaining > 0:
        crops.append((number_of_full_crops * stride, remaining))

    return crops


def load_model(checkpoint_path):
    """Load the trained four-channel, two-class U-Net."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not (MODEL_SOURCE / "model.py").is_file():
        raise FileNotFoundError(
            f"Model source not found: {MODEL_SOURCE / 'model.py'}"
        )

    sys.path.insert(0, str(MODEL_SOURCE))
    from model import UNet

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    model = UNet(
        n_channels=NUM_CHANNELS,
        n_classes=NUM_CLASSES,
        bilinear=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    print("Checkpoint:", checkpoint_path)
    print("Selected epoch:", checkpoint.get("epoch", "not recorded"))
    print(
        "Best validation IoU:",
        checkpoint.get("best_val_iou", "not recorded"),
    )
    print("Device: CPU")

    return model


def predict_full_scene(model, image):
    """
    Predict a complete image using overlapping 300 x 300 patches.

    Overlapping flood probabilities are averaged before thresholding
    at 0.5 to produce the binary mask.
    """
    _, height, width = image.shape

    row_crops = get_axis_crops(height)
    col_crops = get_axis_crops(width)

    probability_sum = np.zeros((height, width), dtype=np.float64)
    overlap_count = np.zeros((height, width), dtype=np.uint16)

    total_patches = len(row_crops) * len(col_crops)

    with torch.inference_mode():
        for row_start, crop_height in row_crops:
            for col_start, crop_width in col_crops:

                patch = image[
                    :,
                    row_start:row_start + crop_height,
                    col_start:col_start + crop_width,
                ]

                padded = np.zeros(
                    (NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE),
                    dtype=np.float32,
                )

                padded[:, :crop_height, :crop_width] = patch

                tensor = torch.from_numpy(
                    np.ascontiguousarray(padded)
                ).unsqueeze(0)

                logits = model(tensor)

                flood_probability = torch.softmax(
                    logits, dim=1
                )[0, 1].numpy()

                probability_sum[
                    row_start:row_start + crop_height,
                    col_start:col_start + crop_width,
                ] += flood_probability[:crop_height, :crop_width]

                overlap_count[
                    row_start:row_start + crop_height,
                    col_start:col_start + crop_width,
                ] += 1

    if np.any(overlap_count == 0):
        raise RuntimeError(
            "Some image pixels did not receive a prediction."
        )

    mean_probability = probability_sum / overlap_count
    flood_mask = mean_probability >= 0.5

    print("Image dimensions:", f"{height} x {width}")
    print("Patches evaluated:", total_patches)
    print(
        "Predicted flood coverage:",
        f"{100 * flood_mask.mean():.2f}%",
    )

    return flood_mask


def display_rgb(raw_image):
    """
    Create an RGB display composite from the first three visible
    PlanetScope bands. All four bands are used for model inference.
    """
    rgb = np.stack(
        [raw_image[2], raw_image[1], raw_image[0]],
        axis=-1,
    ).astype(np.float32)

    if not np.isfinite(rgb).all():
        rgb = np.nan_to_num(rgb)

    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    if high <= low:
        return np.zeros_like(rgb)

    return np.clip((rgb - low) / (high - low), 0, 1)


def calculate_metrics(prediction, raw_label):
    """Calculate binary flood metrics, ignoring raw label value 0."""
    if raw_label.shape != prediction.shape:
        raise ValueError(
            f"Label dimensions {raw_label.shape} do not match "
            f"prediction dimensions {prediction.shape}."
        )

    valid = (raw_label == 1) | (raw_label == 2)
    truth = raw_label == 2

    pred = prediction[valid]
    target = truth[valid]

    tp = int(np.sum(pred & target))
    fp = int(np.sum(pred & ~target))
    fn = int(np.sum(~pred & target))
    tn = int(np.sum(~pred & ~target))

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    return {
        "valid_pixels": int(valid.sum()),
        "IoU": ratio(tp, tp + fp + fn),
        "F1": ratio(2 * tp, 2 * tp + fp + fn),
        "Precision": ratio(tp, tp + fp),
        "Recall": ratio(tp, tp + fn),
        "Accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "Specificity": ratio(tn, tn + fp),
    }


def save_outputs(
    raw_image,
    prediction,
    profile,
    image_path,
    output_dir,
    raw_label=None,
):
    """Save a georeferenced binary mask and visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    mask_tif = output_dir / f"{stem}_predicted_mask.tif"
    mask_png = output_dir / f"{stem}_predicted_mask.png"
    visual_png = output_dir / f"{stem}_visualization.png"

    # GeoTIFF encoding: 0 = non-flood, 1 = flood.
    output_profile = profile.copy()
    output_profile.update(
        count=1,
        dtype="uint8",
        nodata=None,
        compress="deflate",
    )

    with rasterio.open(mask_tif, "w", **output_profile) as dst:
        dst.write(prediction.astype(np.uint8), 1)

    plt.imsave(
        mask_png,
        prediction.astype(np.uint8) * 255,
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    rgb = display_rgb(raw_image)

    if raw_label is not None:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        axes = axes.ravel()

        axes[0].imshow(rgb)
        axes[0].set_title("PlanetScope RGB composite")

        axes[1].imshow(
            raw_label == 2,
            cmap="Reds",
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Ground-truth flood mask")

        axes[2].imshow(
            prediction,
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        axes[2].set_title("Predicted flood mask")

        axes[3].imshow(rgb)
        overlay = np.ma.masked_where(~prediction, prediction)
        axes[3].imshow(
            overlay,
            cmap="Greens",
            vmin=0,
            vmax=1,
            alpha=0.55,
        )
        axes[3].set_title("Predicted flood overlay")

    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        axes[0].imshow(rgb)
        axes[0].set_title("PlanetScope RGB composite")

        axes[1].imshow(
            prediction,
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Predicted flood mask")

        axes[2].imshow(rgb)
        overlay = np.ma.masked_where(~prediction, prediction)
        axes[2].imshow(
            overlay,
            cmap="Greens",
            vmin=0,
            vmax=1,
            alpha=0.55,
        )
        axes[2].set_title("Predicted flood overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(f"FloodPlanet U-Net: {image_path.name}")
    fig.tight_layout()
    fig.savefig(visual_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved outputs:")
    print("GeoTIFF mask:", mask_tif)
    print("PNG mask:", mask_png)
    print("Visualization:", visual_png)


def main():
    parser = argparse.ArgumentParser(
        description="FloodPlanet U-Net flood-inundation inference."
    )

    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Four-band PlanetScope TIFF.",
    )

    parser.add_argument(
        "--label",
        type=Path,
        default=None,
        help="Optional corresponding raw FloodPlanet label TIFF.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained U-Net checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults beside the input image.",
    )

    args = parser.parse_args()

    image_path = args.image.resolve()
    checkpoint_path = args.checkpoint.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if args.label is not None and not args.label.is_file():
        raise FileNotFoundError(
            f"Label image not found: {args.label}"
        )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else image_path.parent / "unet_predictions"
    )

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    raw_image, normalized_image, profile = read_planetscope_image(
        image_path
    )

    model = load_model(checkpoint_path)

    prediction = predict_full_scene(model, normalized_image)

    raw_label = None

    if args.label is not None:
        with rasterio.open(args.label) as src:
            raw_label = src.read(1)

        print("\nMetrics on supplied labelled image:")
        metrics = calculate_metrics(prediction, raw_label)

        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")

    save_outputs(
        raw_image=raw_image,
        prediction=prediction,
        profile=profile,
        image_path=image_path,
        output_dir=output_dir,
        raw_label=raw_label,
    )

    print("\nINFERENCE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
