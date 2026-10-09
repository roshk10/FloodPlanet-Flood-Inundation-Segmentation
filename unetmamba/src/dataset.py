from pathlib import Path
import random

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


# ================================================================
# FLOODPLANET DATASET DEFAULTS
# ================================================================

DEFAULT_PATCH_SIZE = 512
DEFAULT_STRIDE = 256
DEFAULT_IGNORE_INDEX = -1


# ================================================================
# BUILD FLOODPLANET RECORDS
# ================================================================

def build_records(
    dataset_root,
    expected_records=366,
):
    """
    Build PlanetScope image / label pairs from FloodPlanet.

    Expected structure:

        FloodPlanet/
            Bangladesh/
                PS/
                labels/
            Bolivia/
                PS/
                labels/
            ...
    """

    dataset_root = Path(dataset_root)

    records = []

    for event_dir in sorted(
        dataset_root.iterdir()
    ):

        if not event_dir.is_dir():
            continue

        ps_dir = event_dir / "PS"
        label_dir = event_dir / "labels"

        if not ps_dir.exists():
            continue

        if not label_dir.exists():
            continue

        for ps_path in sorted(
            ps_dir.glob("*.tif")
        ):

            chip_id = ps_path.stem

            label_path = (
                label_dir
                / f"{chip_id}.tif"
            )

            if label_path.exists():

                records.append(
                    {
                        "event": event_dir.name,
                        "chip_id": chip_id,
                        "image_path": str(
                            ps_path.relative_to(
                                dataset_root
                            )
                        ),
                        "label_path": str(
                            label_path.relative_to(
                                dataset_root
                            )
                        ),
                    }
                )

    event_names = sorted(
        {
            record["event"]
            for record in records
        }
    )

    if expected_records is not None:

        if len(records) != expected_records:

            raise RuntimeError(
                f"Expected "
                f"{expected_records} records, "
                f"found {len(records)}"
            )

    if len(event_names) != 19:

        raise RuntimeError(
            "Expected 19 FloodPlanet events, "
            f"found {len(event_names)}"
        )

    return records, event_names


# ================================================================
# BUILD LOEO FOLDS
# ================================================================

def build_loeo_folds(
    records,
    event_names,
):
    """
    Build the FloodPlanet leave-one-event-out structure.

    For each event:

        train = every other event
        test  = the held-out event
    """

    loeo_folds = {}

    for holdout_event in event_names:

        loeo_folds[holdout_event] = {

            "train": [
                record
                for record in records
                if record["event"] != holdout_event
            ],

            "test": [
                record
                for record in records
                if record["event"] == holdout_event
            ],
        }

    return loeo_folds


# ================================================================
# FLOODPLANET DATASET
# ================================================================

class FloodPlanetDataset(Dataset):

    def __init__(
        self,
        records,
        dataset_root,
        patch_size=DEFAULT_PATCH_SIZE,
        stride=DEFAULT_STRIDE,
        ignore_index=DEFAULT_IGNORE_INDEX,
        augment=False,
    ):

        self.records = records

        self.dataset_root = Path(
            dataset_root
        )

        self.patch_size = patch_size

        self.stride = stride

        self.ignore_index = ignore_index

        self.augment = augment

        self.samples = []

        # --------------------------------------------------------
        # Build patch index
        # --------------------------------------------------------

        for record in self.records:

            image_path = (
                self.dataset_root
                / record["image_path"]
            )

            label_path = (
                self.dataset_root
                / record["label_path"]
            )

            with rasterio.open(
                label_path
            ) as src:

                height = src.height
                width = src.width

            crop_boxes = (
                self.get_exact_crop_boxes(
                    height,
                    width,
                )
            )

            for crop_box in crop_boxes:
                self.samples.append(
                    {
                        "event": record["event"],
                        "chip_id": record["chip_id"],
                        "image_path": image_path,
                        "label_path": label_path,
                        "crop_box": crop_box,
                    }
                )

        self._image_cache = {}
        self._label_cache = {}

    # ============================================================
    # PATCH GENERATION
    # ============================================================

    def get_exact_crop_boxes(
        self,
        height,
        width,
    ):

        crop_boxes = []

        # --------------------------------------------------------
        # Number of complete positions
        # --------------------------------------------------------

        num_h_crops = 0

        while (
            num_h_crops * self.stride
            + self.patch_size
            <= height
        ):

            num_h_crops += 1

        num_w_crops = 0

        while (
            num_w_crops * self.stride
            + self.patch_size
            <= width
        ):

            num_w_crops += 1

        # --------------------------------------------------------
        # Full patches
        # --------------------------------------------------------

        for i in range(
            num_h_crops
        ):

            for j in range(
                num_w_crops
            ):

                crop_boxes.append(
                    (
                        i * self.stride,
                        j * self.stride,
                        self.patch_size,
                        self.patch_size,
                    )
                )

        # --------------------------------------------------------
        # Right-edge patches
        # --------------------------------------------------------

        remaining_width = (
            width
            - num_w_crops * self.stride
        )

        if remaining_width != 0:

            for i in range(
                num_h_crops
            ):

                crop_boxes.append(
                    (
                        i * self.stride,
                        num_w_crops * self.stride,
                        self.patch_size,
                        remaining_width,
                    )
                )

        # --------------------------------------------------------
        # Bottom-edge patches
        # --------------------------------------------------------

        remaining_height = (
            height
            - num_h_crops * self.stride
        )

        if remaining_height != 0:

            for j in range(
                num_w_crops
            ):

                crop_boxes.append(
                    (
                        num_h_crops * self.stride,
                        j * self.stride,
                        remaining_height,
                        self.patch_size,
                    )
                )

        # --------------------------------------------------------
        # Bottom-right corner
        # --------------------------------------------------------

        if (
            remaining_height != 0
            and remaining_width != 0
        ):

            crop_boxes.append(
                (
                    num_h_crops * self.stride,
                    num_w_crops * self.stride,
                    remaining_height,
                    remaining_width,
                )
            )

        return crop_boxes

    # ============================================================
    # IMAGE LOADING
    # ============================================================

    def load_image(
        self,
        path,
    ):

        with rasterio.open(path) as src:

            image = src.read()

        # FloodPlanet PlanetScope input
        # must contain 4 bands.

        if image.shape[0] != 4:

            raise ValueError(
                "Expected 4 PlanetScope bands, "
                f"but found shape {image.shape}"
            )

        image = image.astype(
            np.float32
        )

        # Original FloodPlanet preprocessing
        image = image / 65536.0

        image = np.clip(
            image,
            0.0,
            1.0,
        )

        return image

    # ============================================================
    # LABEL LOADING
    # ============================================================

    def load_label(
        self,
        path,
    ):

        with rasterio.open(path) as src:

            label = src.read(1)

        unique_values = set(
            np.unique(label).tolist()
        )

        allowed_values = {
            0,
            1,
            2,
        }

        if not unique_values.issubset(
            allowed_values
        ):

            raise ValueError(
                "Unexpected label values: "
                f"{unique_values}"
            )

        # Ignore by default
        binary_label = np.full(
            label.shape,
            self.ignore_index,
            dtype=np.int64,
        )

        # FloodPlanet label mapping
        #
        # original 1 -> background = 0
        # original 2 -> flood      = 1

        binary_label[
            label == 1
        ] = 0

        binary_label[
            label == 2
        ] = 1

        return binary_label

    # ============================================================
    # IMAGE PADDING
    # ============================================================

    def pad_image(
        self,
        image,
        target_height,
        target_width,
    ):

        channels, height, width = (
            image.shape
        )

        padded = np.zeros(
            (
                channels,
                target_height,
                target_width,
            ),
            dtype=image.dtype,
        )

        padded[
            :,
            :height,
            :width,
        ] = image

        return padded

    # ============================================================
    # LABEL PADDING
    # ============================================================

    def pad_label(
        self,
        label,
        target_height,
        target_width,
    ):

        height, width = (
            label.shape
        )

        padded = np.full(
            (
                target_height,
                target_width,
            ),
            self.ignore_index,
            dtype=label.dtype,
        )

        padded[
            :height,
            :width,
        ] = label

        return padded

    # ============================================================
    # CROP PATCH
    # ============================================================

    def crop_patch(
        self,
        image,
        label,
        crop_box,
    ):

        top, left, height, width = (
            crop_box
        )

        image_crop = image[
            :,
            top:top + height,
            left:left + width,
        ]

        label_crop = label[
            top:top + height,
            left:left + width,
        ]

        image_crop = self.pad_image(
            image_crop,
            self.patch_size,
            self.patch_size,
        )

        label_crop = self.pad_label(
            label_crop,
            self.patch_size,
            self.patch_size,
        )

        return (
            image_crop,
            label_crop,
        )

    # ============================================================
    # DATA AUGMENTATION
    # ============================================================

    def augment_pair(
        self,
        image,
        label,
    ):

        # --------------------------------------------------------
        # Horizontal flip
        # --------------------------------------------------------

        if random.random() < 0.5:

            image = np.flip(
                image,
                axis=2,
            ).copy()

            label = np.flip(
                label,
                axis=1,
            ).copy()

        # --------------------------------------------------------
        # Vertical flip
        # --------------------------------------------------------

        if random.random() < 0.5:

            image = np.flip(
                image,
                axis=1,
            ).copy()

            label = np.flip(
                label,
                axis=0,
            ).copy()

        # --------------------------------------------------------
        # Fast Orthogonal Rotation (90, 180, 270 deg)
        # --------------------------------------------------------

        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            label = np.rot90(label, k=k, axes=(0, 1)).copy()

        return (
            image,
            label,
        )

    # ============================================================
    # DATASET LENGTH
    # ============================================================

    def __len__(self):

        return len(
            self.samples
        )

    # ============================================================
    # GET ONE PATCH
    # ============================================================

    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[
            index
        ]

        img_key = str(sample["image_path"])
        if img_key not in self._image_cache:
            self._image_cache[img_key] = self.load_image(sample["image_path"])
        image = self._image_cache[img_key]

        lbl_key = str(sample["label_path"])
        if lbl_key not in self._label_cache:
            self._label_cache[lbl_key] = self.load_label(sample["label_path"])
        label = self._label_cache[lbl_key]

        # --------------------------------------------------------
        # Crop + pad
        # --------------------------------------------------------

        image, label = (
            self.crop_patch(
                image,
                label,
                sample["crop_box"],
            )
        )

        # --------------------------------------------------------
        # Augment
        # --------------------------------------------------------

        if self.augment:

            image, label = (
                self.augment_pair(
                    image,
                    label,
                )
            )

        # --------------------------------------------------------
        # Ensure contiguous arrays
        # --------------------------------------------------------

        image = np.ascontiguousarray(
            image
        )

        label = np.ascontiguousarray(
            label
        )

        # --------------------------------------------------------
        # Convert to PyTorch tensors
        # --------------------------------------------------------

        image_tensor = (
            torch.from_numpy(
                image
            ).float()
        )

        label_tensor = (
            torch.from_numpy(
                label
            ).long()
        )

        return (
            image_tensor,
            label_tensor,
        )