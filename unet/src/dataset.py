from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# ============================================================
# FloodPlanet configuration
# ============================================================

PATCH_HEIGHT = 300
PATCH_WIDTH = 300
PATCH_STRIDE = 150

IGNORE_INDEX = -1


# ============================================================
# Crop generation
# ============================================================

def get_exact_crop_slices(
    height: int,
    width: int,
    crop_height: int = PATCH_HEIGHT,
    crop_width: int = PATCH_WIDTH,
    stride: int = PATCH_STRIDE,
):
    """
    Generate crop coordinates using the 'exact' crop behavior
    used by the official FloodPlanet code.

    Each crop is:

        (row_start, column_start, height, width)

    Full-size crops are created first.

    If the image dimensions do not divide evenly into the
    requested patch size/stride, additional edge crops are
    created so that the entire image is covered.
    """

    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")

    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("Crop dimensions must be positive.")

    if stride <= 0:
        raise ValueError("Stride must be positive.")

    if stride > height or stride > width:
        raise ValueError(
            "Stride cannot be larger than image dimensions."
        )

    crops = []

    # --------------------------------------------------------
    # Find number of full crops along height
    # --------------------------------------------------------

    num_h_crops = 0

    while True:
        if (
            num_h_crops * stride + crop_height
            > height
        ):
            break

        num_h_crops += 1

    # --------------------------------------------------------
    # Find number of full crops along width
    # --------------------------------------------------------

    num_w_crops = 0

    while True:
        if (
            num_w_crops * stride + crop_width
            > width
        ):
            break

        num_w_crops += 1

    # --------------------------------------------------------
    # Full-size crops
    # --------------------------------------------------------

    for i in range(num_h_crops):
        for j in range(num_w_crops):

            crops.append(
                (
                    i * stride,
                    j * stride,
                    crop_height,
                    crop_width,
                )
            )

    # --------------------------------------------------------
    # Remaining areas
    # --------------------------------------------------------

    remaining_height = (
        height - num_h_crops * stride
    )

    remaining_width = (
        width - num_w_crops * stride
    )

    # Right edge
    if remaining_width != 0:

        for i in range(num_h_crops):

            crops.append(
                (
                    i * stride,
                    num_w_crops * stride,
                    crop_height,
                    remaining_width,
                )
            )

    # Bottom edge
    if remaining_height != 0:

        for j in range(num_w_crops):

            crops.append(
                (
                    num_h_crops * stride,
                    j * stride,
                    remaining_height,
                    crop_width,
                )
            )

    # Bottom-right corner
    if (
        remaining_height != 0
        and remaining_width != 0
    ):

        crops.append(
            (
                num_h_crops * stride,
                num_w_crops * stride,
                remaining_height,
                remaining_width,
            )
        )

    return crops


# ============================================================
# FloodPlanet Dataset
# ============================================================

class FloodPlanetDataset(Dataset):
    """
    FloodPlanet PlanetScope semantic-segmentation Dataset.

    One Dataset item corresponds to one 300x300 patch.

    Image:
        [4, 300, 300]

    Label:
        [300, 300]

    Label values:
        -1 = ignore
         0 = non-flood
         1 = flood
    """

    def __init__(
        self,
        root_dir,
        events,
        patch_height=PATCH_HEIGHT,
        patch_width=PATCH_WIDTH,
        stride=PATCH_STRIDE,
        ignore_index=IGNORE_INDEX,
        augment=False,
    ):

        self.root_dir = Path(root_dir)

        self.events = list(events)

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride = stride

        self.ignore_index = ignore_index

        # True only for training.
        self.augment = augment

        # Patch-level index.
        self.samples = []

        self._build_index()

    # ========================================================
    # Build patch index
    # ========================================================

    def _build_index(self):

        for event in sorted(self.events):

            event_dir = self.root_dir / event

            ps_dir = event_dir / "PS"
            label_dir = event_dir / "labels"

            if not ps_dir.exists():
                raise FileNotFoundError(
                    f"PlanetScope directory not found:\n{ps_dir}"
                )

            if not label_dir.exists():
                raise FileNotFoundError(
                    f"Label directory not found:\n{label_dir}"
                )

            image_files = sorted(
                ps_dir.glob("*.tif")
            )

            if len(image_files) == 0:
                raise ValueError(
                    f"No PlanetScope TIFF files found in:\n{ps_dir}"
                )

            for image_path in image_files:

                label_path = (
                    label_dir / image_path.name
                )

                if not label_path.exists():
                    raise FileNotFoundError(
                        f"Label not found for:\n{image_path}"
                    )

                # Label dimensions determine crop layout.
                with rasterio.open(label_path) as src:
                    height = src.height
                    width = src.width

                crop_slices = get_exact_crop_slices(
                    height=height,
                    width=width,
                    crop_height=self.patch_height,
                    crop_width=self.patch_width,
                    stride=self.stride,
                )

                for (
                    h0,
                    w0,
                    crop_h,
                    crop_w,
                ) in crop_slices:

                    self.samples.append(
                        {
                            "event": event,
                            "image_path": image_path,
                            "label_path": label_path,
                            "h0": h0,
                            "w0": w0,
                            "crop_height": crop_h,
                            "crop_width": crop_w,
                            "original_height": height,
                            "original_width": width,
                        }
                    )

        print("Dataset index created.")
        print("Events:", len(self.events))
        print(
            "Patch samples:",
            len(self.samples)
        )

    # ========================================================
    # Dataset length
    # ========================================================

    def __len__(self):
        return len(self.samples)

    # ========================================================
    # Load PlanetScope image
    # ========================================================

    def _load_planetscope(
        self,
        image_path,
    ):
        """
        Load one PlanetScope TIFF.

        Expected final layout:

            [4, H, W]

        The official FloodPlanet PS loader scales uint16
        PlanetScope data by 2^16.
        """

        with rasterio.open(image_path) as src:
            image = src.read()

        if image.ndim != 3:
            raise ValueError(
                f"Unexpected PS dimensions: {image.shape}"
            )

        # Rasterio gives us [C, H, W].
        if image.shape[0] == 4:

            pass

        # Defensive support in case a file is HWC.
        elif image.shape[-1] == 4:

            image = image.transpose(
                2,
                0,
                1,
            )

        else:

            raise ValueError(
                "Expected 4 PlanetScope bands, "
                f"got shape {image.shape}"
            )

        original_dtype = image.dtype

        image = image.astype(
            np.float32,
            copy=False,
        )

        # Match the official PS preprocessing:
        # uint16 → divide by 2^16.
        if original_dtype == np.uint16:

            image = image / (2 ** 16)

        image = np.clip(
            image,
            0.0,
            1.0,
        )

        return image

    # ========================================================
    # Load label
    # ========================================================

    def _load_label(
        self,
        label_path,
    ):
        """
        Convert original FloodPlanet labels:

            0 = no data
            1 = no flood
            2 = flood

        into:

           -1 = ignore
            0 = non-flood
            1 = flood
        """

        with rasterio.open(label_path) as src:
            label = src.read(1)

        if label.ndim != 2:
            raise ValueError(
                f"Unexpected label dimensions: {label.shape}"
            )

        binary_label = np.zeros(
            label.shape,
            dtype=np.int64,
        )

        # Original 2 → flood class 1.
        binary_label[label == 2] = 1

        # Original 1 → non-flood class 0.
        binary_label[label == 1] = 0

        # Original 0 → ignored.
        binary_label[
            label == 0
        ] = self.ignore_index

        return binary_label

    # ========================================================
    # Crop helper
    # ========================================================

    @staticmethod
    def _crop_array(
        array,
        h0,
        w0,
        crop_height,
        crop_width,
    ):
        """
        Crop either:

            [H, W]

        or:

            [C, H, W]
        """

        if array.ndim == 2:

            return array[
                h0:h0 + crop_height,
                w0:w0 + crop_width,
            ]

        if array.ndim == 3:

            return array[
                :,
                h0:h0 + crop_height,
                w0:w0 + crop_width,
            ]

        raise ValueError(
            f"Unsupported array shape: {array.shape}"
        )

    # ========================================================
    # Pad image
    # ========================================================

    def _pad_image(
        self,
        image,
    ):
        """
        Pad an edge image crop to 300x300.

        Image padding value = 0.
        """

        padded = np.zeros(
            (
                image.shape[0],
                self.patch_height,
                self.patch_width,
            ),
            dtype=np.float32,
        )

        padded[
            :,
            :image.shape[1],
            :image.shape[2],
        ] = image

        return padded

    # ========================================================
    # Pad label
    # ========================================================

    def _pad_label(
        self,
        label,
    ):
        """
        Pad an edge label crop to 300x300.

        Padding uses IGNORE_INDEX because padded pixels
        are not real ground-truth pixels.
        """

        padded = np.full(
            (
                self.patch_height,
                self.patch_width,
            ),
            self.ignore_index,
            dtype=np.int64,
        )

        padded[
            :label.shape[0],
            :label.shape[1],
        ] = label

        return padded

    # ========================================================
    # Augmentation
    # ========================================================

    def _apply_augmentations(self, image, label):
        """
        Apply the same random geometric transformation
        to the image and its corresponding label.

        Image shape:
            [C, H, W]

        Label shape:
            [H, W]
        """

        # ----------------------------------------------------
        # Horizontal flip
        # ----------------------------------------------------

        if np.random.rand() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # ----------------------------------------------------
        # Vertical flip
        # ----------------------------------------------------

        if np.random.rand() < 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # ----------------------------------------------------
        # Random rotation
        # ----------------------------------------------------

        if np.random.rand() < 0.5:

            angle = float(
                np.random.uniform(0, 360)
            )

            # =================================================
            # IMAGE
            #
            # [C, H, W]
            #      ↓
            # [1, C, H, W]
            #      ↓ rotate
            # [1, C, H, W]
            #      ↓
            # [C, H, W]
            # =================================================

            image = image.unsqueeze(0)

            image = TF.rotate(
                image,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0.0,
            )

            image = image.squeeze(0)

            # =================================================
            # LABEL
            #
            # [H, W]
            #      ↓
            # [1, 1, H, W]
            #      ↓ rotate
            # [1, 1, H, W]
            #      ↓
            # [H, W]
            # =================================================

            label = label.unsqueeze(0).unsqueeze(0)

            label = TF.rotate(
                label.float(),
                angle=angle,
                interpolation=InterpolationMode.NEAREST,
                expand=False,
                fill=float(self.ignore_index),
            )

            label = (
                label
                .squeeze(0)
                .squeeze(0)
                .long()
            )

        return image, label
    # ========================================================
    # Get one sample
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[index]

        # ----------------------------------------------------
        # Load full PlanetScope image
        # ----------------------------------------------------

        image = self._load_planetscope(
            sample["image_path"]
        )

        # ----------------------------------------------------
        # Load full label
        # ----------------------------------------------------

        label = self._load_label(
            sample["label_path"]
        )

        # ----------------------------------------------------
        # Crop image
        # ----------------------------------------------------

        image = self._crop_array(
            image,
            sample["h0"],
            sample["w0"],
            sample["crop_height"],
            sample["crop_width"],
        )

        # ----------------------------------------------------
        # Crop matching label
        # ----------------------------------------------------

        label = self._crop_array(
            label,
            sample["h0"],
            sample["w0"],
            sample["crop_height"],
            sample["crop_width"],
        )

        # ----------------------------------------------------
        # Pad boundary crops
        # ----------------------------------------------------

        image = self._pad_image(image)

        label = self._pad_label(label)

        # ----------------------------------------------------
        # NumPy → PyTorch
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        label = torch.from_numpy(
            label
        ).long()

        # ----------------------------------------------------
        # Training augmentation only
        # ----------------------------------------------------

        if self.augment:

            image, label = (
                self._apply_augmentations(
                    image,
                    label,
                )
            )

        return image, label