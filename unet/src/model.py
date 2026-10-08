# TODO: implement during Phase 3/4.
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Double Convolution Block
# ============================================================

class DoubleConv(nn.Module):
    """
    Two convolution layers.

    Conv -> BatchNorm -> ReLU
          -> Conv -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# ============================================================
# Encoder / Downsampling Block
# ============================================================

class Down(nn.Module):
    """
    Max pooling followed by DoubleConv.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# ============================================================
# Decoder / Upsampling Block
# ============================================================

class Up(nn.Module):
    """
    Upsampling followed by DoubleConv.

    The implementation uses bilinear interpolation,
    matching the default used by the authors' U-Net code.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=True,
    ):
        super().__init__()

        if bilinear:

            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )

            self.conv = DoubleConv(
                in_channels,
                out_channels,
                in_channels // 2,
            )

        else:

            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )

            self.conv = DoubleConv(
                in_channels,
                out_channels,
            )

    def forward(self, x1, x2):

        # x1 = decoder feature
        # x2 = corresponding encoder feature

        x1 = self.up(x1)

        # Difference between encoder and decoder sizes
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # Pad decoder feature so spatial sizes match
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        # Skip connection
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


# ============================================================
# Output Convolution
# ============================================================

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# FloodPlanet U-Net
# ============================================================

class UNet(nn.Module):
    """
    U-Net used as the FloodPlanet baseline.

    Input:
        [B, 4, 300, 300]

    Output:
        [B, 2, 300, 300]
    """

    def __init__(
        self,
        n_channels=4,
        n_classes=2,
        bilinear=True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(
            n_channels,
            64,
        )

        self.down1 = Down(
            64,
            128,
        )

        self.down2 = Down(
            128,
            256,
        )

        self.down3 = Down(
            256,
            512,
        )

        factor = 2 if bilinear else 1

        self.down4 = Down(
            512,
            1024 // factor,
        )

        # Decoder
        self.up1 = Up(
            1024,
            512 // factor,
            bilinear,
        )

        self.up2 = Up(
            512,
            256 // factor,
            bilinear,
        )

        self.up3 = Up(
            256,
            128 // factor,
            bilinear,
        )

        self.up4 = Up(
            128,
            64,
            bilinear,
        )

        # Two output classes:
        # 0 = non-flood
        # 1 = flood
        self.outc = OutConv(
            64,
            n_classes,
        )

    def forward(self, x):

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder + skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output logits
        logits = self.outc(x)

        return logits