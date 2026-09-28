import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    Same basic convolution block as the original U-Net.
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
                bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    MaxPool2d -> DoubleConv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate for filtering encoder skip features.

    g:
        decoder feature / gating signal

    x:
        encoder skip feature

    The gate produces a spatial attention map and
    suppresses less relevant regions in the skip feature.
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l,
                F_int,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                F_int,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1,
                size=x1.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        psi = F.interpolate(
            psi,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        return x * psi


class UpAttention(nn.Module):
    """
    Upsampling + Attention Gate + skip connection + DoubleConv.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.attention = AttentionGate(
            F_g=out_channels,
            F_l=skip_channels,
            F_int=out_channels // 2
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2
            ]
        )

        x2 = self.attention(x1, x2)

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for 2D medical image segmentation.

    Input:
        [B, 1, H, W]

    Output:
        [B, 2, H, W]

    The main difference from the original U-Net is that
    attention gates are inserted before skip features
    are concatenated with decoder features.
    """

    def __init__(
        self,
        n_channels=1,
        n_classes=2
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(
            n_channels,
            64
        )

        self.down1 = Down(
            64,
            128
        )

        self.down2 = Down(
            128,
            256
        )

        self.down3 = Down(
            256,
            512
        )

        self.down4 = Down(
            512,
            1024
        )

        self.up1 = UpAttention(
            in_channels=1024,
            skip_channels=512,
            out_channels=512
        )

        self.up2 = UpAttention(
            in_channels=512,
            skip_channels=256,
            out_channels=256
        )

        self.up3 = UpAttention(
            in_channels=256,
            skip_channels=128,
            out_channels=128
        )

        self.up4 = UpAttention(
            in_channels=128,
            skip_channels=64,
            out_channels=64
        )

        self.outc = nn.Conv2d(
            64,
            n_classes,
            kernel_size=1
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits
