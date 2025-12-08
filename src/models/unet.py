import torch
import torch.nn as nn

from typing import Literal
from torchvision import models


class DoubleConv(nn.Module):
    """(Conv→BN/GN→ReLU)×2."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: Literal["bn", "gn", "none"] = "bn",
        groups: int = 8,
        p: float = 0.0,
    ):
        super().__init__()

        def N(c):
            if norm == "none":
                return nn.Identity()
            if norm == "bn":
                return nn.BatchNorm2d(c)
            return nn.GroupNorm(groups, c)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            N(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p) if p > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            N(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # noqa: D401
        return self.block(x)


class Down(nn.Module):
    """MaxPool then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, **dc_kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, **dc_kwargs)
        )

    def forward(self, x):  # noqa: D401
        return self.block(x)


class Up(nn.Module):
    """Upsample, concat skip, DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True, **dc_kwargs):
        super().__init__()
        self.up = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            if bilinear
            else nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        )
        self.conv = DoubleConv(in_ch, out_ch, **dc_kwargs)

    def forward(self, x, skip):  # noqa: D401
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1×1 conv to logits."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):  # noqa: D401
        return self.conv(x)


class UNet(nn.Module):
    """U-Net with optional VGG19_BN encoder."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        bilinear: bool = True,
        norm: Literal["bn", "gn", "none"] = "bn",
        groups: int = 8,
        dropout: float = 0.0,
        vgg_backbone: bool = False,  # <-- new flag
    ):
        super().__init__()
        self.vgg_backbone = vgg_backbone
        self.bilinear = bilinear

        if vgg_backbone:
            # ---- VGG19_BN encoder ----
            try:
                vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
            except Exception:
                vgg = models.vgg19_bn(pretrained=True)
            feats = list(vgg.features.children())
            if in_channels != 3 and isinstance(feats[0], nn.Conv2d):
                old = feats[0]
                new = nn.Conv2d(
                    in_channels,
                    old.out_channels,
                    kernel_size=old.kernel_size,
                    stride=old.stride,
                    padding=old.padding,
                    bias=False,
                )
                with torch.no_grad():
                    if in_channels == 1:
                        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                    else:
                        nn.init.kaiming_normal_(new.weight, nonlinearity="relu")
                feats[0] = new
            self.encoder = nn.Sequential(*feats)

            # Decoder channels for VGG19_BN (skip channels: 64,128,256,512; bottom: 512)
            dc = dict(norm=norm, groups=groups, p=dropout)
            self.up1 = Up(512 + 512, 512 if bilinear else 512, bilinear=bilinear, **dc)
            self.up2 = Up(512 + 256, 256 if bilinear else 256, bilinear=bilinear, **dc)
            self.up3 = Up(256 + 128, 128 if bilinear else 128, bilinear=bilinear, **dc)
            self.up4 = Up(128 + 64, 64, bilinear=bilinear, **dc)
            self.outc = OutConv(64, num_classes)

        else:
            # ---- Classic tiny encoder ----
            c1, c2, c3, c4, c5 = (
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
                base_channels * 16,
            )
            dc = dict(norm=norm, groups=groups, p=dropout)
            self.inc = DoubleConv(in_channels, c1, **dc)
            self.down1 = Down(c1, c2, **dc)
            self.down2 = Down(c2, c3, **dc)
            self.down3 = Down(c3, c4, **dc)
            self.down4 = Down(c4, c5 // (2 if bilinear else 1), **dc)

            self.up1 = Up(c5, c4 // (2 if bilinear else 1), bilinear=bilinear, **dc)
            self.up2 = Up(c4, c3 // (2 if bilinear else 1), bilinear=bilinear, **dc)
            self.up3 = Up(c3, c2 // (2 if bilinear else 1), bilinear=bilinear, **dc)
            self.up4 = Up(c2, c1, bilinear=bilinear, **dc)
            self.outc = OutConv(c1, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):  # noqa: D401
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _encode_vgg(self, x):
        """Run VGG and collect pre-pool features."""
        skips = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                skips.append(x)  # pre-pool
                x = layer(x)
            else:
                x = layer(x)
        # skips: [64,128,256,512,512-prepool]; bottom: x after last pool (512)
        s1, s2, s3, s4, s5 = (
            skips  # s5 is pre-final-pool 512, but VGG has two 512 blocks; use s4 as 512 skip
        )
        bottom = x
        return bottom, (s1, s2, s3, s4)

    def forward(self, x):  # noqa: D401
        if self.vgg_backbone:
            bottom, (s1, s2, s3, s4) = self._encode_vgg(x)
            x = self.up1(bottom, s4)  # 512 + 512 → 512
            x = self.up2(x, s3)  # 512 + 256 → 256
            x = self.up3(x, s2)  # 256 + 128 → 128
            x = self.up4(x, s1)  # 128 + 64  → 64
            return self.outc(x)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            return self.outc(x)
