import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Tuple
from torchvision import models


class ConvBlock(nn.Module):
    """(Conv→Norm→ReLU)×2."""

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


class UNetPP(nn.Module):
    """UNet++ (Nested U-Net) with optional VGG19_BN encoder and deep supervision."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        norm: Literal["bn", "gn", "none"] = "bn",
        groups: int = 8,
        dropout: float = 0.0,
        bilinear: bool = True,
        deep_supervision: bool = False,
        vgg_backbone: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.bilinear = bilinear
        self.vgg_backbone = vgg_backbone

        cb = dict(norm=norm, groups=groups, p=dropout)

        if vgg_backbone:
            # --- VGG19_BN encoder ---
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

            # Channel sizes per stage (pre-pool outputs)
            c1, c2, c3, c4, c5 = 64, 128, 256, 512, 512

            # Upsamplers between adjacent levels
            self.up1_0 = self._make_up(c2)
            self.up2_1 = self._make_up(c3)
            self.up3_2 = self._make_up(c4)
            self.up4_3 = self._make_up(c5)

            # Nested decoder blocks
            self.conv0_1 = ConvBlock(c1 + c2, c1, **cb)
            self.conv1_1 = ConvBlock(c2 + c3, c2, **cb)
            self.conv2_1 = ConvBlock(c3 + c4, c3, **cb)
            self.conv3_1 = ConvBlock(c4 + c5, c4, **cb)

            self.conv0_2 = ConvBlock(c1 * 2 + c2, c1, **cb)
            self.conv1_2 = ConvBlock(c2 * 2 + c3, c2, **cb)
            self.conv2_2 = ConvBlock(c3 * 2 + c4, c3, **cb)

            self.conv0_3 = ConvBlock(c1 * 3 + c2, c1, **cb)
            self.conv1_3 = ConvBlock(c2 * 3 + c3, c2, **cb)

            self.conv0_4 = ConvBlock(c1 * 4 + c2, c1, **cb)

        else:
            # --- Lightweight encoder (from scratch) ---
            c1, c2, c3, c4, c5 = (
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
                base_channels * 16,
            )

            self.conv0_0 = ConvBlock(in_channels, c1, **cb)
            self.pool0 = nn.MaxPool2d(2)
            self.conv1_0 = ConvBlock(c1, c2, **cb)
            self.pool1 = nn.MaxPool2d(2)
            self.conv2_0 = ConvBlock(c2, c3, **cb)
            self.pool2 = nn.MaxPool2d(2)
            self.conv3_0 = ConvBlock(c3, c4, **cb)
            self.pool3 = nn.MaxPool2d(2)
            self.conv4_0 = ConvBlock(c4, c5, **cb)

            self.up1_0 = self._make_up(c2)
            self.up2_1 = self._make_up(c3)
            self.up3_2 = self._make_up(c4)
            self.up4_3 = self._make_up(c5)

            self.conv0_1 = ConvBlock(c1 + c2, c1, **cb)
            self.conv1_1 = ConvBlock(c2 + c3, c2, **cb)
            self.conv2_1 = ConvBlock(c3 + c4, c3, **cb)
            self.conv3_1 = ConvBlock(c4 + c5, c4, **cb)

            self.conv0_2 = ConvBlock(c1 * 2 + c2, c1, **cb)
            self.conv1_2 = ConvBlock(c2 * 2 + c3, c2, **cb)
            self.conv2_2 = ConvBlock(c3 * 2 + c4, c3, **cb)

            self.conv0_3 = ConvBlock(c1 * 3 + c2, c1, **cb)
            self.conv1_3 = ConvBlock(c2 * 3 + c3, c2, **cb)

            self.conv0_4 = ConvBlock(c1 * 4 + c2, c1, **cb)

        # Side outputs for deep supervision
        self.out1 = nn.Conv2d(c1, num_classes, 1)
        self.out2 = nn.Conv2d(c1, num_classes, 1)
        self.out3 = nn.Conv2d(c1, num_classes, 1)
        self.out4 = nn.Conv2d(c1, num_classes, 1)

        self.apply(self._init_weights)

    def _make_up(self, channels: int):
        """Create an upsampler for adjacent-level fusion."""
        if self.bilinear:

            def up(x):
                return F.interpolate(
                    x, scale_factor=2, mode="bilinear", align_corners=True
                )

            return up
        else:
            return nn.ConvTranspose2d(channels, channels, 2, stride=2)

    @staticmethod
    def _init_weights(m):
        """Init conv weights."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _encode_vgg(
        self, x
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run VGG and return pre-pool features at 5 scales."""
        feats = []
        cur = x
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                feats.append(cur)  # pre-pool feature
                cur = layer(cur)
            else:
                cur = layer(cur)
        # After the final MaxPool, we don't append again; feats holds 5 pre-pool tensors
        # VGG19_BN has 5 MaxPools → len(feats)=5 with channels [64,128,256,512,512]
        x0_0, x1_0, x2_0, x3_0, x4_0 = feats
        return x0_0, x1_0, x2_0, x3_0, x4_0

    def forward(self, x):  # noqa: D401
        H, W = x.size(2), x.size(3)

        if self.vgg_backbone:
            x0_0, x1_0, x2_0, x3_0, x4_0 = self._encode_vgg(x)
        else:
            x0_0 = self.conv0_0(x)
            x1_0 = self.conv1_0(self.pool0(x0_0))
            x2_0 = self.conv2_0(self.pool1(x1_0))
            x3_0 = self.conv3_0(self.pool2(x2_0))
            x4_0 = self.conv4_0(self.pool3(x3_0))

        # j=1
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(self.up1_0, x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(self.up2_1, x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(self.up3_2, x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(self.up4_3, x4_0)], dim=1))

        # j=2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(self.up1_0, x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(self.up2_1, x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(self.up3_2, x3_1)], dim=1))

        # j=3
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self._up(self.up1_0, x1_2)], dim=1)
        )
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self._up(self.up2_1, x2_2)], dim=1)
        )

        # j=4
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(self.up1_0, x1_3)], dim=1)
        )

        if self.deep_supervision:
            y1 = F.interpolate(
                self.out1(x0_1), size=(H, W), mode="bilinear", align_corners=True
            )
            y2 = F.interpolate(
                self.out2(x0_2), size=(H, W), mode="bilinear", align_corners=True
            )
            y3 = F.interpolate(
                self.out3(x0_3), size=(H, W), mode="bilinear", align_corners=True
            )
            y4 = F.interpolate(
                self.out4(x0_4), size=(H, W), mode="bilinear", align_corners=True
            )
            return [y1, y2, y3, y4]
        else:
            return F.interpolate(
                self.out4(x0_4), size=(H, W), mode="bilinear", align_corners=True
            )

    @staticmethod
    def _up(up, x):  # noqa: D401
        return up(x) if callable(up) else up(x)
