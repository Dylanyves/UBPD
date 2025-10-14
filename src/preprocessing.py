import random
import torch
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from typing import Tuple


class PairedTransform:
    """Apply resize and optional random augments consistently to (image, mask)."""

    def __init__(self, size: Tuple[int, int] | int, aug: bool = True):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.aug = aug

    def __call__(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # deterministic resize
        img = img.resize(self.size, resample=Image.BILINEAR)
        mask = mask.resize(self.size, resample=Image.NEAREST)

        if self.aug:
            # RandomApply(RandomRotation(±10°), p=0.5)
            if random.random() < 0.5:
                angle = random.uniform(-10.0, 10.0)
                img = TF.rotate(
                    img,
                    angle=angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=0,
                )
                mask = TF.rotate(
                    mask,
                    angle=angle,
                    interpolation=TF.InterpolationMode.NEAREST,
                    fill=0,
                )

            # RandomApply(RandomAffine(translate up to 12.5%), p=0.5)
            if random.random() < 0.5:
                max_dx = 0.125 * self.size[0]
                max_dy = 0.125 * self.size[1]
                tx = int(random.uniform(-max_dx, max_dx))
                ty = int(random.uniform(-max_dy, max_dy))
                img = TF.affine(
                    img,
                    angle=0.0,
                    translate=(tx, ty),
                    scale=1.0,
                    shear=(0.0, 0.0),
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=0,
                )
                mask = TF.affine(
                    mask,
                    angle=0.0,
                    translate=(tx, ty),
                    scale=1.0,
                    shear=(0.0, 0.0),
                    interpolation=TF.InterpolationMode.NEAREST,
                    fill=0,
                )

        img_t = TF.to_tensor(img)  # [1,H,W]
        mask_t = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()  # [H,W]
        return img_t, mask_t


def _make_paired_transform(size, aug: bool = True) -> PairedTransform:
    """Return a PairedTransform for (image, mask)."""
    return PairedTransform(size=size, aug=aug)
