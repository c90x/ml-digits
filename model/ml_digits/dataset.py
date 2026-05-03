from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class DigitDataset(Dataset):
    def __init__(self, *, items: list[tuple[Path, int]], r: int):
        self.r = r
        self.items = items

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _to_tensor(img: Image.Image, r: int) -> torch.Tensor:
        data = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(data, dtype=torch.float32).view(1, r, r)

    def __getitem__(self, idx: int):
        item_path, label = self.items[idx]

        img = Image.open(item_path).convert("L")
        img = ImageOps.invert(img)
        img = img.resize((self.r, self.r))

        return self._to_tensor(img, self.r), label


class DigitTransformedDataset(Dataset):
    SUBTLE_RANGES = dict(translate=(-0.1, 0.1), rotate=(-10, 10), scale=(0.7, 1.0))
    FULL_RANGES = dict(translate=(-0.4, 0.4), rotate=(-45, 45), scale=(0.3, 1.0))
    MAX_COMBINATIONS = 40
    SUBTLE_COUNT = 10

    def __init__(
        self, *, items: list[tuple[Path, int]], r: int, rng: np.random.Generator
    ):
        self.r = r
        self.items = items
        self.rng = rng
        self._combinations = self._build_combinations()

    def _build_combinations(self) -> list[dict]:
        subtle = [self._sample(self.SUBTLE_RANGES) for _ in range(self.SUBTLE_COUNT)]
        full = [
            self._sample(self.FULL_RANGES)
            for _ in range(self.MAX_COMBINATIONS - self.SUBTLE_COUNT)
        ]
        return subtle + full

    def _sample(self, ranges: dict) -> dict:
        return dict(
            tx=self.rng.uniform(*ranges["translate"]),
            ty=self.rng.uniform(*ranges["translate"]),
            angle=self.rng.uniform(*ranges["rotate"]),
            scale=self.rng.uniform(*ranges["scale"]),
        )

    @staticmethod
    def _apply_transform(
        img: Image.Image, tx: float, ty: float, angle: float, scale: float
    ) -> Image.Image:
        """
        Compose scale + rotate + translate into a single affine pass.
        tx/ty are fractions of image size. Rotation is in degrees.
        The digit is transformed around the image center, so no internal clipping occurs.
        """
        w, h = img.size
        cx, cy = w / 2, h / 2

        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        # 2x3 affine matrix: scale + rotate around center, then translate
        # Output pixel (x,y) samples from input at M @ [x, y, 1]
        m00 = cos_a / scale
        m01 = sin_a / scale
        m10 = -sin_a / scale
        m11 = cos_a / scale

        # Keep center fixed, then apply translation (in pixel units)
        dx = tx * w
        dy = ty * h
        m02 = cx - m00 * cx - m01 * cy - dx
        m12 = cy - m10 * cx - m11 * cy - dy

        # Pillow AFFINE transform: data[x,y] = input[m00*x + m01*y + m02, m10*x + m11*y + m12]
        img = img.transform(
            img.size,
            Image.Transform.AFFINE,
            (m00, m01, m02, m10, m11, m12),
            resample=Image.Resampling.BILINEAR,
        )
        return img

    @staticmethod
    def _to_tensor(img: Image.Image, r: int) -> torch.Tensor:
        data = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(data, dtype=torch.float32).view(1, r, r)

    def __len__(self) -> int:
        return len(self.items) * self.MAX_COMBINATIONS

    def __getitem__(self, idx: int):
        item_idx = idx // self.MAX_COMBINATIONS
        combo_idx = idx % self.MAX_COMBINATIONS

        item_path, label = self.items[item_idx]
        params = self._combinations[combo_idx]

        img = Image.open(item_path).convert("L")
        img = ImageOps.invert(img)
        img = img.resize((self.r, self.r))
        img = self._apply_transform(img, **params)

        return self._to_tensor(img, self.r), label


class DigitTransformedDataset2(Dataset):
    SUBTLE_RANGES = dict(translate=(-0.1, 0.1), rotate=(-10, 10), scale=(0.7, 1.0))
    FULL_RANGES = dict(translate=(-0.4, 0.4), rotate=(-45, 45), scale=(0.3, 1.0))
    MAX_COMBINATIONS = 4
    SUBTLE_COUNT = 1

    def __init__(
        self, *, items: list[tuple[Path, int]], r: int, rng: np.random.Generator
    ):
        self.r = r
        self.items = items
        self.rng = rng
        self._combinations = self._build_combinations()

    def reroll(self) -> None:
        self._combinations = self._build_combinations()

    def _build_combinations(self) -> list[dict]:
        subtle = [self._sample(self.SUBTLE_RANGES) for _ in range(self.SUBTLE_COUNT)]
        full = [
            self._sample(self.FULL_RANGES)
            for _ in range(self.MAX_COMBINATIONS - self.SUBTLE_COUNT)
        ]
        return subtle + full

    def _sample(self, ranges: dict) -> dict:
        return dict(
            tx=self.rng.uniform(*ranges["translate"]),
            ty=self.rng.uniform(*ranges["translate"]),
            angle=self.rng.uniform(*ranges["rotate"]),
            scale=self.rng.uniform(*ranges["scale"]),
        )

    @staticmethod
    def _apply_transform(
        img: Image.Image, tx: float, ty: float, angle: float, scale: float
    ) -> Image.Image:
        """
        Compose scale + rotate + translate into a single affine pass.
        tx/ty are fractions of image size. Rotation is in degrees.
        The digit is transformed around the image center, so no internal clipping occurs.
        """
        w, h = img.size
        cx, cy = w / 2, h / 2

        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        # 2x3 affine matrix: scale + rotate around center, then translate
        # Output pixel (x,y) samples from input at M @ [x, y, 1]
        m00 = cos_a / scale
        m01 = sin_a / scale
        m10 = -sin_a / scale
        m11 = cos_a / scale

        # Keep center fixed, then apply translation (in pixel units)
        dx = tx * w
        dy = ty * h
        m02 = cx - m00 * cx - m01 * cy - dx
        m12 = cy - m10 * cx - m11 * cy - dy

        # Pillow AFFINE transform: data[x,y] = input[m00*x + m01*y + m02, m10*x + m11*y + m12]
        img = img.transform(
            img.size,
            Image.Transform.AFFINE,
            (m00, m01, m02, m10, m11, m12),
            resample=Image.Resampling.BILINEAR,
        )
        return img

    @staticmethod
    def _to_tensor(img: Image.Image, r: int) -> torch.Tensor:
        data = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(data, dtype=torch.float32).view(1, r, r)

    def __len__(self) -> int:
        return len(self.items) * self.MAX_COMBINATIONS

    def __getitem__(self, idx: int):
        item_idx = idx // self.MAX_COMBINATIONS
        combo_idx = idx % self.MAX_COMBINATIONS

        item_path, label = self.items[item_idx]
        params = self._combinations[combo_idx]

        img = Image.open(item_path).convert("L")
        img = ImageOps.invert(img)
        img = img.resize((self.r, self.r))
        img = self._apply_transform(img, **params)

        return self._to_tensor(img, self.r), label
