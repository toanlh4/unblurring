"""Image loading/saving and helpers. Images are float64 arrays in [0, 1]."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load(path: str | Path, gray: bool = False) -> np.ndarray:
    img = Image.open(path)
    img = img.convert("L" if gray else "RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def save(path: str | Path, image: np.ndarray) -> None:
    out = np.clip(np.round(image * 255), 0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(path)


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Real optical blur happens in linear light; use this before deblurring photos."""
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0, None)
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * image ** (1 / 2.4) - 0.055)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (higher = closer)."""
    mse = np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def kernel_preview(kernel: np.ndarray, size: int = 128) -> np.ndarray:
    """Scale a kernel up (nearest neighbour) and normalize it for viewing."""
    k = kernel / kernel.max()
    scale = max(1, size // max(k.shape))
    return np.kron(k, np.ones((scale, scale)))


def contact_sheet(images: list[np.ndarray], labels: list[str] | None = None, columns: int = 4) -> np.ndarray:
    """Tile same-sized images into a grid (labels are drawn with PIL)."""
    from PIL import ImageDraw

    h, w = images[0].shape[:2]
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * w, rows * h))
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(images):
        tile = Image.fromarray(np.clip(np.round(im * 255), 0, 255).astype(np.uint8)).convert("RGB")
        x, y = (i % columns) * w, (i // columns) * h
        sheet.paste(tile, (x, y))
        if labels:
            draw.rectangle([x, y, x + 8 * len(labels[i]) + 8, y + 16], fill=(0, 0, 0))
            draw.text((x + 4, y + 2), labels[i], fill=(255, 255, 0))
    return np.asarray(sheet, dtype=np.float64) / 255.0
