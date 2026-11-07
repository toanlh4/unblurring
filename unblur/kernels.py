"""Blur kernel (point spread function) generators.

Every kernel is a 2D float64 array with odd width/height, centered, and
normalized so it sums to 1. Shapes are rasterized with supersampling so
fractional sizes and angles produce smooth, anti-aliased kernels - this
matters because deconvolution is very sensitive to the exact kernel.
"""

from __future__ import annotations

import numpy as np

SUPERSAMPLE = 8


def _grid(half: int, ss: int = SUPERSAMPLE) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel sample coordinates for a (2*half+1)^2 kernel, centered at 0."""
    size = 2 * half + 1
    n = size * ss
    coords = (np.arange(n) + 0.5) / ss - size / 2
    return np.meshgrid(coords, coords)  # x, y


def _downsample(mask: np.ndarray, ss: int = SUPERSAMPLE) -> np.ndarray:
    h, w = mask.shape
    return mask.reshape(h // ss, ss, w // ss, ss).mean(axis=(1, 3))


def _normalize(k: np.ndarray) -> np.ndarray:
    total = k.sum()
    if total <= 0:
        # Degenerate (zero-size) kernel: identity.
        k = np.zeros((1, 1))
        k[0, 0] = 1.0
        return k
    return k / total


def _soften(k: np.ndarray, softness: float) -> np.ndarray:
    """Optionally blur the kernel edges with a small Gaussian."""
    if softness <= 0:
        return k
    g = gaussian(softness)
    pad = g.shape[0] // 2
    k = np.pad(k, pad)
    from .deconvolve import convolve  # local import avoids a cycle

    return np.clip(convolve(k, g, boundary="zero"), 0, None)


def identity() -> np.ndarray:
    return np.ones((1, 1))


def box(width: float, height: float | None = None, softness: float = 0.0) -> np.ndarray:
    """Axis-aligned box blur of the given size in pixels."""
    height = width if height is None else height
    half = int(np.ceil(max(width, height) / 2))
    x, y = _grid(half)
    mask = (np.abs(x) <= width / 2) & (np.abs(y) <= height / 2)
    return _normalize(_soften(_downsample(mask.astype(float)), softness))


def disk(radius: float, softness: float = 0.0) -> np.ndarray:
    """Circular (defocus / bokeh) kernel."""
    half = int(np.ceil(radius))
    x, y = _grid(half)
    mask = x**2 + y**2 <= radius**2
    return _normalize(_soften(_downsample(mask.astype(float)), softness))


def polygon(radius: float, sides: int = 6, rotation: float = 0.0, softness: float = 0.0) -> np.ndarray:
    """Regular polygon kernel (e.g. hexagonal lens bokeh).

    `radius` is the circumradius in pixels, `rotation` in degrees.
    """
    if sides < 3:
        raise ValueError("polygon needs at least 3 sides")
    half = int(np.ceil(radius))
    x, y = _grid(half)
    # A point is inside if it is behind every edge's half-plane.
    apothem = radius * np.cos(np.pi / sides)
    mask = np.ones_like(x, dtype=bool)
    for i in range(sides):
        a = np.deg2rad(rotation) + (i + 0.5) * 2 * np.pi / sides
        mask &= x * np.cos(a) + y * np.sin(a) <= apothem
    return _normalize(_soften(_downsample(mask.astype(float)), softness))


def line(length: float, angle: float = 0.0, thickness: float = 1.0, softness: float = 0.0) -> np.ndarray:
    """Linear motion blur kernel.

    `length` in pixels, `angle` in degrees counter-clockwise from +x.
    """
    half = int(np.ceil(max(length, thickness) / 2))
    x, y = _grid(half)
    a = np.deg2rad(angle)
    # Image y axis points down, so negate for counter-clockwise angles.
    along = x * np.cos(a) - y * np.sin(a)
    across = x * np.sin(a) + y * np.cos(a)
    mask = (np.abs(along) <= length / 2) & (np.abs(across) <= thickness / 2)
    return _normalize(_soften(_downsample(mask.astype(float)), softness))


def gaussian(sigma: float) -> np.ndarray:
    """Gaussian kernel. Note: hard to invert - it kills high frequencies."""
    if sigma <= 0:
        return identity()
    half = int(np.ceil(3 * sigma))
    r = np.arange(-half, half + 1)
    g = np.exp(-(r**2) / (2 * sigma**2))
    return _normalize(np.outer(g, g))


KERNELS = {
    "box": box,
    "disk": disk,
    "polygon": polygon,
    "line": line,
    "gaussian": gaussian,
}


def make_kernel(shape: str, **params) -> np.ndarray:
    """Build a kernel by name, e.g. make_kernel("line", length=20, angle=30)."""
    try:
        fn = KERNELS[shape]
    except KeyError:
        raise ValueError(f"unknown kernel {shape!r}; choose from {sorted(KERNELS)}") from None
    return fn(**params)
