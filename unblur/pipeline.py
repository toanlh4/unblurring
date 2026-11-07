"""The deblur settings and processing steps shared by the CLI and the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from . import imageio
from .deconvolve import deconvolve
from .kernels import make_kernel


class Cancelled(Exception):
    """Raised from a progress callback to abandon a running deconvolution."""


@lru_cache(maxsize=16)
def _build_kernel(
    shape: str, size: float, angle: float, sides: int, thickness: float, softness: float
) -> np.ndarray:
    if shape == "line":
        return make_kernel(shape, length=size, angle=angle, thickness=thickness, softness=softness)
    if shape == "disk":
        return make_kernel(shape, radius=size, softness=softness)
    if shape == "polygon":
        return make_kernel(shape, radius=size, sides=sides, rotation=angle, softness=softness)
    if shape == "box":
        return make_kernel(shape, width=size, softness=softness)
    return make_kernel(shape, sigma=size)


@dataclass(frozen=True)
class Params:
    """Everything that determines a deblurred result. Lengths are in pixels."""

    kernel: str = "disk"
    size: float = 5.0  # line length / disk+polygon radius / box width / gaussian sigma
    angle: float = 0.0  # line angle or polygon rotation, degrees
    sides: int = 6
    thickness: float = 1.0
    softness: float = 0.0
    method: str = "wiener"  # or "rl"
    nsr: float = 1e-3
    iterations: int = 30
    boundary: str = "reflect"
    linear: bool = False

    def build_kernel(self, scale: float = 1.0) -> np.ndarray:
        """The kernel for an image resized by `scale` from the one the lengths refer to."""
        return _build_kernel(
            self.kernel,
            self.size * scale,
            self.angle,
            self.sides,
            self.thickness * scale,
            self.softness * scale,
        )


def downscale(image: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
    """Shrink so the longest side is at most `max_side` (None = no limit).

    Returns the image and the scale actually applied, so the kernel can be
    scaled to match. Area averaging keeps the blur close to the original's.
    """
    h, w = image.shape[:2]
    if max_side is None or max(h, w) <= max_side:
        return image, 1.0
    factor = max_side / max(h, w)
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    channels = [
        np.asarray(Image.fromarray(image[..., c].astype(np.float32)).resize(size, Image.Resampling.BOX))
        for c in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1).astype(np.float64), size[0] / w


def process(image: np.ndarray, params: Params, scale: float = 1.0, callback=None) -> np.ndarray:
    """Deblur `image` (sRGB, in [0, 1]); the result is sRGB, clipped to [0, 1].

    `callback(i, view)` runs after each Richardson-Lucy iteration. `view()`
    returns the current estimate ready for display, and is separate so a
    caller only pays for that when it wants to show the estimate.
    """
    kernel = params.build_kernel(scale)

    def finish(estimate: np.ndarray) -> np.ndarray:
        if params.linear:
            estimate = imageio.linear_to_srgb(estimate)
        return np.clip(estimate, 0, 1)

    working = imageio.srgb_to_linear(image) if params.linear else image
    if params.method == "wiener":
        out = deconvolve(working, kernel, "wiener", nsr=params.nsr, boundary=params.boundary)
    else:
        step = None if callback is None else lambda i, est: callback(i, lambda: finish(est))
        out = deconvolve(
            working, kernel, "richardson_lucy", iterations=params.iterations, boundary=params.boundary, callback=step
        )
    return finish(out)
