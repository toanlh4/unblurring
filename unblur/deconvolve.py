"""FFT-based convolution and deconvolution.

Images are float arrays in [0, 1], shape (H, W) or (H, W, C). Kernels are
2D arrays from `unblur.kernels`. All operations run per channel by
broadcasting the kernel's transfer function over the channel axis.

Boundary handling: the FFT treats the image as periodic, so the left edge
"wraps" into the right edge and causes ringing. To avoid that we pad the
image (reflect by default) by the kernel size before transforming and crop
the result afterwards. Padded sizes are rounded up to FFT-friendly lengths
(products of small primes), which can be many times faster than odd sizes.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft

EPS = 1e-12
BOUNDARIES = ("reflect", "edge", "zero", "wrap")


def psf2otf(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Transfer function (OTF) of `kernel` for an FFT of size `shape`.

    The kernel is zero-padded and rolled so its center sits at pixel (0, 0),
    which keeps the output aligned with the input (no shift).
    """
    kh, kw = kernel.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return sfft.rfft2(padded, workers=-1)


# A crop region: the original image's (top, left, height, width) inside the padded one.
Region = tuple[int, int, int, int]


def _pad(image: np.ndarray, kernel: np.ndarray, boundary: str, margin: int = 1) -> tuple[np.ndarray, Region]:
    """Pad by `margin` x kernel size on each side, plus extra at the bottom/right
    to reach an FFT-friendly size."""
    h, w = image.shape[:2]
    if boundary == "wrap":
        return image, (0, 0, h, w)
    ph, pw = margin * kernel.shape[0], margin * kernel.shape[1]
    eh = sfft.next_fast_len(h + 2 * ph, real=True) - (h + 2 * ph)
    ew = sfft.next_fast_len(w + 2 * pw, real=True) - (w + 2 * pw)
    widths = [(ph, ph + eh), (pw, pw + ew)] + [(0, 0)] * (image.ndim - 2)
    mode = {"reflect": "symmetric", "edge": "edge", "zero": "constant"}[boundary]
    return np.pad(image, widths, mode=mode), (ph, pw, h, w)


def _taper_weight(n: int, before: int, after: int) -> np.ndarray:
    """1 in the middle, cosine ramp down to 0 across the `before`/`after` borders."""
    w = np.ones(n)
    for pad, side in ((before, slice(0, before)), (after, slice(n - after, n))):
        if pad > 0:
            ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(pad) + 0.5) / pad)
            w[side] = ramp if side.start == 0 else ramp[::-1]
    return w


def _pad_for_deconv(image: np.ndarray, kernel: np.ndarray, boundary: str) -> tuple[np.ndarray, Region]:
    """Pad, then edge-taper so the padded image is consistent with the
    periodic (circular) blur model the FFT assumes.

    Without this, the jump where the padded image wraps around is "explained"
    by the inverse filter as huge amplified frequencies, and ringing covers
    the whole image. Like MATLAB's edgetaper: near the outer border we blend
    toward a circularly-blurred copy, which the circular model fits exactly.
    """
    padded, region = _pad(image, kernel, boundary, margin=2)
    if boundary == "wrap":
        return padded, region
    top, left, rh, rw = region
    H_, W_ = padded.shape[:2]
    H = _otf_for(padded, kernel)
    w = np.outer(_taper_weight(H_, top, H_ - top - rh), _taper_weight(W_, left, W_ - left - rw))
    if padded.ndim == 3:
        w = w[..., None]
    for _ in range(3):  # a few passes, as MATLAB recommends
        reblurred = _ifft(_fft(padded) * H, padded.shape[:2])
        padded = w * padded + (1 - w) * reblurred
    return padded, region


def _crop(image: np.ndarray, region: Region) -> np.ndarray:
    top, left, h, w = region
    return image[top : top + h, left : left + w]


def _fft(image: np.ndarray) -> np.ndarray:
    return sfft.rfft2(image, axes=(0, 1), workers=-1)


def _ifft(spectrum: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return sfft.irfft2(spectrum, s=shape, axes=(0, 1), workers=-1)


def _otf_for(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    otf = psf2otf(kernel, image.shape[:2])
    # Add a channel axis so it broadcasts over (H, W, C) spectra.
    return otf[..., None] if image.ndim == 3 else otf


def convolve(image: np.ndarray, kernel: np.ndarray, boundary: str = "reflect") -> np.ndarray:
    """Blur `image` with `kernel`. boundary: reflect | edge | zero | wrap."""
    padded, pad = _pad(np.asarray(image, dtype=np.float64), kernel, boundary)
    out = _ifft(_fft(padded) * _otf_for(padded, kernel), padded.shape[:2])
    return _crop(out, pad)


def wiener(image: np.ndarray, kernel: np.ndarray, nsr: float = 1e-3, boundary: str = "reflect") -> np.ndarray:
    """Wiener deconvolution.

        X = conj(H) * G / (|H|^2 + nsr)

    `nsr` is the noise-to-signal power ratio. 0 gives a pure inverse filter
    (only usable on noise-free, uncompressed images); larger values suppress
    noise and ringing at the cost of sharpness. Typical range 1e-5 .. 1e-1.
    """
    padded, pad = _pad_for_deconv(np.asarray(image, dtype=np.float64), kernel, boundary)
    H = _otf_for(padded, kernel)
    X = np.conj(H) * _fft(padded) / (np.abs(H) ** 2 + max(nsr, EPS))
    return _crop(_ifft(X, padded.shape[:2]), pad)


def richardson_lucy(
    image: np.ndarray,
    kernel: np.ndarray,
    iterations: int = 30,
    boundary: str = "reflect",
    callback=None,
) -> np.ndarray:
    """Richardson-Lucy deconvolution (iterative, assumes Poisson noise).

    More iterations = sharper but noisier. Output stays non-negative.
    `callback(i, estimate)` is called after each iteration if given
    (useful later for a GUI progress bar / live preview).
    """
    padded, pad = _pad_for_deconv(np.clip(np.asarray(image, dtype=np.float64), 0, None), kernel, boundary)
    shape = padded.shape[:2]
    H = _otf_for(padded, kernel)
    Hc = np.conj(H)  # convolution with the flipped kernel

    estimate = np.full_like(padded, padded.mean())
    for i in range(iterations):
        reblurred = _ifft(_fft(estimate) * H, shape)
        ratio = padded / np.maximum(reblurred, EPS)
        estimate = estimate * _ifft(_fft(ratio) * Hc, shape)
        estimate = np.maximum(estimate, 0)
        if callback is not None:
            callback(i + 1, _crop(estimate, pad))
    return _crop(estimate, pad)


METHODS = ("wiener", "richardson_lucy")


def deconvolve(image: np.ndarray, kernel: np.ndarray, method: str = "wiener", **params) -> np.ndarray:
    """Dispatch to a deconvolution method by name."""
    if method == "wiener":
        return wiener(image, kernel, **params)
    if method in ("richardson_lucy", "rl"):
        return richardson_lucy(image, kernel, **params)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
