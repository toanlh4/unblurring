import numpy as np
import pytest

from unblur import convolve, deconvolve, make_kernel
from unblur.imageio import psnr


def chart(h=96, w=128, channels=None):
    rng = np.random.default_rng(1)
    img = np.zeros((h, w))
    img[20:60, 30:70] = 1.0
    img[:, 90:] = np.linspace(0, 1, h)[:, None]
    img += 0.2 * rng.random((h, w))
    img = np.clip(img, 0, 1)
    return np.stack([img, img[::-1], img[:, ::-1]], -1) if channels else img


@pytest.mark.parametrize(
    "shape,params",
    [
        ("line", dict(length=15, angle=30)),
        ("disk", dict(radius=4.5)),
        ("polygon", dict(radius=5, sides=6)),
        ("box", dict(width=7)),
        ("gaussian", dict(sigma=2)),
    ],
)
def test_kernels_are_normalized_odd_and_nonnegative(shape, params):
    k = make_kernel(shape, **params)
    assert k.shape[0] % 2 == 1 and k.shape[1] % 2 == 1
    assert k.min() >= 0
    assert k.sum() == pytest.approx(1.0)


def test_symmetric_kernel_is_centered():
    k = make_kernel("disk", radius=5)
    np.testing.assert_allclose(k, k[::-1, ::-1], atol=1e-12)


def test_convolve_with_identity_is_noop():
    img = chart(channels=True)
    np.testing.assert_allclose(convolve(img, np.ones((1, 1))), img, atol=1e-10)


def test_convolve_preserves_mean_and_does_not_shift():
    img = np.zeros((65, 65))
    img[32, 32] = 1.0
    out = convolve(img, make_kernel("disk", radius=6), boundary="zero")
    assert out.sum() == pytest.approx(1.0)
    yy, xx = np.indices(out.shape)
    assert (out * yy).sum() == pytest.approx(32)  # centroid stays put
    assert (out * xx).sum() == pytest.approx(32)


@pytest.mark.parametrize("channels", [False, True])
def test_wiener_inverts_noise_free_blur(channels):
    img = chart(channels=channels)
    k = make_kernel("line", length=11, angle=20)
    blurred = convolve(img, k, boundary="wrap")
    restored = deconvolve(blurred, k, "wiener", nsr=1e-10, boundary="wrap")
    assert psnr(restored, img) > 50


def test_wiener_improves_noisy_blur():
    img = chart(channels=True)
    k = make_kernel("disk", radius=4)
    blurred = convolve(img, k) + np.random.default_rng(0).normal(0, 0.003, img.shape)
    restored = deconvolve(blurred, k, "wiener", nsr=1e-3)
    assert psnr(restored, img) > psnr(blurred, img) + 3


def test_richardson_lucy_improves_and_stays_nonnegative():
    img = chart()
    k = make_kernel("polygon", radius=4, sides=6)
    blurred = convolve(img, k)
    restored = deconvolve(blurred, k, "richardson_lucy", iterations=40)
    assert restored.min() >= 0
    assert psnr(restored, img) > psnr(blurred, img) + 3


def test_wrong_kernel_size_is_worse_than_right_one():
    img = chart()
    k = make_kernel("line", length=13, angle=0)
    blurred = convolve(img, k)
    right = psnr(deconvolve(blurred, k, nsr=1e-4), img)
    wrong = psnr(deconvolve(blurred, make_kernel("line", length=9, angle=0), nsr=1e-4), img)
    assert right > wrong + 3


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        make_kernel("star", radius=3)
    with pytest.raises(ValueError):
        deconvolve(chart(), np.ones((1, 1)), "magic")
