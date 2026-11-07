"""Command-line interface.

    python -m unblur demo                         # synthetic end-to-end test
    python -m unblur blur   in.png out.png --kernel line --size 25 --angle 30
    python -m unblur deblur in.png out.png --kernel line --size 25 --angle 30 --nsr 1e-3
    python -m unblur sweep  in.png sheet.png --kernel disk --sizes 4:12:1

Run `python -m unblur <command> -h` for all options.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import imageio
from .deconvolve import BOUNDARIES, convolve, deconvolve
from .kernels import KERNELS, make_kernel


def kernel_from_args(args: argparse.Namespace, size: float | None = None) -> np.ndarray:
    """Map the generic --size option onto each kernel's own parameter."""
    size = args.size if size is None else size
    soft = args.softness
    if args.kernel == "line":
        return make_kernel("line", length=size, angle=args.angle, thickness=args.thickness, softness=soft)
    if args.kernel == "disk":
        return make_kernel("disk", radius=size, softness=soft)
    if args.kernel == "polygon":
        return make_kernel("polygon", radius=size, sides=args.sides, rotation=args.angle, softness=soft)
    if args.kernel == "box":
        return make_kernel("box", width=size, softness=soft)
    return make_kernel("gaussian", sigma=size)


def deblur_with_args(image: np.ndarray, kernel: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.linear:
        image = imageio.srgb_to_linear(image)
    if args.method == "wiener":
        out = deconvolve(image, kernel, "wiener", nsr=args.nsr, boundary=args.boundary)
    else:
        out = deconvolve(image, kernel, "richardson_lucy", iterations=args.iterations, boundary=args.boundary)
    if args.linear:
        out = imageio.linear_to_srgb(out)
    return np.clip(out, 0, 1)


def add_kernel_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("kernel")
    g.add_argument("--kernel", choices=sorted(KERNELS), default="disk")
    g.add_argument("--size", type=float, default=5.0, help="line length / disk+polygon radius / box width / gaussian sigma (px)")
    g.add_argument("--angle", type=float, default=0.0, help="line angle or polygon rotation (degrees)")
    g.add_argument("--sides", type=int, default=6, help="polygon sides")
    g.add_argument("--thickness", type=float, default=1.0, help="line thickness (px)")
    g.add_argument("--softness", type=float, default=0.0, help="gaussian softening of kernel edges (px)")
    g.add_argument("--save-kernel", type=Path, help="also save a preview of the kernel")


def add_deblur_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("deconvolution")
    g.add_argument("--method", choices=["wiener", "rl"], default="wiener")
    g.add_argument("--nsr", type=float, default=1e-3, help="Wiener noise-to-signal ratio")
    g.add_argument("--iterations", type=int, default=30, help="Richardson-Lucy iterations")
    g.add_argument("--boundary", choices=BOUNDARIES, default="reflect")
    g.add_argument("--linear", action="store_true", help="deblur in linear light (for real photos)")
    g.add_argument("--gray", action="store_true", help="load as grayscale")


def cmd_blur(args: argparse.Namespace) -> None:
    image = imageio.load(args.input, gray=args.gray)
    kernel = kernel_from_args(args)
    out = convolve(image, kernel)
    if args.noise > 0:
        out = out + np.random.default_rng(0).normal(0, args.noise, out.shape)
    imageio.save(args.output, np.clip(out, 0, 1))
    if args.save_kernel:
        imageio.save(args.save_kernel, imageio.kernel_preview(kernel))
    print(f"blurred {args.input} with {args.kernel} {kernel.shape} -> {args.output}")


def cmd_deblur(args: argparse.Namespace) -> None:
    image = imageio.load(args.input, gray=args.gray)
    kernel = kernel_from_args(args)
    t = time.perf_counter()
    out = deblur_with_args(image, kernel, args)
    dt = time.perf_counter() - t
    imageio.save(args.output, out)
    if args.save_kernel:
        imageio.save(args.save_kernel, imageio.kernel_preview(kernel))
    print(f"deblurred {args.input} ({image.shape[1]}x{image.shape[0]}) in {dt * 1000:.0f} ms -> {args.output}")
    if args.reference:
        ref = imageio.load(args.reference, gray=args.gray)
        print(f"PSNR blurred: {imageio.psnr(image, ref):.2f} dB  deblurred: {imageio.psnr(out, ref):.2f} dB")


def parse_range(text: str) -> list[float]:
    """'4:12:1' -> [4, 5, ..., 12]; '3,5,8' -> [3, 5, 8]."""
    if ":" in text:
        start, stop, step = (float(v) for v in text.split(":"))
        return list(np.round(np.arange(start, stop + step / 2, step), 4))
    return [float(v) for v in text.split(",")]


def cmd_sweep(args: argparse.Namespace) -> None:
    """Try many kernel sizes and tile the results - the CLI version of
    'scrubbing the radius until it looks right'."""
    image = imageio.load(args.input, gray=args.gray)
    sizes = parse_range(args.sizes)
    results = [deblur_with_args(image, kernel_from_args(args, s), args) for s in sizes]
    sheet = imageio.contact_sheet(results, [f"size {s:g}" for s in sizes], columns=args.columns)
    imageio.save(args.output, sheet)
    print(f"swept {len(sizes)} sizes -> {args.output}")


def make_test_image(size: int = 256) -> np.ndarray:
    """Synthetic RGB test chart: text, edges, gradients, fine stripes."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 110, 110], fill=(200, 40, 40))
    d.ellipse([140, 20, 235, 115], fill=(40, 90, 200))
    for i in range(0, 90, 6):
        d.line([20 + i, 140, 20 + i, 230], fill=(20, 20, 20), width=2)
    for y, txt in enumerate(["UNBLUR", "deconv 123", "Wiener / RL"]):
        d.text((130, 140 + y * 25), txt, fill=(0, 0, 0), font_size=18)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    ramp = np.linspace(0, 1, size)
    arr[-12:, :, 1] = ramp
    return arr


def cmd_demo(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    sharp = make_test_image()
    rng = np.random.default_rng(0)
    cases = [
        ("line", make_kernel("line", length=21, angle=30)),
        ("disk", make_kernel("disk", radius=6)),
        ("hexagon", make_kernel("polygon", radius=7, sides=6)),
        ("gaussian", make_kernel("gaussian", sigma=3)),
    ]
    tiles, labels = [], []
    for name, kernel in cases:
        blurred = convolve(sharp, kernel)
        noisy = np.clip(blurred + rng.normal(0, args.noise, blurred.shape), 0, 1)
        noisy = np.round(noisy * 255) / 255  # 8-bit quantization, like a saved PNG
        w = np.clip(deconvolve(noisy, kernel, "wiener", nsr=args.nsr), 0, 1)
        rl = np.clip(deconvolve(noisy, kernel, "richardson_lucy", iterations=args.iterations), 0, 1)
        print(
            f"{name:9s} kernel {kernel.shape[0]:2d}x{kernel.shape[1]:<2d}  PSNR  "
            f"blurred {imageio.psnr(noisy, sharp):5.2f}  wiener {imageio.psnr(w, sharp):5.2f}  "
            f"RL {imageio.psnr(rl, sharp):5.2f} dB"
        )
        kprev = np.zeros_like(sharp)
        kp = imageio.kernel_preview(kernel, 200)
        kprev[: kp.shape[0], : kp.shape[1]] = kp[..., None]
        tiles += [sharp, noisy, w, rl, kprev]
        labels += ["original", f"{name} blurred", "wiener", f"RL x{args.iterations}", "kernel"]
    imageio.save(out_dir / "demo_sheet.png", imageio.contact_sheet(tiles, labels, columns=5))
    imageio.save(out_dir / "demo_sharp.png", sharp)
    imageio.save(out_dir / "demo_blurred_line.png", convolve(sharp, cases[0][1]))
    print(f"wrote {out_dir / 'demo_sheet.png'} (+ demo_sharp.png, demo_blurred_line.png)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m unblur", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("blur", help="blur an image with a known kernel (to make test data)")
    p.add_argument("input"), p.add_argument("output")
    p.add_argument("--noise", type=float, default=0.0, help="add Gaussian noise with this std (0..1 scale)")
    p.add_argument("--gray", action="store_true")
    add_kernel_args(p)
    p.set_defaults(func=cmd_blur)

    p = sub.add_parser("deblur", help="deconvolve an image")
    p.add_argument("input"), p.add_argument("output")
    p.add_argument("--reference", help="sharp original, to report PSNR")
    add_kernel_args(p), add_deblur_args(p)
    p.set_defaults(func=cmd_deblur)

    p = sub.add_parser("sweep", help="deblur with a range of kernel sizes into one contact sheet")
    p.add_argument("input"), p.add_argument("output")
    p.add_argument("--sizes", default="2:9:1", help="start:stop:step or comma list")
    p.add_argument("--columns", type=int, default=4)
    add_kernel_args(p), add_deblur_args(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("demo", help="blur a synthetic chart with several kernels and deblur it")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--noise", type=float, default=0.002)
    p.add_argument("--nsr", type=float, default=1e-3)
    p.add_argument("--iterations", type=int, default=50)
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
