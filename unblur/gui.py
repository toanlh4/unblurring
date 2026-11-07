"""Desktop app: tune the kernel and deconvolution with live before/after preview.

Rendering runs on a worker thread so the window stays responsive. While you
drag a slider only the newest settings are queued, and a running
Richardson-Lucy pass is abandoned in favor of them.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import imageio
from .deconvolve import BOUNDARIES
from .pipeline import Cancelled, Params, downscale, process

SHAPES = {
    "line": ("Motion (line)", "Length"),
    "disk": ("Defocus (disk)", "Radius"),
    "polygon": ("Lens bokeh (polygon)", "Radius"),
    "box": ("Box", "Width"),
    "gaussian": ("Gaussian", "Sigma"),
}
PREVIEW_SIZES = {"640 px": 640, "1024 px": 1024, "1600 px": 1600, "Full resolution": None}
DEFAULT_PREVIEW = "1024 px"
VIEW_MODES = ("Split", "Result", "Original")
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)"
SAVE_FILTER = "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif);;BMP (*.bmp);;WebP (*.webp)"
PROGRESS_INTERVAL = 0.2  # seconds between live Richardson-Lucy previews


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(image * 255), 0, 255).astype(np.uint8)


def to_pixmap(rgb: np.ndarray) -> QPixmap:
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888))


def kernel_pixmap(kernel: np.ndarray, size: int) -> QPixmap:
    """A square picture of the kernel: pixels stay visible, and it is never stretched."""
    gray = to_uint8(kernel / kernel.max())
    h, w = gray.shape
    image = QImage(gray.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    mode = Qt.TransformationMode.SmoothTransformation if max(h, w) > size else Qt.TransformationMode.FastTransformation
    return QPixmap.fromImage(image).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, mode)


@dataclass
class Job:
    id: int
    image: np.ndarray | None  # None = only draw the kernel, there is nothing to deblur yet
    params: Params
    scale: float = 1.0
    save_path: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)


class Worker(QObject):
    """Lives on the render thread; runs one job at a time."""

    kernel_ready = Signal(object)
    progress = Signal(int, int, object)  # job id, iteration, uint8 estimate
    finished = Signal(int, object, float)  # job id, uint8 result (None if cancelled), seconds
    failed = Signal(int, str)

    @Slot(object)
    def run(self, job: Job) -> None:
        start = time.perf_counter()
        last_shown = [start]

        def on_iteration(i, view):
            if job.cancel.is_set():
                raise Cancelled
            now = time.perf_counter()
            if now - last_shown[0] >= PROGRESS_INTERVAL:
                last_shown[0] = now
                self.progress.emit(job.id, i, to_uint8(view()))

        try:
            self.kernel_ready.emit(job.params.build_kernel())
            if job.image is None:
                self.finished.emit(job.id, None, 0.0)
                return
            result = process(job.image, job.params, job.scale, on_iteration)
            if job.cancel.is_set():
                raise Cancelled
            if job.save_path:
                imageio.save(job.save_path, result)
            self.finished.emit(job.id, to_uint8(result), time.perf_counter() - start)
        except Cancelled:
            self.finished.emit(job.id, None, 0.0)
        except Exception as e:  # shown to the user; the thread must survive bad settings
            self.failed.emit(job.id, str(e) or type(e).__name__)


class LogSpinBox(QDoubleSpinBox):
    """A spin box for values spanning decades: shows 1.0e-03, arrows step by a ratio."""

    def textFromValue(self, value: float) -> str:
        return f"{value:.1e}"

    def valueFromText(self, text: str) -> float:
        return float(text)

    def validate(self, text: str, pos: int):
        try:
            ok = self.minimum() <= float(text) <= self.maximum()
        except ValueError:
            ok = False
        state = QValidator.State.Acceptable if ok else QValidator.State.Intermediate
        return state, text, pos

    def stepBy(self, steps: int) -> None:
        self.setValue(self.value() * 10 ** (steps / 10))


class ParamControl(QWidget):
    """A slider and a number box showing the same value."""

    valueChanged = Signal(float)

    def __init__(self, low, high, step, value, decimals=1, suffix="", log=False, tooltip=""):
        super().__init__()
        self._log = log
        self._low, self._step = (math.log10(low), step) if log else (low, step)
        span = (math.log10(high) - self._low) if log else (high - low)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, round(span / step))
        self.spin = LogSpinBox() if log else QDoubleSpinBox()
        self.spin.setDecimals(12 if log else decimals)
        self.spin.setRange(low, high)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(84)
        self.setToolTip(tooltip)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.set_value(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, value: float) -> None:
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setValue(value)
        self.slider.setValue(self._slider_pos(self.spin.value()))
        for w in (self.slider, self.spin):
            w.blockSignals(False)

    def _slider_pos(self, value: float) -> int:
        position = math.log10(value) if self._log else value
        return round((position - self._low) / self._step)

    def _from_slider(self, pos: int) -> None:
        position = self._low + pos * self._step
        self.spin.blockSignals(True)
        self.spin.setValue(10**position if self._log else position)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._slider_pos(value))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)


class ImageCanvas(QWidget):
    """Before/after viewer. Wheel zooms, dragging pans, the split line can be dragged.

    Positions are in original-image pixels, so a downscaled preview and its
    full-size counterpart line up and zoom stays put when the resolution changes.
    """

    zoomChanged = Signal(float)
    fileDropped = Signal(str)
    HANDLE_GRAB = 8

    def __init__(self):
        super().__init__()
        self.before: QPixmap | None = None
        self.after: QPixmap | None = None
        self.logical = (0, 0)  # full image width, height
        self.mode = "Split"
        self.split = 0.5  # fraction of the widget width
        self.zoom = 1.0
        self.origin = QPointF(0, 0)
        self._fit = True
        self._drag: str | None = None
        self._drag_from = QPointF()
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)

    def set_image(self, before: QPixmap, logical: tuple[int, int]) -> None:
        self.before, self.after, self.logical = before, None, logical
        self.fit()

    def set_before(self, before: QPixmap) -> None:
        self.before = before
        self.update()

    def set_after(self, after: QPixmap) -> None:
        self.after = after
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._update_cursor(self.mapFromGlobal(self.cursor().pos()))
        self.update()

    def fit(self) -> None:
        self._fit = True
        w, h = self.logical
        if w and h:
            self._set_zoom(min(self.width() / w, self.height() / h))
            self.origin = QPointF((self.width() - w * self.zoom) / 2, (self.height() - h * self.zoom) / 2)
        self.update()

    def actual_size(self) -> None:
        self._zoom_about(QPointF(self.width() / 2, self.height() / 2), 1.0)

    def _set_zoom(self, zoom: float) -> None:
        self.zoom = zoom
        self.zoomChanged.emit(zoom)

    def _zoom_about(self, anchor: QPointF, zoom: float) -> None:
        zoom = min(max(zoom, 0.02), 32.0)
        self.origin = anchor - (anchor - self.origin) * (zoom / self.zoom)
        self._fit = False
        self._set_zoom(zoom)
        self.update()

    def _split_x(self) -> float:
        return self.split * self.width()

    def _on_handle(self, x: float) -> bool:
        return self.mode == "Split" and self.after is not None and abs(x - self._split_x()) <= self.HANDLE_GRAB

    def _update_cursor(self, pos) -> None:
        if self._on_handle(pos.x()):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.before else Qt.CursorShape.ArrowCursor)

    def resizeEvent(self, event) -> None:
        if self._fit:
            self.fit()

    def wheelEvent(self, event) -> None:
        if self.before:
            self._zoom_about(event.position(), self.zoom * 1.2 ** (event.angleDelta().y() / 120))

    def mousePressEvent(self, event) -> None:
        if self.before and event.button() == Qt.MouseButton.LeftButton:
            self._drag = "split" if self._on_handle(event.position().x()) else "pan"
            self._drag_from = event.position() - self.origin
            if self._drag == "pan":
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._drag == "split":
            self.split = min(max(event.position().x() / self.width(), 0.0), 1.0)
            self.update()
        elif self._drag == "pan":
            self.origin = event.position() - self._drag_from
            self._fit = False
            self.update()
        else:
            self._update_cursor(event.position())

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None
        self._update_cursor(event.position())

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.fileDropped.emit(urls[0].toLocalFile())

    def _draw(self, painter: QPainter, pixmap: QPixmap, target: QRectF) -> None:
        # Hard pixel edges when zoomed far in, so you can see what is really there.
        magnified = target.width() / pixmap.width() >= 3
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, not magnified)
        painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))

    def _label(self, painter: QPainter, text: str, x: float, right: bool = False) -> None:
        metrics = painter.fontMetrics()
        w = metrics.horizontalAdvance(text) + 14
        box = QRectF(x - w - 8 if right else x + 8, 8, w, metrics.height() + 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(38, 38, 38))
        if not self.before:
            p.setPen(QColor(170, 170, 170))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open an image (Ctrl+O) or drop one here")
            return
        w, h = self.logical
        target = QRectF(self.origin.x(), self.origin.y(), w * self.zoom, h * self.zoom)
        after = self.after or self.before
        if self.mode == "Original":
            self._draw(p, self.before, target)
            self._label(p, "Original", 0)
        elif self.mode == "Result" or self.after is None:
            self._draw(p, after, target)
            self._label(p, "Result" if self.after else "Original", 0)
        else:
            x = self._split_x()
            p.setClipRect(QRectF(0, 0, x, self.height()))
            self._draw(p, self.before, target)
            p.setClipRect(QRectF(x, 0, self.width() - x, self.height()))
            self._draw(p, after, target)
            p.setClipping(False)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(QColor(255, 255, 255, 220), 2))
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            p.setBrush(QColor(255, 255, 255))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x, self.height() / 2), 7, 7)
            self._label(p, "Original", 0)
            self._label(p, "Result", self.width(), right=True)


class MainWindow(QMainWindow):
    _submit = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unblur")
        self.resize(1280, 800)
        self.image: np.ndarray | None = None
        self.path: Path | None = None
        self.preview: np.ndarray | None = None
        self.preview_scale = 1.0
        self._last_dir = ""
        self._next_id = 0
        self._running: Job | None = None
        self._queued: Job | None = None
        self._save_dialog: QProgressDialog | None = None

        self.canvas = ImageCanvas()
        self.canvas.fileDropped.connect(self.open_path)
        self.canvas.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self.setCentralWidget(self.canvas)

        # A short delay merges bursts of changes, so a drag doesn't queue stale work.
        self._debounce = QTimer(self, singleShot=True, interval=30)
        self._debounce.timeout.connect(self._render_preview)

        self._build_panel()
        self._build_menu()
        self._build_status()
        self._update_visibility()

        self._thread = QThread(self)
        self._worker = Worker()
        self._worker.moveToThread(self._thread)
        self._submit.connect(self._worker.run)
        self._worker.kernel_ready.connect(lambda k: self.kernel_view.setPixmap(kernel_pixmap(k, self.kernel_view.width())))
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()
        self.reset_settings()

    # --- layout ---------------------------------------------------------

    def _build_panel(self) -> None:
        self.open_button = QPushButton("Open…")
        self.open_button.clicked.connect(self.open_dialog)
        self.save_button = QPushButton("Save result…")
        self.save_button.setToolTip("Renders at full resolution, whatever the preview size")
        self.save_button.clicked.connect(self.save_dialog)
        self.save_button.setEnabled(False)
        files = QHBoxLayout()
        files.addWidget(self.open_button)
        files.addWidget(self.save_button)

        self.shape = QComboBox()
        for key, (label, _) in SHAPES.items():
            self.shape.addItem(label, key)
        self.shape.setToolTip(
            "Match what the blur looks like. A small light in the photo is smeared\n"
            "into the kernel's shape: streak = line, circle = disk, hexagon = polygon."
        )
        self.size = ParamControl(0.5, 150, 0.25, 5, decimals=2, suffix=" px", tooltip=(
            "Line: full streak length. Disk/polygon: radius (half the width of a smeared dot).\n"
            "Scrub until ghost copies and rings disappear."))
        self.angle = ParamControl(0, 180, 0.5, 0, suffix="°", tooltip=(
            "Line direction or polygon rotation, counter-clockwise from horizontal."))
        self.sides = ParamControl(3, 12, 1, 6, decimals=0, tooltip="Number of polygon sides (lens aperture blades).")
        self.thickness = ParamControl(0.25, 20, 0.25, 1, decimals=2, suffix=" px", tooltip="Line thickness.")
        self.softness = ParamControl(0, 10, 0.1, 0, suffix=" px", tooltip=(
            "Rounds off the kernel's edges, for blurs that aren't perfectly crisp."))
        self.kernel_view = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.kernel_view.setFixedSize(128, 128)
        self.kernel_view.setStyleSheet("background: black;")
        self.kernel_view.setToolTip("The kernel at full resolution. Compare it with a smeared dot in the photo.")
        kernel_row = QHBoxLayout()
        kernel_row.addStretch()
        kernel_row.addWidget(self.kernel_view)
        kernel_row.addStretch()

        kernel_box = QGroupBox("Kernel")
        self.kernel_form = QFormLayout(kernel_box)
        self.kernel_form.addRow("Shape", self.shape)
        self.kernel_form.addRow("Size", self.size)
        self.kernel_form.addRow("Angle", self.angle)
        self.kernel_form.addRow("Sides", self.sides)
        self.kernel_form.addRow("Thickness", self.thickness)
        self.kernel_form.addRow("Softness", self.softness)
        self.kernel_form.addRow(kernel_row)

        self.method = QComboBox()
        self.method.addItem("Wiener (fast)", "wiener")
        self.method.addItem("Richardson-Lucy (iterative)", "rl")
        self.nsr = ParamControl(1e-7, 1.0, 0.05, 1e-3, log=True, tooltip=(
            "Noise-to-signal ratio. Lower = sharper but more ringing and grain.\n"
            "Noisy photos and JPEGs usually need 1e-3 to 1e-2."))
        self.iterations = ParamControl(1, 300, 1, 30, decimals=0, tooltip=(
            "More = sharper but noisier, and proportionally slower."))
        self.boundary = QComboBox()
        self.boundary.addItems(BOUNDARIES)
        self.boundary.setToolTip("How the image is extended past its edges. Reflect suits most photos.")
        self.linear = QCheckBox("Linear light")
        self.linear.setToolTip("Deblur in linear light. Recommended for real photos; removes halos around bright areas.")

        deconv_box = QGroupBox("Deconvolution")
        self.deconv_form = QFormLayout(deconv_box)
        self.deconv_form.addRow("Method", self.method)
        self.deconv_form.addRow("NSR", self.nsr)
        self.deconv_form.addRow("Iterations", self.iterations)
        self.deconv_form.addRow("Edges", self.boundary)
        self.deconv_form.addRow("", self.linear)

        self.preview_size = QComboBox()
        self.preview_size.addItems(PREVIEW_SIZES)
        self.preview_size.setCurrentText(DEFAULT_PREVIEW)
        self.preview_size.setToolTip(
            "Longest side of the live preview. Smaller is faster but only approximates\n"
            "the final result, so check at full resolution before trusting fine detail."
        )
        self.view_mode = QComboBox()
        self.view_mode.addItems(VIEW_MODES)
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.canvas.fit)
        self.actual_button = QPushButton("100%")
        self.actual_button.clicked.connect(self.canvas.actual_size)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self.fit_button)
        zoom_row.addWidget(self.actual_button)

        view_box = QGroupBox("View")
        view_form = QFormLayout(view_box)
        view_form.addRow("Compare", self.view_mode)
        view_form.addRow("Preview", self.preview_size)
        view_form.addRow("Zoom", zoom_row)

        reset = QPushButton("Reset settings")
        reset.clicked.connect(self.reset_settings)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addLayout(files)
        for box in (kernel_box, deconv_box, view_box):
            layout.addWidget(box)
        layout.addWidget(reset)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(350)
        dock = QWidget()
        outer = QHBoxLayout(dock)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll)
        outer.addWidget(self.canvas, 1)
        self.setCentralWidget(dock)

        for control in (self.size, self.angle, self.sides, self.thickness, self.softness, self.nsr, self.iterations):
            control.valueChanged.connect(self._schedule)
        for combo in (self.shape, self.method):
            combo.currentIndexChanged.connect(self._update_visibility)
        for combo in (self.shape, self.method, self.boundary):
            combo.currentIndexChanged.connect(self._schedule)
        self.linear.toggled.connect(self._schedule)
        self.preview_size.currentIndexChanged.connect(self._preview_size_changed)
        self.view_mode.currentTextChanged.connect(self._view_mode_changed)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, shortcut, slot in (
            ("&Open…", QKeySequence.StandardKey.Open, self.open_dialog),
            ("&Save result…", QKeySequence.StandardKey.Save, self.save_dialog),
            ("&Quit", QKeySequence.StandardKey.Quit, self.close),
        ):
            action = QAction(text, self, shortcut=shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)
            if slot == self.save_dialog:
                self.save_action = action
                action.setEnabled(False)

        view_menu = self.menuBar().addMenu("&View")
        self.mode_actions = QActionGroup(self)
        for i, mode in enumerate(VIEW_MODES):
            action = QAction(mode, self, checkable=True, shortcut=f"Ctrl+{i + 1}")
            action.setChecked(mode == self.view_mode.currentText())
            action.triggered.connect(lambda _=False, m=mode: self.view_mode.setCurrentText(m))
            self.mode_actions.addAction(action)
            view_menu.addAction(action)
        view_menu.addSeparator()
        for text, shortcut, slot in (("&Fit", "F", self.canvas.fit), ("&100%", "1", self.canvas.actual_size)):
            action = QAction(text, self, shortcut=shortcut)
            action.triggered.connect(slot)
            view_menu.addAction(action)

    def _build_status(self) -> None:
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setFormat("%v / %m iterations")
        self.progress.hide()
        self.zoom_label = QLabel("")
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.zoom_label)

    def _update_visibility(self) -> None:
        shape = self.shape.currentData()
        self.kernel_form.labelForField(self.size).setText(SHAPES[shape][1])
        self.kernel_form.setRowVisible(self.angle, shape in ("line", "polygon"))
        self.kernel_form.setRowVisible(self.sides, shape == "polygon")
        self.kernel_form.setRowVisible(self.thickness, shape == "line")
        self.kernel_form.setRowVisible(self.softness, shape != "gaussian")
        wiener = self.method.currentData() == "wiener"
        self.deconv_form.setRowVisible(self.nsr, wiener)
        self.deconv_form.setRowVisible(self.iterations, not wiener)

    # --- settings -------------------------------------------------------

    def params(self) -> Params:
        return Params(
            kernel=self.shape.currentData(),
            size=self.size.value(),
            angle=self.angle.value(),
            sides=int(self.sides.value()),
            thickness=self.thickness.value(),
            softness=self.softness.value(),
            method=self.method.currentData(),
            nsr=self.nsr.value(),
            iterations=int(self.iterations.value()),
            boundary=self.boundary.currentText(),
            linear=self.linear.isChecked(),
        )

    def reset_settings(self) -> None:
        d = Params()
        controls = (self.size, self.angle, self.sides, self.thickness, self.softness, self.nsr, self.iterations)
        values = (d.size, d.angle, d.sides, d.thickness, d.softness, d.nsr, d.iterations)
        for control, value in zip(controls, values):
            control.set_value(value)
        self.shape.setCurrentIndex(self.shape.findData(d.kernel))
        self.method.setCurrentIndex(self.method.findData(d.method))
        self.boundary.setCurrentText(d.boundary)
        self.linear.setChecked(d.linear)
        self._schedule()

    def _view_mode_changed(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self.mode_actions.actions()[VIEW_MODES.index(mode)].setChecked(True)

    def _preview_size_changed(self) -> None:
        if self.image is not None:
            self._make_preview()
            self.canvas.set_before(to_pixmap(to_uint8(self.preview)))
            self._schedule()

    def _make_preview(self) -> None:
        self.preview, self.preview_scale = downscale(self.image, PREVIEW_SIZES[self.preview_size.currentText()])

    # --- files ----------------------------------------------------------

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open image", self._last_dir, IMAGE_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        try:
            image = imageio.load(path)
        except Exception as e:
            QMessageBox.warning(self, "Could not open image", f"{path}\n\n{e}")
            return
        self.path = Path(path)
        self._last_dir = str(self.path.parent)
        self.image = image
        self._make_preview()
        h, w = image.shape[:2]
        self.setWindowTitle(f"{self.path.name} ({w}×{h}) - Unblur")
        self.canvas.set_image(to_pixmap(to_uint8(self.preview)), (w, h))
        self.save_button.setEnabled(True)
        self.save_action.setEnabled(True)
        self._schedule()

    def save_dialog(self) -> None:
        if self.image is None:
            return
        default = str(self.path.with_name(f"{self.path.stem}_unblurred.png"))
        path, _ = QFileDialog.getSaveFileName(self, "Save result", default, SAVE_FILTER)
        if path:
            self.save_to(path)

    def save_to(self, path: str) -> None:
        """Render at full resolution and write it. The dialog blocks edits until it finishes."""
        job = self._new_job(self.image, 1.0, save_path=path)
        rl = job.params.method == "rl"
        dialog = QProgressDialog("Rendering at full resolution…", "Cancel", 0, job.params.iterations if rl else 0, self)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.canceled.connect(job.cancel.set)
        self._save_dialog = dialog
        dialog.show()
        self._debounce.stop()
        self._enqueue(job)

    # --- rendering ------------------------------------------------------

    def _new_job(self, image, scale, save_path=None) -> Job:
        self._next_id += 1
        return Job(self._next_id, image, self.params(), scale, save_path)

    def _schedule(self) -> None:
        self._debounce.start()

    def _render_preview(self) -> None:
        self._enqueue(self._new_job(self.preview, self.preview_scale))

    def _enqueue(self, job: Job) -> None:
        """Run `job` now, or next if busy. Only the newest waiting job is kept."""
        if self._running is None:
            self._start(job)
        else:
            self._queued = job
            if not self._running.save_path:
                self._running.cancel.set()

    def _start(self, job: Job) -> None:
        self._running = job
        if job.image is not None and not job.save_path:
            rl = job.params.method == "rl"
            self.statusBar().showMessage("Rendering…")
            self.progress.setRange(0, job.params.iterations)
            self.progress.setValue(0)
            self.progress.setVisible(rl)
        self._submit.emit(job)

    def _advance(self) -> None:
        self._running = None
        if self._queued is not None:
            job, self._queued = self._queued, None
            self._start(job)

    @Slot(int, int, object)
    def _on_progress(self, job_id: int, iteration: int, rgb: np.ndarray) -> None:
        job = self._running
        if job is None or job.id != job_id or job.cancel.is_set():
            return
        if job.save_path:
            if self._save_dialog:
                self._save_dialog.setValue(iteration)
            return
        self.progress.setValue(iteration)
        self.canvas.set_after(to_pixmap(rgb))

    @Slot(int, object, float)
    def _on_finished(self, job_id: int, rgb: np.ndarray | None, seconds: float) -> None:
        job = self._running
        if job.save_path:
            self._close_save_dialog()
            if rgb is not None:
                self.statusBar().showMessage(f"Saved {job.save_path} ({seconds:.1f} s)")
        elif rgb is not None:
            self.canvas.set_after(to_pixmap(rgb))
            self.progress.hide()
            h, w = rgb.shape[:2]
            detail = "full resolution" if job.scale == 1.0 else f"preview {w}×{h}"
            self.statusBar().showMessage(f"{detail} · {seconds * 1000:.0f} ms")
        self._advance()

    @Slot(int, str)
    def _on_failed(self, job_id: int, message: str) -> None:
        job = self._running
        self.progress.hide()
        if job.save_path:
            self._close_save_dialog()
            QMessageBox.warning(self, "Could not save", f"{job.save_path}\n\n{message}")
        else:
            self.statusBar().showMessage(f"Error: {message}")
        self._advance()

    def _close_save_dialog(self) -> None:
        if self._save_dialog:
            self._save_dialog.close()
            self._save_dialog = None

    def closeEvent(self, event) -> None:
        self._debounce.stop()
        self._queued = None
        if self._running:
            self._running.cancel.set()
        self._thread.quit()
        self._thread.wait()
        event.accept()


def run(image_path: str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    if image_path:
        window.open_path(image_path)
    return app.exec()
