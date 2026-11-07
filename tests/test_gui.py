"""Drives the real window headlessly, through the same widgets a user touches."""

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QPushButton

from unblur import convolve, imageio, make_kernel
from unblur.gui import MainWindow, Worker, kernel_pixmap
from unblur.imageio import psnr
from unblur.pipeline import Cancelled, Params, downscale, process


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    w = MainWindow()
    w.show()
    yield w
    w.close()


def wait_until(condition, timeout=20.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def idle(w):
    return not w._debounce.isActive() and w._running is None and w._queued is None


def sharp_image(h=120, w=160):
    rng = np.random.default_rng(3)
    img = np.full((h, w, 3), 0.8)
    img[20:70, 25:75] = (0.8, 0.1, 0.1)
    img[60:110, 90:140] = (0.1, 0.2, 0.8)
    return np.clip(img + 0.1 * rng.random((h, w, 3)), 0, 1)


@pytest.fixture
def blurred_file(tmp_path):
    """Sharp image, and a PNG of it blurred by a 13 px line at 30 degrees."""
    sharp = sharp_image()
    path = tmp_path / "blurred.png"
    imageio.save(path, convolve(sharp, make_kernel("line", length=13, angle=30)))
    return sharp, path


def result_of(w):
    assert wait_until(lambda: idle(w))
    return w.canvas.after


def pixels(pixmap):
    image = pixmap.toImage().convertToFormat(pixmap.toImage().Format.Format_RGB888)
    h, width, stride = image.height(), image.width(), image.bytesPerLine()
    return np.frombuffer(image.constBits(), np.uint8).reshape(h, stride)[:, : width * 3].reshape(h, width, 3) / 255.0


def test_window_starts_empty_with_a_kernel_preview(window):
    assert window.params() == Params()
    assert window.kernel_form.labelForField(window.size).text() == "Radius"
    assert wait_until(lambda: idle(window) and window.kernel_view.pixmap() is not None)
    assert not window.save_button.isEnabled()
    assert window.canvas.after is None


def test_controls_follow_the_kernel_and_method(window):
    form = window.kernel_form
    window.shape.setCurrentIndex(window.shape.findData("line"))
    assert form.labelForField(window.size).text() == "Length"
    assert form.isRowVisible(window.angle) and form.isRowVisible(window.thickness)
    assert not form.isRowVisible(window.sides)
    window.shape.setCurrentIndex(window.shape.findData("gaussian"))
    assert form.labelForField(window.size).text() == "Sigma"
    assert not form.isRowVisible(window.angle) and not form.isRowVisible(window.softness)
    window.method.setCurrentIndex(window.method.findData("rl"))
    assert window.deconv_form.isRowVisible(window.iterations)
    assert not window.deconv_form.isRowVisible(window.nsr)


def test_slider_and_box_stay_in_sync(window):
    window.size.slider.setValue(window.size.slider.value() + 4)  # 4 steps of 0.25
    assert window.size.value() == pytest.approx(6.0)
    window.size.spin.setValue(10.0)
    assert window.size.slider.value() == round((10.0 - 0.5) / 0.25)
    assert window.params().size == 10.0


def test_nsr_control_is_logarithmic(window):
    window.nsr.set_value(1e-3)
    window.nsr.slider.setValue(window.nsr.slider.value() + 20)  # 20 steps of 0.05 decades
    assert window.nsr.value() == pytest.approx(1e-2)
    assert window.nsr.spin.textFromValue(1e-2) == "1.0e-02"
    window.nsr.spin.lineEdit().setText("5e-5")
    window.nsr.spin.interpretText()
    assert window.params().nsr == pytest.approx(5e-5)


def test_right_settings_restore_the_image_and_wrong_ones_do_not(window, blurred_file):
    sharp, path = blurred_file
    window.open_path(str(path))
    window.shape.setCurrentIndex(window.shape.findData("line"))
    window.nsr.set_value(1e-3)
    window.angle.set_value(30)
    window.size.set_value(13)
    window._schedule()
    right = psnr(pixels(result_of(window)), sharp)
    window.size.set_value(25)
    window._schedule()
    wrong = psnr(pixels(result_of(window)), sharp)
    assert right > psnr(imageio.load(path), sharp) + 3
    assert right > wrong + 3


def test_rapid_changes_end_on_the_last_settings(window, blurred_file):
    sharp, path = blurred_file
    window.open_path(str(path))
    window.shape.setCurrentIndex(window.shape.findData("line"))
    window.method.setCurrentIndex(window.method.findData("rl"))
    window.iterations.set_value(40)
    window.angle.set_value(30)
    for size in np.arange(3, 13.25, 0.25):  # like dragging the slider
        window.size.set_value(float(size))
        window._schedule()
        QApplication.processEvents()
    shown = pixels(result_of(window))
    expected = process(imageio.load(path), window.params())
    assert window.params().size == 13
    assert np.abs(shown - expected).max() < 2 / 255


def test_preview_scales_the_kernel_and_save_uses_full_resolution(window, tmp_path):
    big = np.kron(sharp_image(), np.ones((10, 10, 1)))  # 1600 x 1200
    source = tmp_path / "big.png"
    imageio.save(source, big)
    window.preview_size.setCurrentText("640 px")
    window.open_path(str(source))
    window.size.set_value(20)
    window._schedule()
    after = result_of(window)
    assert (after.width(), after.height()) == (640, 480)
    assert window.preview_scale == pytest.approx(0.4)

    out = tmp_path / "out" / "result.png"
    window.save_to(str(out))
    assert wait_until(lambda: idle(window) and window._save_dialog is None, timeout=60)
    saved = imageio.load(out)
    assert saved.shape == big.shape
    expected = process(imageio.load(source), window.params())
    assert np.abs(saved - expected).max() < 1 / 255 + 1e-9
    assert "Saved" in window.statusBar().currentMessage()


def test_cancelling_a_save_writes_nothing(window, blurred_file, tmp_path):
    _, path = blurred_file
    window.open_path(str(path))
    assert wait_until(lambda: idle(window))
    window.method.setCurrentIndex(window.method.findData("rl"))
    window.iterations.set_value(300)
    out = tmp_path / "never.png"
    window.save_to(str(out))
    window._save_dialog.findChild(QPushButton).click()
    assert wait_until(lambda: window._save_dialog is None)
    assert not out.exists()


def test_save_failure_is_reported_and_the_app_keeps_working(window, blurred_file, tmp_path, monkeypatch):
    _, path = blurred_file
    window.open_path(str(path))
    assert wait_until(lambda: idle(window))
    warnings = []
    monkeypatch.setattr("unblur.gui.QMessageBox.warning", lambda *a: warnings.append(a[1]))
    window.save_to(str(tmp_path / "bad.notanimageformat"))
    assert wait_until(lambda: bool(warnings))
    assert warnings == ["Could not save"]
    window.size.set_value(7)
    window._schedule()
    assert wait_until(lambda: idle(window))
    assert "ms" in window.statusBar().currentMessage()


def test_opening_a_bad_file_keeps_the_current_image(window, blurred_file, tmp_path, monkeypatch):
    _, path = blurred_file
    window.open_path(str(path))
    bad = tmp_path / "not_an_image.png"
    bad.write_text("nope")
    warnings = []
    monkeypatch.setattr("unblur.gui.QMessageBox.warning", lambda *a: warnings.append(a[1]))
    window.open_path(str(bad))
    assert warnings == ["Could not open image"]
    assert window.path == path


def test_impossible_settings_show_an_error_not_a_crash(window, blurred_file):
    _, path = blurred_file
    window.open_path(str(path))
    window.boundary.setCurrentText("wrap")
    window.size.set_value(150)  # disk 301 px wide, image only 160
    window._schedule()
    assert wait_until(lambda: idle(window))
    assert "larger than the image" in window.statusBar().currentMessage()
    window.size.set_value(5)
    window._schedule()
    assert wait_until(lambda: idle(window))
    assert "ms" in window.statusBar().currentMessage()


def test_canvas_zoom_keeps_the_point_under_the_cursor(window, blurred_file):
    _, path = blurred_file
    window.open_path(str(path))
    canvas = window.canvas
    anchor = QPointF(200, 150)
    before = (anchor - canvas.origin) / canvas.zoom
    fitted = canvas.zoom
    event = QWheelEvent(anchor, canvas.mapToGlobal(anchor), QPoint(0, 0), QPoint(0, 240),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    canvas.wheelEvent(event)
    after = (anchor - canvas.origin) / canvas.zoom
    assert canvas.zoom == pytest.approx(fitted * 1.2**2)
    assert (after.x(), after.y()) == pytest.approx((before.x(), before.y()))
    canvas.fit()
    assert canvas.zoom == pytest.approx(min(canvas.width() / 160, canvas.height() / 120))


def _mouse(kind, x, y=50):
    pos = QPointF(x, y)
    return QMouseEvent(kind, pos, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)


def test_dragging_moves_the_split_line_or_pans(window, blurred_file):
    _, path = blurred_file
    window.open_path(str(path))
    assert wait_until(lambda: idle(window) and window.canvas.after is not None)
    canvas = window.canvas
    width = canvas.width()
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, width * 0.5))
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, width * 0.25))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, width * 0.25))
    assert canvas.split == pytest.approx(0.25)

    origin = canvas.origin
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, width * 0.75))
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, width * 0.75 + 30))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, width * 0.75 + 30))
    assert canvas.origin.x() == pytest.approx(origin.x() + 30)
    assert canvas.split == pytest.approx(0.25)


def test_reset_restores_defaults(window):
    window.shape.setCurrentIndex(window.shape.findData("line"))
    window.size.set_value(42)
    window.method.setCurrentIndex(window.method.findData("rl"))
    window.linear.setChecked(True)
    window.reset_settings()
    assert window.params() == Params()


def test_kernel_picture_is_never_stretched(app):
    tall = np.ones((10, 2)) / 20
    pixmap = kernel_pixmap(tall, 128)
    assert (pixmap.width(), pixmap.height()) == (25, 128)


def test_downscale_reports_the_scale_it_applied():
    image = np.random.default_rng(0).random((100, 301, 3))
    small, scale = downscale(image, 100)
    assert small.shape == (33, 100, 3)
    assert scale == pytest.approx(100 / 301)
    assert downscale(image, 500)[0] is image and downscale(image, 500)[1] == 1.0
    assert downscale(image, None)[1] == 1.0


def test_process_can_be_cancelled_from_its_callback():
    def cancel(i, view):
        assert view().max() <= 1
        raise Cancelled

    with pytest.raises(Cancelled):
        process(sharp_image(), Params(method="rl", iterations=5), callback=cancel)


def test_worker_reports_cancellation_as_no_result(app):
    worker = Worker()
    finished = []
    worker.finished.connect(lambda i, r, s: finished.append((i, r)))
    from unblur.gui import Job

    event = threading.Event()
    event.set()
    worker.run(Job(1, sharp_image(), Params(), cancel=event))
    assert finished == [(1, None)]
