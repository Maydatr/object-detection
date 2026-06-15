"""
main.py - Akıllı Odak Tanıma (PyQt6)

Görünüm modları:
  Tekil - tek model, tıkla-kilitle + odak paneli aktif
  Grid  - seçili 1-3 model yan yana karşılaştırma
"""
from __future__ import annotations

import os
import sys
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from backends import MODEL_REGISTRY, create_backend, detections_to_legend
from capture_engine import CaptureEngine
from sources import list_cameras


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    image = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def scale_pixmap_to_fit(pixmap: QPixmap, w: int, h: int) -> QPixmap:
    return pixmap.scaled(
        w,
        h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def compute_display_rect(
    frame_w: int, frame_h: int, view_w: int, view_h: int
) -> tuple[int, int, int, int]:
    scale = min(view_w / frame_w, view_h / frame_h)
    disp_w = int(frame_w * scale)
    disp_h = int(frame_h * scale)
    offset_x = (view_w - disp_w) // 2
    offset_y = (view_h - disp_h) // 2
    return offset_x, offset_y, disp_w, disp_h


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.deleteLater()


def widget_to_frame(
    wx: int,
    wy: int,
    view_w: int,
    view_h: int,
    frame_size: tuple[int, int] | None,
    display_rect: tuple[int, int, int, int] | None,
) -> tuple[int, int] | None:
    if frame_size is None or display_rect is None:
        return None
    frame_w, frame_h = frame_size
    ox, oy, dw, dh = display_rect
    if not (ox <= wx < ox + dw and oy <= wy < oy + dh):
        return None
    fx = int((wx - ox) * frame_w / dw)
    fy = int((wy - oy) * frame_h / dh)
    return fx, fy


# ---------------------------------------------------------------------------
# Detection legend overlay (renders on top of ScaledVideoLabel)
# ---------------------------------------------------------------------------
class DetectionLegend(QFrame):
    _MAX_ITEMS = 8

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(150)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(3)

        header = QLabel("Tespitler")
        header.setStyleSheet(
            "background: transparent; font-size: 10px; font-weight: bold;"
            " color: rgba(255,255,255,160);"
        )
        outer.addWidget(header)

        self._summary_container = QWidget()
        self._summary_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._summary_layout = QVBoxLayout(self._summary_container)
        self._summary_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_layout.setSpacing(1)
        outer.addWidget(self._summary_container)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: rgba(255,255,255,35);")
        sep.setFixedHeight(1)
        outer.addWidget(sep)

        self._items_container = QWidget()
        self._items_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._items_layout = QVBoxLayout(self._items_container)
        self._items_layout.setContentsMargins(0, 0, 0, 0)
        self._items_layout.setSpacing(2)
        outer.addWidget(self._items_container)

        self._empty_label = QLabel("Tespit yok")
        self._empty_label.setStyleSheet(
            "background: transparent; font-size: 10px;"
            " color: rgba(255,255,255,90); font-style: italic;"
        )
        outer.addWidget(self._empty_label)

        self._accent_hex = "#2F6DF6"

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 8, 8)
        p.fillPath(path, QColor(0, 0, 0, 185))
        p.setPen(QColor(255, 255, 255, 30))
        p.drawPath(path)
        p.end()

    def update_data(self, data: dict | None) -> None:
        _clear_layout(self._summary_layout)
        _clear_layout(self._items_layout)

        if not data or not data.get("items"):
            self._summary_container.setVisible(False)
            self._items_container.setVisible(False)
            self._empty_label.setVisible(True)
            self.adjustSize()
            return

        self._accent_hex = data.get("accent_hex", "#2F6DF6")
        self._empty_label.setVisible(False)

        summary = data.get("summary", [])
        if summary:
            self._summary_container.setVisible(True)
            for s in summary:
                lbl = QLabel(f"{s['name']}  x{s['count']}  {s['max_conf']:.2f}")
                lbl.setStyleSheet(
                    "background: transparent; font-size: 10px;"
                    " color: rgba(255,255,255,200);"
                )
                self._summary_layout.addWidget(lbl)
        else:
            self._summary_container.setVisible(False)

        items = data.get("items", [])
        if items:
            self._items_container.setVisible(True)
            for item in items[: self._MAX_ITEMS]:
                self._items_layout.addWidget(self._make_item_row(item))
        else:
            self._items_container.setVisible(False)

        self.adjustSize()

    def _make_item_row(self, item: dict) -> QWidget:
        w = QWidget()
        w.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)

        dot = QFrame()
        dot.setFixedSize(8, 8)
        color = "#2F6DF6" if item.get("is_focused") else self._accent_hex
        dot.setStyleSheet(f"background-color: {color}; border-radius: 4px;")
        row.addWidget(dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        name = item.get("name", "?")
        conf = item.get("confidence")
        tid = item.get("track_id")
        conf_str = f"{conf:.2f}" if conf is not None else "--"
        tid_str = f"  ID:{tid}" if tid is not None else ""
        lbl = QLabel(f"{name}  {conf_str}{tid_str}")
        if item.get("is_focused"):
            lbl.setStyleSheet(
                "background: transparent; font-size: 10px;"
                " color: #2F6DF6; font-weight: bold;"
            )
        else:
            lbl.setStyleSheet(
                "background: transparent; font-size: 10px;"
                " color: rgba(255,255,255,200);"
            )
        row.addWidget(lbl)
        row.addStretch()
        return w


# ---------------------------------------------------------------------------
# Capture bridge (worker thread -> GUI thread via queued signals)
# ---------------------------------------------------------------------------
class CaptureBridge(QObject):
    panel_ready = pyqtSignal(str, object, float, int, object)
    focus_info = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    device = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.engine = CaptureEngine(
            on_panel=self._emit_panel,
            on_focus=self._emit_focus,
            on_status=self._emit_status,
            on_error=self._emit_error,
            on_device=self._emit_device,
            on_finished=self._emit_finished,
        )

    def _emit_panel(self, model_id: str, frame: np.ndarray, latency_ms: float, count: int, legend_data: dict) -> None:
        self.panel_ready.emit(model_id, frame, latency_ms, count, legend_data)

    def _emit_focus(self, info: dict[str, Any] | None) -> None:
        self.focus_info.emit(info)

    def _emit_status(self, message: str) -> None:
        self.status.emit(message)

    def _emit_error(self, message: str) -> None:
        self.error.emit(message)

    def _emit_device(self, device: str) -> None:
        self.device.emit(device)

    def _emit_finished(self) -> None:
        self.finished.emit()


# ---------------------------------------------------------------------------
# Scaled video label (single + grid)
# ---------------------------------------------------------------------------
class ScaledVideoLabel(QLabel):
    clicked = pyqtSignal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setStyleSheet("background-color: #000; border-radius: 8px;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._source_pixmap: QPixmap | None = None
        self._frame_size: tuple[int, int] | None = None
        self._display_rect: tuple[int, int, int, int] | None = None
        self._legend = DetectionLegend(self)
        self._legend.setVisible(False)

    def clear(self) -> None:
        self._source_pixmap = None
        self._frame_size = None
        self._display_rect = None
        self._legend.setVisible(False)
        super().clear()

    def set_frame(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        self._source_pixmap = frame_to_pixmap(frame)
        return self._apply_scale()

    def set_legend_data(self, data: dict | None) -> None:
        if data is None:
            self._legend.setVisible(False)
            return
        self._legend.update_data(data)
        self._legend.setVisible(True)
        self._position_legend()

    def _apply_scale(self) -> tuple[int, int, int, int] | None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return None
        vw, vh = self.width(), self.height()
        if vw <= 0 or vh <= 0 or self._frame_size is None:
            return None
        frame_w, frame_h = self._frame_size
        self._display_rect = compute_display_rect(frame_w, frame_h, vw, vh)
        ox, oy, dw, dh = self._display_rect
        super().setPixmap(scale_pixmap_to_fit(self._source_pixmap, dw, dh))
        self._position_legend()
        return self._display_rect

    def _position_legend(self) -> None:
        if self._display_rect is None or not self._legend.isVisible():
            return
        margin = 10
        ox, oy, dw, dh = self._display_rect
        lw = self._legend.width()
        lh = self._legend.height()
        x = max(ox, ox + dw - lw - margin)
        y = max(oy, oy + dh - lh - margin)
        self._legend.setGeometry(x, y, lw, lh)
        self._legend.raise_()

    def resizeEvent(self, event) -> None:
        self._apply_scale()
        super().resizeEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self.clicked.emit(int(pos.x()), int(pos.y()))
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Grid model panel
# ---------------------------------------------------------------------------
class ModelGridPanel(QWidget):
    def __init__(self, model_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model_id = model_id
        self.setStyleSheet("background-color: #000; border-radius: 8px;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.model_badge = QLabel(model_id)
        self.model_badge.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.latency_badge = QLabel("-- ms | -- fps | 0 obj")
        self.latency_badge.setStyleSheet("font-size: 11px;")
        header.addWidget(self.model_badge)
        header.addStretch()
        header.addWidget(self.latency_badge)
        layout.addLayout(header)

        self.image_label = ScaledVideoLabel()
        self.image_label.setStyleSheet("background-color: #111; border-radius: 6px;")
        layout.addWidget(self.image_label, stretch=1)

    def update_frame(
        self,
        frame: np.ndarray,
        latency_ms: float,
        count: int,
        legend_data: dict | None = None,
    ) -> None:
        model_fps = 1000.0 / max(latency_ms, 1.0)
        self.latency_badge.setText(
            f"{latency_ms:.0f} ms | {model_fps:.0f} fps | {count} obj"
        )
        self.image_label.set_frame(frame)
        self.image_label.set_legend_data(legend_data)


# ---------------------------------------------------------------------------
# Clickable card for image grid
# ---------------------------------------------------------------------------
class ClickableCard(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QFrame { background-color: #1e1e1e; border-radius: 8px; border: 1px solid #333; }"
            "QFrame:hover { background-color: #252525; border: 1px solid #555; }"
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Image inference worker (background thread for static image processing)
# ---------------------------------------------------------------------------
class ImageInferenceWorker(QThread):
    # index, annotated_frame, count, legend_data, raw_frame, detections
    result_ready = pyqtSignal(int, object, int, object, object, object)

    def __init__(self, paths: list[str], backend, conf: float) -> None:
        super().__init__()
        self._paths = paths
        self._backend = backend
        self._conf = conf

    def run(self) -> None:
        self._backend.load()
        for i, path in enumerate(self._paths):
            frame = cv2.imread(path)
            if frame is None:
                self.result_ready.emit(i, None, 0, None, None, None)
                continue
            detections = self._backend.infer(frame, self._conf)
            annotated = self._backend.annotate(frame, detections)
            legend = detections_to_legend(
                detections, accent_hex=self._backend.accent_color_hex
            )
            self.result_ready.emit(i, annotated, len(detections), legend, frame, detections)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    SRC_CAMERA = 0
    SRC_FILE = 1
    SRC_RTSP = 2
    SRC_IMAGE = 3

    VIEW_SINGLE = 0
    VIEW_GRID = 1
    VIEW_IMAGE = 2
    VIEW_IMAGE_DETAIL = 3

    _IMG_COLS = 3
    _IMG_THUMB_W = 280
    _IMG_THUMB_H = 200
    _STRIP_THUMB_W = 90
    _STRIP_THUMB_H = 64

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Araç Hasar Tespiti")
        self.setMinimumSize(1080, 680)
        self.resize(1440, 860)

        self._video_file_path = ""
        self._frame_size: tuple[int, int] | None = None
        self._display_rect: tuple[int, int, int, int] | None = None
        self._grid_panels: dict[str, ModelGridPanel] = {}
        self._camera_indices: list[int] = []
        self._model_checkboxes: dict[str, QCheckBox] = {}
        self._image_paths: list[str] = []
        self._image_cards: list[dict] = []
        self._image_worker: ImageInferenceWorker | None = None
        self._image_backend = None
        self._detail_index: int = -1
        self._strip_thumb_labels: list[QLabel] = []

        self.bridge = CaptureBridge()
        self._connect_bridge()

        self._build_ui()
        self._refresh_cameras()
        self._on_view_changed(self.VIEW_SINGLE)

    def _connect_bridge(self) -> None:
        self.bridge.panel_ready.connect(self._on_panel_ready)
        self.bridge.focus_info.connect(self._on_focus_info)
        self.bridge.status.connect(self._log)
        self.bridge.error.connect(self._on_error)
        self.bridge.device.connect(self._set_device)
        self.bridge.finished.connect(self._on_capture_finished)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        root.addWidget(self._build_sidebar(), stretch=0)
        root.addWidget(self._build_content(), stretch=1)
        root.addWidget(self._build_inspector(), stretch=0)

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(360)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(14)

        # Görünüm (gizli — uygulama tek-model modunda kilitli)
        view_box = QGroupBox("Görünüm")
        view_layout = QHBoxLayout(view_box)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Tekil", "Grid"])
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        view_layout.addWidget(self.view_combo)
        layout.addWidget(view_box)
        view_box.setVisible(False)

        # Kaynak
        src_box = QGroupBox("Kaynak")
        src_layout = QVBoxLayout(src_box)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Kamera", "Video", "RTSP", "Resim"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        src_layout.addWidget(self.source_combo)

        self.source_stack = QStackedWidget()
        cam_page = QWidget()
        cam_layout = QHBoxLayout(cam_page)
        self.camera_combo = QComboBox()
        self.refresh_button = QPushButton("Yenile")
        self.refresh_button.clicked.connect(self._refresh_cameras)
        cam_layout.addWidget(self.camera_combo, stretch=1)
        cam_layout.addWidget(self.refresh_button)
        self.source_stack.addWidget(cam_page)

        file_page = QWidget()
        file_layout = QHBoxLayout(file_page)
        self.file_field = QLineEdit()
        self.file_field.setPlaceholderText("Video dosyası seçilmedi")
        self.file_field.setReadOnly(True)
        self.browse_button = QPushButton("Gözat")
        self.browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_field, stretch=1)
        file_layout.addWidget(self.browse_button)
        self.source_stack.addWidget(file_page)

        rtsp_page = QWidget()
        rtsp_layout = QVBoxLayout(rtsp_page)
        self.rtsp_field = QLineEdit()
        self.rtsp_field.setPlaceholderText("rtsp://kullanıcı:şifre@ip:554/stream")
        rtsp_layout.addWidget(self.rtsp_field)
        self.source_stack.addWidget(rtsp_page)

        image_page = QWidget()
        image_page_layout = QVBoxLayout(image_page)
        image_page_layout.setContentsMargins(0, 4, 0, 0)
        self.image_browse_button = QPushButton("Gözat...")
        self.image_browse_button.clicked.connect(self._browse_images)
        self.image_count_label = QLabel("Resim secilmedi")
        self.image_count_label.setStyleSheet("font-size: 11px; color: #888;")
        image_page_layout.addWidget(self.image_browse_button)
        image_page_layout.addWidget(self.image_count_label)
        self.source_stack.addWidget(image_page)

        src_layout.addWidget(self.source_stack)
        layout.addWidget(src_box)

        # Model Secimi (gizli — tek hasar modeline kilitli)
        model_box = QGroupBox("Model Seçimi")
        model_layout = QVBoxLayout(model_box)
        self.model_stack = QStackedWidget()

        single_page = QWidget()
        single_layout = QVBoxLayout(single_page)
        single_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_REGISTRY.keys()))
        single_layout.addWidget(self.model_combo)
        self.model_stack.addWidget(single_page)

        grid_page = QWidget()
        grid_layout = QVBoxLayout(grid_page)
        grid_layout.addWidget(QLabel("Kıyaslanacak modeller (1-3 seç):"))
        for name in MODEL_REGISTRY:
            cb = QCheckBox(name)
            self._model_checkboxes[name] = cb
            grid_layout.addWidget(cb)
        grid_layout.addStretch()
        self.model_stack.addWidget(grid_page)

        model_layout.addWidget(self.model_stack)
        layout.addWidget(model_box)
        model_box.setVisible(False)

        # Parametreler
        param_box = QGroupBox("Parametreler")
        param_layout = QVBoxLayout(param_box)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence"))
        conf_row.addStretch()
        self.conf_value = QLabel("0.25")
        self.conf_value.setStyleSheet("font-weight: bold;")
        conf_row.addWidget(self.conf_value)
        param_layout.addLayout(conf_row)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        param_layout.addWidget(self.conf_slider)

        skip_row = QHBoxLayout()
        skip_row.addWidget(QLabel("Skip frames"))
        skip_row.addStretch()
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(1, 10)
        self.skip_spin.setValue(2)
        skip_row.addWidget(self.skip_spin)
        param_layout.addLayout(skip_row)

        self.save_checkbox = QCheckBox("Tespit edilen kareleri kaydet (./output)")
        param_layout.addWidget(self.save_checkbox)
        layout.addWidget(param_box)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_content(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        toolbar = QHBoxLayout()
        title_col = QVBoxLayout()
        title = QLabel("Araç Hasar Tespiti")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        subtitle = QLabel("Broken Glass · Dent · Scratch · Wreck | Tıkla ve kilitle")
        subtitle.setStyleSheet("font-size: 12px; color: #888;")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        toolbar.addLayout(title_col)
        toolbar.addStretch()
        self.device_badge = QLabel("DEVICE: -")
        self.device_badge.setStyleSheet("font-weight: bold; font-size: 11px;")
        toolbar.addWidget(self.device_badge)
        self.start_button = QPushButton("Başlat")
        self.start_button.clicked.connect(self._start_capture)
        self.stop_button = QPushButton("Durdur")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_capture)
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.stop_button)
        layout.addLayout(toolbar)

        self.view_stack = QStackedWidget()

        single_page = QWidget()
        single_layout = QVBoxLayout(single_page)
        single_layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = ScaledVideoLabel()
        self.video_label.clicked.connect(self._on_video_click)
        single_layout.addWidget(self.video_label)
        self.view_stack.addWidget(single_page)

        grid_page = QWidget()
        grid_outer = QVBoxLayout(grid_page)
        grid_outer.setContentsMargins(0, 0, 0, 0)
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(8)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_scroll.setWidget(self.grid_container)
        grid_outer.addWidget(self.grid_scroll)
        self.view_stack.addWidget(grid_page)

        image_grid_page = QWidget()
        image_grid_outer = QVBoxLayout(image_grid_page)
        image_grid_outer.setContentsMargins(0, 0, 0, 0)
        self.img_scroll = QScrollArea()
        self.img_scroll.setWidgetResizable(True)
        self.img_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.img_grid_container = QWidget()
        self.img_grid_layout = QGridLayout(self.img_grid_container)
        self.img_grid_layout.setSpacing(10)
        self.img_grid_layout.setContentsMargins(4, 4, 4, 4)
        self.img_grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.img_scroll.setWidget(self.img_grid_container)
        image_grid_outer.addWidget(self.img_scroll)
        self.view_stack.addWidget(image_grid_page)

        # --- lightbox / detail page ---
        detail_page = QWidget()
        detail_outer = QVBoxLayout(detail_page)
        detail_outer.setContentsMargins(0, 0, 0, 0)
        detail_outer.setSpacing(0)

        # top bar
        detail_topbar = QWidget()
        detail_topbar.setFixedHeight(44)
        detail_topbar.setStyleSheet("background-color: #181818; border-bottom: 1px solid #2a2a2a;")
        topbar_row = QHBoxLayout(detail_topbar)
        topbar_row.setContentsMargins(10, 0, 10, 0)
        self.detail_back_btn = QPushButton("< Geri")
        self.detail_back_btn.setFixedWidth(80)
        self.detail_back_btn.clicked.connect(self._close_image_detail)
        self.detail_filename_lbl = QLabel("")
        self.detail_filename_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #ddd;")
        self.detail_prev_btn = QPushButton("<")
        self.detail_prev_btn.setFixedWidth(36)
        self.detail_prev_btn.clicked.connect(self._detail_prev)
        self.detail_next_btn = QPushButton(">")
        self.detail_next_btn.setFixedWidth(36)
        self.detail_next_btn.clicked.connect(self._detail_next)
        self.detail_pos_lbl = QLabel("")
        self.detail_pos_lbl.setStyleSheet("font-size: 11px; color: #888;")
        self.detail_boxes_cb = QCheckBox("Kutular")
        self.detail_boxes_cb.setChecked(True)
        self.detail_boxes_cb.toggled.connect(self._on_detail_toggle)
        self.detail_masks_cb = QCheckBox("Segmentler")
        self.detail_masks_cb.setChecked(True)
        self.detail_masks_cb.toggled.connect(self._on_detail_toggle)
        topbar_row.addWidget(self.detail_back_btn)
        topbar_row.addWidget(self.detail_filename_lbl)
        topbar_row.addStretch()
        topbar_row.addWidget(self.detail_boxes_cb)
        topbar_row.addWidget(self.detail_masks_cb)
        topbar_row.addWidget(self.detail_pos_lbl)
        topbar_row.addWidget(self.detail_prev_btn)
        topbar_row.addWidget(self.detail_next_btn)
        detail_outer.addWidget(detail_topbar)

        # body: large image + right detection panel
        detail_body = QWidget()
        detail_body_row = QHBoxLayout(detail_body)
        detail_body_row.setContentsMargins(8, 8, 8, 8)
        detail_body_row.setSpacing(10)

        self.detail_img_label = QLabel()
        self.detail_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_img_label.setStyleSheet("background-color: #0d0d0d; border-radius: 8px;")
        self.detail_img_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.detail_img_label.setMinimumSize(400, 300)
        detail_body_row.addWidget(self.detail_img_label, stretch=1)

        right_panel = QWidget()
        right_panel.setFixedWidth(220)
        right_panel.setStyleSheet("background-color: #1a1a1a; border-radius: 8px;")
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(10, 10, 10, 10)
        right_panel_layout.setSpacing(6)
        det_header = QLabel("Tespitler")
        det_header.setStyleSheet("font-size: 13px; font-weight: bold; color: #ccc;")
        right_panel_layout.addWidget(det_header)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #333;")
        sep.setFixedHeight(1)
        right_panel_layout.addWidget(sep)
        self.detail_det_scroll = QScrollArea()
        self.detail_det_scroll.setWidgetResizable(True)
        self.detail_det_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.detail_det_scroll.setStyleSheet("background: transparent;")
        self.detail_det_container = QWidget()
        self.detail_det_container.setStyleSheet("background: transparent;")
        self.detail_det_layout = QVBoxLayout(self.detail_det_container)
        self.detail_det_layout.setContentsMargins(0, 0, 0, 0)
        self.detail_det_layout.setSpacing(4)
        self.detail_det_layout.addStretch()
        self.detail_det_scroll.setWidget(self.detail_det_container)
        right_panel_layout.addWidget(self.detail_det_scroll, stretch=1)
        detail_body_row.addWidget(right_panel)
        detail_outer.addWidget(detail_body, stretch=1)

        # bottom strip
        detail_strip_wrapper = QWidget()
        detail_strip_wrapper.setFixedHeight(self._STRIP_THUMB_H + 20)
        detail_strip_wrapper.setStyleSheet("background-color: #111; border-top: 1px solid #2a2a2a;")
        strip_wrapper_layout = QHBoxLayout(detail_strip_wrapper)
        strip_wrapper_layout.setContentsMargins(6, 6, 6, 6)
        self.detail_strip_scroll = QScrollArea()
        self.detail_strip_scroll.setWidgetResizable(True)
        self.detail_strip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.detail_strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.detail_strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.detail_strip_container = QWidget()
        self.detail_strip_layout = QHBoxLayout(self.detail_strip_container)
        self.detail_strip_layout.setContentsMargins(0, 0, 0, 0)
        self.detail_strip_layout.setSpacing(6)
        self.detail_strip_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.detail_strip_scroll.setWidget(self.detail_strip_container)
        strip_wrapper_layout.addWidget(self.detail_strip_scroll)
        detail_outer.addWidget(detail_strip_wrapper)

        self.view_stack.addWidget(detail_page)

        layout.addWidget(self.view_stack, stretch=1)
        return container

    def _build_inspector(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(380)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(14)

        # Odak nesne
        self.focus_box = QGroupBox("Odak Nesne")
        focus_layout = QVBoxLayout(self.focus_box)
        self.focus_empty = QLabel("Nesneye tıklayarak kilitleyin")
        self.focus_empty.setWordWrap(True)
        focus_layout.addWidget(self.focus_empty)

        self.focus_form = QWidget()
        form = QFormLayout(self.focus_form)
        self.focus_fields: dict[str, QLabel] = {}
        self.color_swatch = QFrame()
        self.color_swatch.setFixedSize(28, 28)
        self.color_swatch.setStyleSheet(
            "background-color: #888; border: 1px solid #555; border-radius: 6px;"
        )
        rows = [
            ("Sınıf", "label"),
            ("Güven", "confidence"),
            ("Takip ID", "track_id"),
            ("Boyut", "size"),
            ("En/Boy", "aspect"),
            ("Konum", "position"),
            ("Renk", "color"),
            ("Süre", "duration"),
            ("Yön", "direction"),
        ]
        for key_text, key in rows:
            if key == "color":
                color_row = QHBoxLayout()
                color_row.addWidget(self.color_swatch)
                val = QLabel("-")
                self.focus_fields[key] = val
                color_row.addWidget(val)
                color_row.addStretch()
                color_widget = QWidget()
                color_widget.setLayout(color_row)
                form.addRow(key_text, color_widget)
            else:
                val = QLabel("-")
                self.focus_fields[key] = val
                form.addRow(key_text, val)
        self.focus_form.setVisible(False)
        focus_layout.addWidget(self.focus_form)

        self.unlock_button = QPushButton("Kilidi bırak")
        self.unlock_button.setEnabled(False)
        self.unlock_button.clicked.connect(self._unlock_focus)
        focus_layout.addWidget(self.unlock_button)
        layout.addWidget(self.focus_box)

        # Canli durum
        stats_box = QGroupBox("Canlı Durum")
        stats_layout = QHBoxLayout(stats_box)
        fps_col = QVBoxLayout()
        self.fps_label = QLabel("0")
        self.fps_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        fps_col.addWidget(self.fps_label)
        fps_col.addWidget(QLabel("MODEL FPS"))
        count_col = QVBoxLayout()
        self.count_label = QLabel("0")
        self.count_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        count_col.addWidget(self.count_label)
        count_col.addWidget(QLabel("TESPİT"))
        stats_layout.addLayout(fps_col)
        stats_layout.addLayout(count_col)
        layout.addWidget(stats_box)

        # Log
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(120)
        self.log_output.setStyleSheet("font-family: monospace; font-size: 11px;")
        log_layout.addWidget(self.log_output)
        layout.addWidget(log_box)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_view_changed(self, view: int) -> None:
        is_single = view == self.VIEW_SINGLE
        self.view_stack.setCurrentIndex(view)
        self.model_stack.setCurrentIndex(0 if is_single else 1)
        self.focus_box.setVisible(is_single)

    def _on_source_changed(self, idx: int) -> None:
        self.source_stack.setCurrentIndex(idx)
        if idx == self.SRC_IMAGE:
            cur = self.view_stack.currentIndex()
            if cur not in (self.VIEW_IMAGE, self.VIEW_IMAGE_DETAIL):
                self.view_stack.setCurrentIndex(self.VIEW_IMAGE)
            self.skip_spin.setEnabled(False)
        else:
            if self.view_stack.currentIndex() in (self.VIEW_IMAGE, self.VIEW_IMAGE_DETAIL):
                self.view_stack.setCurrentIndex(self.VIEW_SINGLE)
            self.skip_spin.setEnabled(True)

    def _on_conf_changed(self, value: int) -> None:
        self.conf_value.setText(f"{value / 100.0:.2f}")

    def _refresh_cameras(self) -> None:
        self.camera_combo.clear()
        self._camera_indices.clear()
        cameras = list_cameras()
        for cam in cameras:
            self.camera_combo.addItem(cam.label)
            self._camera_indices.append(cam.index)
        self._log(f"{len(cameras)} kamera bulundu.")

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Video seç",
            "",
            "Video (*.mp4 *.mov *.avi *.mkv);;Tüm dosyalar (*)",
        )
        if path:
            self._video_file_path = path
            self.file_field.setText(path)

    def _browse_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Resim seç",
            "",
            "Resim (*.jpg *.jpeg *.png *.bmp *.webp);;Tüm dosyalar (*)",
        )
        if not paths:
            return
        self._image_paths = paths
        count = len(paths)
        self.image_count_label.setText(f"{count} resim secildi")
        self._log(f"{count} resim secildi.")
        self._populate_image_grid()

    def _populate_image_grid(self) -> None:
        _clear_layout(self.img_grid_layout)
        self._image_cards = []

        for i, path in enumerate(self._image_paths):
            card = ClickableCard(i)
            card.clicked.connect(self._show_image_detail)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(6, 6, 6, 6)
            card_layout.setSpacing(4)

            img_label = QLabel()
            img_label.setFixedSize(self._IMG_THUMB_W, self._IMG_THUMB_H)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setStyleSheet("background-color: #111; border-radius: 4px; border: none;")

            orig_pixmap = QPixmap(path)
            if not orig_pixmap.isNull():
                img_label.setPixmap(
                    scale_pixmap_to_fit(orig_pixmap, self._IMG_THUMB_W, self._IMG_THUMB_H)
                )
            else:
                orig_pixmap = QPixmap()
                img_label.setText("Yuklenemedi")

            name_label = QLabel(os.path.basename(path))
            name_label.setStyleSheet("font-size: 11px; color: #aaa; border: none;")
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setWordWrap(True)

            badge_label = QLabel("")
            badge_label.setFixedHeight(20)
            badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge_label.setStyleSheet("font-size: 11px; font-weight: bold; border: none;")

            card_layout.addWidget(img_label)
            card_layout.addWidget(name_label)
            card_layout.addWidget(badge_label)

            row = i // self._IMG_COLS
            col = i % self._IMG_COLS
            self.img_grid_layout.addWidget(card, row, col)

            self._image_cards.append({
                "img_label": img_label,
                "badge_label": badge_label,
                "orig_pixmap": orig_pixmap,
                "annotated_pixmap": None,
                "legend_data": None,
                "count": 0,
                "path": path,
            })

        for c in range(self._IMG_COLS):
            self.img_grid_layout.setColumnStretch(c, 1)

    def _run_image_inference(self) -> None:
        if not self._image_paths:
            self._alert("Resim yok", "Once resim secin.")
            return
        backends = self._resolve_backends()
        if not backends:
            return
        backend = backends[0]
        conf = self.conf_slider.value() / 100.0

        self._image_backend = backend
        self._image_worker = ImageInferenceWorker(self._image_paths, backend, conf)
        self._image_worker.result_ready.connect(self._on_image_result)
        self._image_worker.finished.connect(self._on_image_inference_finished)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self._set_inputs_enabled(False)
        self._log(f"Resim modu: {len(self._image_paths)} resim isleniyor...")
        self._image_worker.start()

    def _on_image_result(self, index: int, frame, count: int, legend_data, raw_frame, raw_detections) -> None:
        if index >= len(self._image_cards):
            return
        card = self._image_cards[index]
        img_label: QLabel = card["img_label"]
        badge_label: QLabel = card["badge_label"]

        annotated_pixmap: QPixmap | None = None
        if frame is not None:
            annotated_pixmap = frame_to_pixmap(frame)
            img_label.setPixmap(
                scale_pixmap_to_fit(annotated_pixmap, self._IMG_THUMB_W, self._IMG_THUMB_H)
            )

        card["annotated_pixmap"] = annotated_pixmap
        card["legend_data"] = legend_data
        card["count"] = count
        card["raw_frame"] = raw_frame
        card["raw_detections"] = raw_detections

        if count > 0:
            color = "#e05050" if count >= 3 else "#e0a020"
            badge_label.setText(f"{count} tespit")
            badge_label.setStyleSheet(
                f"font-size: 11px; font-weight: bold; color: {color}; border: none;"
            )
        else:
            badge_label.setText("Tespit yok")
            badge_label.setStyleSheet("font-size: 11px; color: #666; border: none;")

        # strip thumbnail guncelle (detail aciksa)
        if index < len(self._strip_thumb_labels) and annotated_pixmap:
            thumb = self._strip_thumb_labels[index]
            thumb.setPixmap(
                scale_pixmap_to_fit(annotated_pixmap, self._STRIP_THUMB_W, self._STRIP_THUMB_H)
            )

        # detay sayfasi bu resmi gosteriyorsa guncelle
        if (
            self.view_stack.currentIndex() == self.VIEW_IMAGE_DETAIL
            and self._detail_index == index
        ):
            self._update_detail_view(index)

    def _on_image_inference_finished(self) -> None:
        self.start_button.setEnabled(True)
        self._set_inputs_enabled(True)
        self._log("Resim modu tamamlandi.")

    # ------------------------------------------------------------------
    # Lightbox / detail view
    # ------------------------------------------------------------------
    def _show_image_detail(self, index: int) -> None:
        if not self._image_cards:
            return
        self._build_detail_strip()
        self._update_detail_view(index)
        self.view_stack.setCurrentIndex(self.VIEW_IMAGE_DETAIL)

    def _close_image_detail(self) -> None:
        self.view_stack.setCurrentIndex(self.VIEW_IMAGE)

    def _detail_prev(self) -> None:
        if self._detail_index > 0:
            self._update_detail_view(self._detail_index - 1)

    def _detail_next(self) -> None:
        if self._detail_index < len(self._image_cards) - 1:
            self._update_detail_view(self._detail_index + 1)

    def _build_detail_strip(self) -> None:
        _clear_layout(self.detail_strip_layout)
        self._strip_thumb_labels = []
        for i, card in enumerate(self._image_cards):
            thumb = QLabel()
            thumb.setFixedSize(self._STRIP_THUMB_W, self._STRIP_THUMB_H)
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb.setCursor(Qt.CursorShape.PointingHandCursor)
            pixmap = card.get("annotated_pixmap") or card.get("orig_pixmap") or QPixmap()
            if not pixmap.isNull():
                thumb.setPixmap(scale_pixmap_to_fit(pixmap, self._STRIP_THUMB_W, self._STRIP_THUMB_H))
            thumb.setStyleSheet(
                "background-color: #222; border-radius: 4px; border: 2px solid transparent;"
            )
            idx = i
            thumb.mousePressEvent = lambda ev, j=idx: self._update_detail_view(j)  # type: ignore[method-assign]
            self.detail_strip_layout.addWidget(thumb)
            self._strip_thumb_labels.append(thumb)

    def _render_detail_image(self, index: int) -> None:
        if index < 0 or index >= len(self._image_cards):
            return
        card = self._image_cards[index]
        raw_frame = card.get("raw_frame")
        raw_detections = card.get("raw_detections")
        show_boxes = self.detail_boxes_cb.isChecked()
        show_masks = self.detail_masks_cb.isChecked()

        if raw_frame is not None and raw_detections is not None and self._image_backend is not None:
            if not show_boxes and not show_masks:
                pixmap = card.get("orig_pixmap") or QPixmap()
            else:
                annotated = self._image_backend.annotate(
                    raw_frame, raw_detections,
                    show_boxes=show_boxes, show_masks=show_masks,
                )
                pixmap = frame_to_pixmap(annotated)
        else:
            pixmap = card.get("annotated_pixmap") or card.get("orig_pixmap") or QPixmap()

        if pixmap and not pixmap.isNull():
            vw = self.detail_img_label.width() or 800
            vh = self.detail_img_label.height() or 600
            self.detail_img_label.setPixmap(scale_pixmap_to_fit(pixmap, vw, vh))

    def _on_detail_toggle(self) -> None:
        if self._detail_index >= 0:
            self._render_detail_image(self._detail_index)

    def _update_detail_view(self, index: int) -> None:
        if index < 0 or index >= len(self._image_cards):
            return
        self._detail_index = index
        card = self._image_cards[index]

        # top bar
        self.detail_filename_lbl.setText(os.path.basename(card["path"]))
        total = len(self._image_cards)
        self.detail_pos_lbl.setText(f"{index + 1} / {total}")
        self.detail_prev_btn.setEnabled(index > 0)
        self.detail_next_btn.setEnabled(index < total - 1)

        # main image (toggle-aware)
        self._render_detail_image(index)

        # detection panel
        _clear_layout(self.detail_det_layout)
        legend_data = card.get("legend_data")
        if legend_data and legend_data.get("summary"):
            for entry in legend_data["summary"]:
                row_w = QWidget()
                row_w.setStyleSheet("background: transparent;")
                row = QHBoxLayout(row_w)
                row.setContentsMargins(0, 2, 0, 2)
                row.setSpacing(6)
                dot = QFrame()
                dot.setFixedSize(8, 8)
                accent = legend_data.get("accent_hex", "#2F6DF6")
                dot.setStyleSheet(f"background-color: {accent}; border-radius: 4px;")
                name_lbl = QLabel(entry["name"])
                name_lbl.setStyleSheet("color: #ddd; font-size: 12px; background: transparent;")
                count_lbl = QLabel(f"x{entry['count']}")
                count_lbl.setStyleSheet("color: #aaa; font-size: 11px; background: transparent;")
                conf_lbl = QLabel(f"{entry['max_conf']:.0%}")
                conf_lbl.setStyleSheet("color: #888; font-size: 11px; background: transparent;")
                row.addWidget(dot, alignment=Qt.AlignmentFlag.AlignVCenter)
                row.addWidget(name_lbl)
                row.addStretch()
                row.addWidget(count_lbl)
                row.addWidget(conf_lbl)
                self.detail_det_layout.addWidget(row_w)
            self.detail_det_layout.addStretch()
        else:
            empty_lbl = QLabel("Henuz islenmedi" if not legend_data else "Tespit yok")
            empty_lbl.setStyleSheet("color: #666; font-size: 12px; font-style: italic; background: transparent;")
            self.detail_det_layout.addWidget(empty_lbl)
            self.detail_det_layout.addStretch()

        # strip highlight
        for i, lbl in enumerate(self._strip_thumb_labels):
            if i == index:
                lbl.setStyleSheet(
                    "background-color: #2a2a2a; border-radius: 4px; border: 2px solid #4a90e2;"
                )
            else:
                lbl.setStyleSheet(
                    "background-color: #222; border-radius: 4px; border: 2px solid transparent;"
                )

        # scroll strip to show current
        if index < len(self._strip_thumb_labels):
            self.detail_strip_scroll.ensureWidgetVisible(self._strip_thumb_labels[index])

    def _on_video_click(self, wx: int, wy: int) -> None:
        if not self.bridge.engine.is_running():
            return
        if self.view_combo.currentIndex() != self.VIEW_SINGLE:
            return
        vw = self.video_label.width()
        vh = self.video_label.height()
        coords = widget_to_frame(wx, wy, vw, vh, self._frame_size, self._display_rect)
        if coords is None:
            return
        fx, fy = coords
        self.bridge.engine.request_focus_at(fx, fy)
        self._log(f"Odak isteği: ({fx}, {fy})")

    def _unlock_focus(self) -> None:
        self.bridge.engine.set_focus(None)
        self._reset_focus_panel()

    def _resolve_source(self) -> str | None:
        source_type = self.source_combo.currentIndex()
        if source_type == self.SRC_CAMERA:
            idx = self.camera_combo.currentIndex()
            if idx < 0 or idx >= len(self._camera_indices):
                return None
            return str(self._camera_indices[idx])
        if source_type == self.SRC_FILE:
            return self._video_file_path or None
        if source_type == self.SRC_RTSP:
            text = self.rtsp_field.text().strip()
            return text or None
        return None

    def _resolve_backends(self) -> list | None:
        if self.view_combo.currentIndex() == self.VIEW_SINGLE:
            name = self.model_combo.currentText()
            return [create_backend(name)]
        selected = [
            name for name, cb in self._model_checkboxes.items() if cb.isChecked()
        ]
        if len(selected) < 1:
            self._alert("Model seçilmedi", "En az 1 model seçin.")
            return None
        if len(selected) > 3:
            self._alert("Çok fazla", "En fazla 3 model seçin.")
            return None
        return [create_backend(name) for name in selected]

    def _clear_grid_panels(self) -> None:
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for c in range(3):
            self.grid_layout.setColumnStretch(c, 0)
        self.grid_layout.setRowStretch(0, 0)
        self._grid_panels.clear()

    def _build_grid_panels(self, model_ids: list[str]) -> None:
        self._clear_grid_panels()
        cols = len(model_ids)
        for i, mid in enumerate(model_ids):
            panel = ModelGridPanel(mid)
            self.grid_layout.addWidget(panel, 0, i)
            self._grid_panels[mid] = panel
        for c in range(cols):
            self.grid_layout.setColumnStretch(c, 1)
        self.grid_layout.setRowStretch(0, 1)

    def _start_capture(self) -> None:
        if self.source_combo.currentIndex() == self.SRC_IMAGE:
            self._run_image_inference()
            return
        if self.bridge.engine.is_running():
            return
        source = self._resolve_source()
        if not source:
            self._alert("Kaynak yok", "Lütfen geçerli bir kaynak seçin veya girin.")
            return
        backends = self._resolve_backends()
        if not backends:
            return
        if self.view_combo.currentIndex() == self.VIEW_GRID:
            self._build_grid_panels([b.model_id for b in backends])
        self.bridge.engine.configure(
            backends=backends,
            source=source,
            skip_frames=self.skip_spin.value(),
            conf=self.conf_slider.value() / 100.0,
            save_frames=self.save_checkbox.isChecked(),
            save_dir="./output",
        )
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_inputs_enabled(False)
        self._reset_focus_panel()
        self._log(f"Yakalama başlatılıyor: {source} | {[b.model_id for b in backends]}")
        self.bridge.engine.start()

    def _stop_capture(self) -> None:
        if self.bridge.engine.is_running():
            self.bridge.engine.stop()
            self._log("Durduruluyor...")

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.view_combo.setEnabled(enabled)
        self.source_combo.setEnabled(enabled)
        self.camera_combo.setEnabled(enabled)
        self.refresh_button.setEnabled(enabled)
        self.rtsp_field.setEnabled(enabled)
        self.file_field.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.image_browse_button.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        for cb in self._model_checkboxes.values():
            cb.setEnabled(enabled)
        self.conf_slider.setEnabled(enabled)
        is_image_mode = self.source_combo.currentIndex() == self.SRC_IMAGE
        self.skip_spin.setEnabled(enabled and not is_image_mode)
        self.save_checkbox.setEnabled(enabled)

    def _reset_focus_panel(self) -> None:
        self.focus_empty.setText("Nesneye tıklayarak kilitleyin")
        self.focus_empty.setVisible(True)
        self.focus_form.setVisible(False)
        self.unlock_button.setEnabled(False)
        for field in self.focus_fields.values():
            field.setText("-")
        self.color_swatch.setStyleSheet(
            "background-color: #888; border: 1px solid #555; border-radius: 6px;"
        )

    # ------------------------------------------------------------------
    # Engine callbacks (via CaptureBridge signals, main thread)
    # ------------------------------------------------------------------
    def _on_panel_ready(
        self, model_id: str, frame: np.ndarray, latency_ms: float, count: int, legend_data: dict
    ) -> None:
        model_fps = 1000.0 / max(latency_ms, 1.0)
        self.fps_label.setText(f"{model_fps:.0f}")
        self.count_label.setText(str(count))
        if self.view_combo.currentIndex() == self.VIEW_SINGLE:
            h, w = frame.shape[:2]
            self._frame_size = (w, h)
            self._display_rect = self.video_label.set_frame(frame)
            self.video_label.set_legend_data(legend_data)
        else:
            panel = self._grid_panels.get(model_id)
            if panel:
                panel.update_frame(frame, latency_ms, count, legend_data)

    def _on_focus_info(self, info: dict | None) -> None:
        if info is None:
            self._reset_focus_panel()
            return
        if info.get("status") == "kayip":
            self.focus_empty.setText(f"Odak kayboldu (ID: {info.get('track_id', '-')})")
            self.focus_empty.setVisible(True)
            self.focus_form.setVisible(False)
            self.unlock_button.setEnabled(True)
            return
        self.focus_empty.setVisible(False)
        self.focus_form.setVisible(True)
        self.unlock_button.setEnabled(True)
        self.focus_fields["label"].setText(str(info.get("label", "-")))
        conf = info.get("confidence")
        self.focus_fields["confidence"].setText(
            f"{conf:.0%}" if conf is not None else "-"
        )
        self.focus_fields["track_id"].setText(str(info.get("track_id", "-")))
        self.focus_fields["size"].setText(
            f"{info.get('width', '-')} x {info.get('height', '-')} px"
        )
        self.focus_fields["aspect"].setText(str(info.get("aspect_ratio", "-")))
        self.focus_fields["position"].setText(
            f"({info.get('center_x', '-')}, {info.get('center_y', '-')})"
        )
        color_bgr = info.get("color_bgr")
        if color_bgr:
            b, g, r = color_bgr
            self.color_swatch.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); "
                "border: 1px solid #555; border-radius: 6px;"
            )
            self.focus_fields["color"].setText(f"RGB({r}, {g}, {b})")
        else:
            self.color_swatch.setStyleSheet(
                "background-color: #888; border: 1px solid #555; border-radius: 6px;"
            )
            self.focus_fields["color"].setText("-")
        duration = info.get("duration_s")
        self.focus_fields["duration"].setText(
            f"{duration:.1f} sn" if duration is not None else "-"
        )
        self.focus_fields["direction"].setText(str(info.get("direction", "-")))

    def _on_capture_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_inputs_enabled(True)
        if self.view_combo.currentIndex() == self.VIEW_SINGLE:
            self.video_label.clear()
        self._frame_size = None
        self._display_rect = None
        self._reset_focus_panel()
        self._log("Yakalama durdu.")

    def _on_error(self, message: str) -> None:
        self._log(f"HATA: {message}")
        self._alert("Hata", message)

    def _set_device(self, device: str) -> None:
        self.device_badge.setText(f"DEVICE: {device.upper()}")

    def _log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def _alert(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def closeEvent(self, event) -> None:
        self._stop_capture()
        if self.bridge.engine.is_running():
            self.bridge.engine.join(timeout=3.0)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
