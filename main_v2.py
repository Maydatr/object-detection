"""
main_v2.py - Multi-photo vehicle damage inspection (Kaporta Kontrol Raporu).

Tek oturum = tek arac. N fotograf islenir, hasarlar panele atanir, kaporta semasi uzerinde rapor.
"""
from __future__ import annotations

import platform
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PyQt6.QtCore import QByteArray, QRectF, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPixmap
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from backends import UltralyticsBackend
from mask_intersection import (
    assign_damages_to_panels,
    build_part_mask_dict,
    detections_to_masks,
)
from session import (
    PANEL_STATUS_NO_DATA,
    PANEL_STATUS_OK,
    VehicleSession,
)
from sources import list_cameras
from ui_theme import APP_STYLESHEET, PREVIEW_STYLE, SCHEMA_BG_STYLE

PROJECT_ROOT = Path(__file__).resolve().parent
PANEL_CONFIG_PATH = PROJECT_ROOT / "panel_config.yaml"
PARTS_WEIGHT = PROJECT_ROOT / "weights" / "car-parts-seg.pt"
DAMAGE_WEIGHT = PROJECT_ROOT / "weights" / "car-damage-seg-v2.pt"

GALLERY_THUMB_W = 96
GALLERY_THUMB_H = 68
DAMAGE_ROW_H = 36

PHOTO_PENDING = "pending"
PHOTO_DONE = "done"
PHOTO_ERROR = "error"


def load_panel_config() -> dict[str, Any]:
    with open(PANEL_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    image = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def scale_pixmap(pixmap: QPixmap, w: int, h: int) -> QPixmap:
    return pixmap.scaled(
        w, h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _parse_svg_matrix(transform_str: str) -> tuple[float, float, float, float, float, float] | None:
    """Parse matrix(a,b,c,d,e,f) from a transform attribute string."""
    m = re.match(r'matrix\s*\(\s*([^)]+)\s*\)', transform_str or "")
    if not m:
        return None
    vals = [float(v) for v in re.split(r'[\s,]+', m.group(1).strip()) if v]
    if len(vals) == 6:
        return (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])
    return None


def _path_centroid(d_attr: str, transform_str: str | None) -> tuple[float, float, float] | None:
    """Compute approximate (cx, cy, radius) for an SVG path element."""
    nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', d_attr)]
    if len(nums) < 2:
        return None
    pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    if transform_str:
        mt = _parse_svg_matrix(transform_str)
        if mt:
            a, b, c, d, e, f = mt
            pairs = [(a * x + c * y + e, b * x + d * y + f) for x, y in pairs]
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    r = max((max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2, 10.0)
    return cx, cy, r


def _build_svg_hit_zones(
    svg_text: str,
    panel_elements: dict[str, list[str]],
) -> dict[str, list[tuple[float, float, float]]]:
    """panel_id -> list of (svg_x, svg_y, radius) clickable zones from mapped paths."""
    root = ET.fromstring(svg_text)
    elem_to_panel = {eid: pid for pid, eids in panel_elements.items() for eid in eids}
    zones: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for el in root.iter():
        eid = el.get("id")
        if not eid or eid not in elem_to_panel:
            continue
        tag = el.tag.split("}")[-1]
        panel_id = elem_to_panel[eid]

        if tag == "path":
            result = _path_centroid(el.get("d", ""), el.get("transform"))
            if result:
                zones[panel_id].append(result)

    return dict(zones)


class InspectionWorker(QThread):
    photo_done = pyqtSignal(str, object, object, object)
    photo_failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished_all = pyqtSignal()

    def __init__(
        self,
        paths: list[str],
        parts_backend: UltralyticsBackend,
        damage_backend: UltralyticsBackend,
        conf: float,
        min_overlap: float,
    ) -> None:
        super().__init__()
        self._paths = paths
        self._parts = parts_backend
        self._damage = damage_backend
        self._conf = conf
        self._min_overlap = min_overlap
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        total = len(self._paths)
        try:
            self.status.emit("Parca modeli yukleniyor...")
            self._parts.load()
            self.status.emit("Hasar modeli yukleniyor...")
            self._damage.load()
        except Exception as exc:
            self.error.emit(f"Model yukleme hatasi: {exc}")
            self.finished_all.emit()
            return

        done_count = 0
        for path in self._paths:
            if self._stop:
                break
            self.status.emit(f"Isleniyor: {Path(path).name}")
            frame = cv2.imread(path)
            if frame is None:
                self.error.emit(f"Goruntu okunamadi: {path}")
                self.photo_failed.emit(path)
                done_count += 1
                self.progress.emit(done_count, total)
                continue

            try:
                part_det = self._parts.infer(frame, self._conf)
                damage_det = self._damage.infer(frame, self._conf)
            except Exception as exc:
                self.error.emit(f"Inference hatasi ({Path(path).name}): {exc}")
                self.photo_failed.emit(path)
                done_count += 1
                self.progress.emit(done_count, total)
                continue

            part_masks_list = detections_to_masks(part_det, ignore_classes={"object"})
            damage_masks_list = detections_to_masks(damage_det)
            part_masks = build_part_mask_dict(part_masks_list)
            seen_panels = set(part_masks.keys())
            assignments = assign_damages_to_panels(
                damage_masks_list, part_masks, min_ratio=self._min_overlap
            )

            annotated = frame.copy()
            annotated = self._parts.annotate(annotated, part_det, show_boxes=False)
            annotated = self._damage.annotate(annotated, damage_det)

            self.photo_done.emit(path, annotated, assignments, seen_panels)
            done_count += 1
            self.progress.emit(done_count, total)

        self.finished_all.emit()


class StatusLegendWidget(QWidget):
    """Renk aciklamasi — yatay hizali."""

    def __init__(self, config: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 2)
        layout.setSpacing(16)

        colors = config.get("status_colors", {})
        labels = config.get("status_labels", {})
        order = ["ok", "light", "heavy", "no_data"]

        for key in order:
            if key not in colors:
                continue
            row = QHBoxLayout()
            row.setSpacing(6)
            dot = QFrame()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(
                f"background-color: {colors[key]}; border-radius: 5px; border: 1px solid #555;"
            )
            text = labels.get(key, key)
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size: 11px; color: #aaa;")
            row.addWidget(dot)
            row.addWidget(lbl)
            layout.addLayout(row)

        layout.addStretch()


class GalleryThumb(QFrame):
    clicked = pyqtSignal(int)
    double_clicked = pyqtSignal(int)

    def __init__(
        self,
        index: int,
        thumb: QPixmap,
        status: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._index = index
        self._status = status
        self.setFixedSize(GALLERY_THUMB_W + 8, GALLERY_THUMB_H + 8)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(0)

        self._img = QLabel()
        self._img.setFixedSize(GALLERY_THUMB_W, GALLERY_THUMB_H)
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setStyleSheet("background-color: #111; border-radius: 4px;")
        if not thumb.isNull():
            self._img.setPixmap(scale_pixmap(thumb, GALLERY_THUMB_W, GALLERY_THUMB_H))

        lay.addWidget(self._img)
        self.set_status(status)
        self.set_selected(False)

    def set_pixmap(self, thumb: QPixmap) -> None:
        if not thumb.isNull():
            self._img.setPixmap(scale_pixmap(thumb, GALLERY_THUMB_W, GALLERY_THUMB_H))

    def set_status(self, status: str) -> None:
        self._status = status
        if status == PHOTO_DONE:
            accent = "#4CAF50"
        elif status == PHOTO_ERROR:
            accent = "#F44336"
        else:
            accent = "#555"
        self._img.setStyleSheet(
            f"background-color: #111; border-radius: 4px; border: 1px solid {accent};"
        )

    def set_selected(self, selected: bool) -> None:
        if selected:
            self.setStyleSheet(
                "background: #252525; border: 2px solid #2F6DF6; border-radius: 6px;"
            )
        else:
            self.setStyleSheet(
                "background: transparent; border: 2px solid transparent; border-radius: 6px;"
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self._index)
        super().mouseDoubleClickEvent(event)


class DamageRowWidget(QWidget):
    def __init__(self, color: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(DAMAGE_ROW_H)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(8)
        dot = QFrame()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background-color: {color}; border-radius: 4px;")
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 12px; color: #ddd;")
        lbl.setWordWrap(False)
        layout.addWidget(dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(lbl, stretch=1, alignment=Qt.AlignmentFlag.AlignVCenter)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(0, DAMAGE_ROW_H)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(0, DAMAGE_ROW_H)


class SchemaWidget(QWidget):
    panel_clicked = pyqtSignal(str)
    wave_finished = pyqtSignal()

    HIGHLIGHT_STROKE = "#FFD54F"
    HIGHLIGHT_WIDTH = 3.0
    WAVE_COLOR = "#29B6F6"
    WAVE_INTERVAL_MS = 380

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = load_panel_config()
        self._panel_elements: dict[str, list[str]] = self._config.get("panel_elements", {})
        self._panel_labels: dict[str, str] = self._config.get("panel_labels", {})

        schema_svg_rel = self._config.get("schema_svg", "car_svg/car.svg")
        schema_svg_path = PROJECT_ROOT / schema_svg_rel
        self._template_svg = schema_svg_path.read_text(encoding="utf-8")

        self._vb_w, self._vb_h = self._parse_viewbox(self._template_svg)
        self._renderer = QSvgRenderer()
        self._panel_status: dict[str, str] = {}
        self._damage_counts: dict[str, int] = {}
        self._panels_seen: set[str] = set()
        self._highlight_panel: str | None = None
        self._hover_panel: str | None = None
        self._mouse_x = 0.0
        self._mouse_y = 0.0
        self._hit_zones = _build_svg_hit_zones(self._template_svg, self._panel_elements)
        self._wave_panels = [
            pid for pid in self._config.get("schema_panels", [])
            if self._panel_elements.get(pid)
        ]
        self._wave_active = False
        self._wave_index = 0
        self._wave_panel: str | None = None
        self._wave_timer = QTimer(self)
        self._wave_timer.timeout.connect(self._wave_step)
        self.setMinimumSize(540, 324)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self._refresh_renderer()

    @staticmethod
    def _parse_viewbox(svg_text: str) -> tuple[float, float]:
        """Return (width, height) from the SVG root viewBox attribute."""
        m = re.search(r'viewBox\s*=\s*["\']([^"\']+)["\']', svg_text)
        if m:
            parts = re.split(r'[\s,]+', m.group(1).strip())
            if len(parts) == 4:
                return float(parts[2]), float(parts[3])
        return 1200.0, 720.0

    def _set_element_fill(self, root: ET.Element, element_id: str, color: str) -> None:
        for el in root.iter():
            if el.get("id") == element_id:
                style = el.get("style", "")
                if "fill:" in style:
                    el.set("style", re.sub(r"fill:[^;]+", f"fill:{color}", style))
                else:
                    el.set("fill", color)

    def _set_element_stroke(
        self, root: ET.Element, element_id: str, color: str, width: float
    ) -> None:
        for el in root.iter():
            if el.get("id") == element_id:
                style = el.get("style", "")
                if "stroke:" in style:
                    new_style = re.sub(r"stroke:[^;]+", f"stroke:{color}", style)
                    new_style = re.sub(
                        r"stroke-width:[^;]+", f"stroke-width:{width}", new_style
                    )
                    el.set("style", new_style)
                else:
                    el.set("stroke", color)
                    el.set("stroke-width", str(width))

    def _build_colored_svg(self, panel_status: dict[str, str]) -> bytes:
        root = ET.fromstring(self._template_svg)
        schema_panels = self._config.get("schema_panels", [])
        colors = self._config["status_colors"]

        for panel_id in schema_panels:
            if self._wave_panel and panel_id == self._wave_panel:
                color = self.WAVE_COLOR
            else:
                status = panel_status.get(panel_id, PANEL_STATUS_NO_DATA)
                color = colors.get(status, colors["no_data"])
            targets = self._panel_elements.get(panel_id, [panel_id])
            for eid in targets:
                self._set_element_fill(root, eid, color)

        stroke_panel = self._highlight_panel
        if stroke_panel and not (self._wave_active and stroke_panel == self._wave_panel):
            targets = self._panel_elements.get(
                stroke_panel, [stroke_panel]
            )
            for eid in targets:
                self._set_element_stroke(
                    root, eid, self.HIGHLIGHT_STROKE, self.HIGHLIGHT_WIDTH
                )

        return ET.tostring(root, encoding="utf-8")

    def start_wave(self) -> None:
        if self._wave_active or not self._wave_panels:
            return
        self._wave_active = True
        self._wave_index = 0
        self._wave_panel = self._wave_panels[0]
        self._refresh_renderer()
        self._wave_timer.start(self.WAVE_INTERVAL_MS)

    def stop_wave(self) -> None:
        self._wave_timer.stop()
        self._wave_active = False
        self._wave_panel = None
        self._refresh_renderer()

    def _wave_step(self) -> None:
        self._wave_index += 1
        if self._wave_index >= len(self._wave_panels):
            self.stop_wave()
            self.wave_finished.emit()
            return
        self._wave_panel = self._wave_panels[self._wave_index]
        self._refresh_renderer()

    def _refresh_renderer(self) -> None:
        svg_bytes = self._build_colored_svg(self._panel_status)
        self._renderer.load(QByteArray(svg_bytes))
        self.update()

    def set_highlight(self, panel_id: str | None) -> None:
        if self._highlight_panel == panel_id:
            return
        self._highlight_panel = panel_id
        self._refresh_renderer()

    def apply_session(self, session: VehicleSession) -> None:
        schema_panels = self._config.get("schema_panels", [])
        self._panel_status = {
            pid: session.get_panel_status(pid) for pid in schema_panels
        }
        self._damage_counts = {
            pid: len(session.get_damages_for_panel(pid)) for pid in schema_panels
        }
        self._panels_seen = set(session.panels_seen)
        self._refresh_renderer()

    def _hover_text(self, panel_id: str) -> str | None:
        count = self._damage_counts.get(panel_id, 0)
        if count <= 0:
            return None
        label = self._panel_labels.get(panel_id, panel_id)
        return f"{label} - {count} hasar"

    def _draw_hover_overlay(self, painter: QPainter) -> None:
        if not self._hover_panel:
            return
        text = self._hover_text(self._hover_panel)
        if not text:
            return
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        fm = QFontMetrics(font)
        x = self._mouse_x + 12
        y = self._mouse_y - 8
        if x + fm.horizontalAdvance(text) > self.width() - 4:
            x = self._mouse_x - fm.horizontalAdvance(text) - 12
        if y - fm.height() < 4:
            y = self._mouse_y + fm.height() + 8
        painter.setPen(QColor("#e0e0e0"))
        painter.drawText(int(x), int(y), text)

    def _fit_rect(self) -> QRectF:
        size = self._renderer.defaultSize()
        if size.width() <= 0 or size.height() <= 0:
            return QRectF(self.rect())
        aspect = size.width() / size.height()
        avail_w = self.width()
        avail_h = self.height()
        if avail_w / avail_h > aspect:
            draw_h = avail_h
            draw_w = draw_h * aspect
        else:
            draw_w = avail_w
            draw_h = draw_w / aspect
        x = (avail_w - draw_w) / 2.0
        y = (avail_h - draw_h) / 2.0
        return QRectF(x, y, draw_w, draw_h)

    def _widget_to_svg(self, wx: float, wy: float) -> tuple[float, float]:
        rect = self._fit_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return -1.0, -1.0
        sx = (wx - rect.x()) / rect.width() * self._vb_w
        sy = (wy - rect.y()) / rect.height() * self._vb_h
        return sx, sy

    def _hit_test(self, svg_x: float, svg_y: float) -> str | None:
        best_panel: str | None = None
        best_dist = float("inf")
        for panel_id, circles in self._hit_zones.items():
            for cx, cy, r in circles:
                dist_sq = (svg_x - cx) ** 2 + (svg_y - cy) ** 2
                if dist_sq <= r * r and dist_sq < best_dist:
                    best_dist = dist_sq
                    best_panel = panel_id
        return best_panel

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        self._mouse_x = pos.x()
        self._mouse_y = pos.y()
        svg_x, svg_y = self._widget_to_svg(pos.x(), pos.y())
        panel = self._hit_test(svg_x, svg_y)
        if panel != self._hover_panel:
            self._hover_panel = panel
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover_panel is not None:
            self._hover_panel = None
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            svg_x, svg_y = self._widget_to_svg(pos.x(), pos.y())
            panel = self._hit_test(svg_x, svg_y)
            if panel:
                self.panel_clicked.emit(panel)
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._renderer.isValid():
            target = self._fit_rect()
            self._renderer.render(painter, target)
            self._draw_hover_overlay(painter)
        else:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sema yuklenemedi")
        painter.end()

    def export_png(self, path: str) -> bool:
        if not self._renderer.isValid():
            return False
        size = self._renderer.defaultSize()
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(pixmap)
        self._renderer.render(painter)
        painter.end()
        return pixmap.save(path)

    def export_pdf(self, path: str) -> bool:
        if not self._renderer.isValid():
            return False
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        painter = QPainter(printer)
        rect = painter.viewport()
        size = self._renderer.defaultSize()
        painter.setViewport(rect)
        painter.setWindow(0, 0, size.width(), size.height())
        self._renderer.render(painter)
        painter.end()
        return True


class MainWindow(QMainWindow):
    PAGE_MAIN = 0
    PAGE_DETAIL = 1

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kaporta Kontrol Raporu")
        self.setMinimumSize(1080, 680)
        self.resize(1440, 860)

        self._session = VehicleSession()
        self._pending_paths: list[str] = []
        self._photo_status: dict[str, str] = {}
        self._photo_thumbs: dict[str, QPixmap] = {}
        self._worker: InspectionWorker | None = None
        self._config = load_panel_config()
        self._conf = 0.25
        self._min_overlap = 0.10
        self._last_annotated: dict[str, np.ndarray] = {}
        self._legend_panel_ids: list[str] = []
        self._processing_total = 0
        self._processing_done = 0
        self._detail_index = -1
        self._preview_index = -1
        self._gallery_widgets: list[GalleryThumb] = []
        self._nav_lock = False

        self._build_ui()
        self._update_session_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        root.addWidget(self._build_sidebar(), stretch=0)
        root.addWidget(self._build_content(), stretch=1)
        root.addWidget(self._build_inspector(), stretch=0)

        self._legend.currentRowChanged.connect(self._on_legend_selected)
        self._schema.panel_clicked.connect(self._on_schema_panel_clicked)
        self._schema.wave_finished.connect(self._on_panel_wave_finished)

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self._status_bar_label = QLabel("Hazir")
        status_bar.addWidget(self._status_bar_label)

        self._check_weights()

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(360)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        session_grp = QGroupBox("Oturum")
        session_lay = QVBoxLayout(session_grp)
        self._lbl_session = QLabel()
        self._lbl_session.setWordWrap(True)
        self._lbl_session.setObjectName("mutedLabel")
        session_lay.addWidget(self._lbl_session)
        btn_new = QPushButton("Yeni Arac Oturumu")
        btn_new.clicked.connect(self._on_new_session)
        session_lay.addWidget(btn_new)
        btn_reset = QPushButton("Oturumu Sifirla")
        btn_reset.clicked.connect(self._on_reset_session)
        session_lay.addWidget(btn_reset)
        layout.addWidget(session_grp)

        photo_grp = QGroupBox("Fotograflar")
        photo_lay = QVBoxLayout(photo_grp)
        btn_files = QPushButton("Dosya Sec")
        btn_files.clicked.connect(self._on_add_files)
        btn_folder = QPushButton("Klasor Sec")
        btn_folder.clicked.connect(self._on_add_folder)
        btn_camera = QPushButton("Kameradan Cek")
        btn_camera.clicked.connect(self._on_camera_capture)
        src_row = QHBoxLayout()
        src_row.addWidget(btn_files)
        src_row.addWidget(btn_folder)
        photo_lay.addLayout(src_row)
        photo_lay.addWidget(btn_camera)
        self._sidebar_photo_label = QLabel("Henuz fotograf yok")
        self._sidebar_photo_label.setWordWrap(True)
        self._sidebar_photo_label.setObjectName("mutedLabel")
        photo_lay.addWidget(self._sidebar_photo_label)
        layout.addWidget(photo_grp)

        param_grp = QGroupBox("Parametreler")
        param_lay = QVBoxLayout(param_grp)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Guven esigi"))
        conf_row.addStretch()
        self._lbl_conf = QLabel("0.25")
        self._lbl_conf.setStyleSheet("font-weight: bold;")
        conf_row.addWidget(self._lbl_conf)
        param_lay.addLayout(conf_row)
        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(5, 95)
        self._conf_slider.setValue(25)
        self._conf_slider.valueChanged.connect(self._on_conf_changed)
        param_lay.addWidget(self._conf_slider)

        overlap_row = QHBoxLayout()
        overlap_row.addWidget(QLabel("Min overlap-ratio"))
        overlap_row.addStretch()
        self._lbl_overlap = QLabel("0.10")
        self._lbl_overlap.setStyleSheet("font-weight: bold;")
        overlap_row.addWidget(self._lbl_overlap)
        param_lay.addLayout(overlap_row)
        self._overlap_slider = QSlider(Qt.Orientation.Horizontal)
        self._overlap_slider.setRange(5, 50)
        self._overlap_slider.setValue(10)
        self._overlap_slider.valueChanged.connect(self._on_overlap_changed)
        param_lay.addWidget(self._overlap_slider)
        layout.addWidget(param_grp)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_content(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._view_stack = QStackedWidget()

        main_page = QWidget()
        main_lay = QVBoxLayout(main_page)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(12)

        toolbar = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        title = QLabel("Kaporta Kontrol Raporu")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Cok fotografli arac hasar muayenesi | Kaporta semasi")
        subtitle.setObjectName("pageSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        toolbar.addLayout(title_col)
        toolbar.addStretch()

        self._status_badge = QLabel("MODEL: -")
        self._status_badge.setObjectName("badgeLabel")
        toolbar.addWidget(self._status_badge)

        self._btn_cancel = QPushButton("Iptal")
        self._btn_cancel.setObjectName("dangerButton")
        self._btn_cancel.setVisible(False)
        self._btn_cancel.clicked.connect(self._on_cancel_processing)

        self._btn_process = QPushButton("Fotograflari Isle")
        self._btn_process.setObjectName("primaryButton")
        self._btn_process.clicked.connect(self._on_process_photos)

        toolbar.addWidget(self._btn_cancel)
        toolbar.addWidget(self._btn_process)
        main_lay.addLayout(toolbar)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        main_lay.addWidget(self._progress)

        split = QSplitter(Qt.Orientation.Vertical)

        schema_grp = QGroupBox("Kaporta Semasi")
        schema_grp_lay = QVBoxLayout(schema_grp)
        schema_grp_lay.setContentsMargins(6, 6, 6, 6)
        schema_scroll = QScrollArea()
        schema_scroll.setWidgetResizable(True)
        schema_scroll.setFrameShape(QFrame.Shape.NoFrame)
        schema_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        schema_scroll.setStyleSheet(SCHEMA_BG_STYLE)
        self._schema = SchemaWidget()
        schema_scroll.setWidget(self._schema)
        schema_grp_lay.addWidget(schema_scroll)
        schema_btn_row = QHBoxLayout()
        self._btn_panel_wave = QPushButton("Panel Kontrolu")
        self._btn_panel_wave.setToolTip("Tum panelleri sirayla yakip sondurur (test)")
        self._btn_panel_wave.clicked.connect(self._on_panel_wave_test)
        schema_btn_row.addStretch()
        schema_btn_row.addWidget(self._btn_panel_wave)
        schema_grp_lay.addLayout(schema_btn_row)
        self._status_legend = StatusLegendWidget(self._config)
        schema_grp_lay.addWidget(self._status_legend)
        split.addWidget(schema_grp)

        preview_grp = QGroupBox("Onizleme")
        preview_lay = QVBoxLayout(preview_grp)
        preview_lay.setContentsMargins(6, 6, 6, 6)
        preview_lay.setSpacing(8)

        preview_header = QHBoxLayout()
        self._preview_filename = QLabel("")
        self._preview_filename.setStyleSheet("font-size: 12px; color: #aaa;")
        preview_header.addWidget(self._preview_filename, stretch=1)
        self._btn_detail = QPushButton("Detay")
        self._btn_detail.setEnabled(False)
        self._btn_detail.clicked.connect(self._open_detail_for_current_photo)
        preview_header.addWidget(self._btn_detail)
        preview_lay.addLayout(preview_header)

        self._preview = QLabel("Fotograf secin veya ekleyin")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(180)
        self._preview.setStyleSheet(PREVIEW_STYLE)
        preview_lay.addWidget(self._preview, stretch=1)

        gallery_wrap = QWidget()
        gallery_wrap.setStyleSheet("background-color: #111; border-radius: 6px;")
        gallery_outer = QVBoxLayout(gallery_wrap)
        gallery_outer.setContentsMargins(6, 6, 6, 6)
        gallery_outer.setSpacing(4)
        gallery_title = QLabel("Galeri")
        gallery_title.setStyleSheet("font-size: 11px; color: #666;")
        gallery_outer.addWidget(gallery_title)

        self._gallery_scroll = QScrollArea()
        self._gallery_scroll.setWidgetResizable(True)
        self._gallery_scroll.setFixedHeight(GALLERY_THUMB_H + 28)
        self._gallery_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._gallery_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._gallery_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._gallery_scroll.setStyleSheet("background: transparent;")

        self._gallery_container = QWidget()
        self._gallery_layout = QHBoxLayout(self._gallery_container)
        self._gallery_layout.setContentsMargins(0, 0, 0, 0)
        self._gallery_layout.setSpacing(6)
        self._gallery_layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._gallery_scroll.setWidget(self._gallery_container)
        gallery_outer.addWidget(self._gallery_scroll)
        preview_lay.addWidget(gallery_wrap)

        split.addWidget(preview_grp)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([600, 240])
        main_lay.addWidget(split, stretch=1)

        self._view_stack.addWidget(main_page)

        detail_page = self._build_detail_page()
        self._view_stack.addWidget(detail_page)

        layout.addWidget(self._view_stack, stretch=1)
        return container

    def _build_detail_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        topbar = QWidget()
        topbar.setFixedHeight(44)
        topbar.setStyleSheet("background-color: #181818; border-bottom: 1px solid #2a2a2a;")
        top_row = QHBoxLayout(topbar)
        top_row.setContentsMargins(10, 0, 10, 0)

        self._detail_back_btn = QPushButton("< Geri")
        self._detail_back_btn.setFixedWidth(80)
        self._detail_back_btn.clicked.connect(self._close_detail)

        self._detail_filename_lbl = QLabel("")
        self._detail_filename_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #ddd;")

        self._detail_prev_btn = QPushButton("<")
        self._detail_prev_btn.setFixedWidth(36)
        self._detail_prev_btn.clicked.connect(self._detail_prev)

        self._detail_next_btn = QPushButton(">")
        self._detail_next_btn.setFixedWidth(36)
        self._detail_next_btn.clicked.connect(self._detail_next)

        self._detail_pos_lbl = QLabel("")
        self._detail_pos_lbl.setStyleSheet("font-size: 11px; color: #888;")

        top_row.addWidget(self._detail_back_btn)
        top_row.addWidget(self._detail_filename_lbl)
        top_row.addStretch()
        top_row.addWidget(self._detail_pos_lbl)
        top_row.addWidget(self._detail_prev_btn)
        top_row.addWidget(self._detail_next_btn)
        outer.addWidget(topbar)

        self._detail_img = QLabel()
        self._detail_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detail_img.setStyleSheet(PREVIEW_STYLE)
        outer.addWidget(self._detail_img, stretch=1)

        return page

    def _build_inspector(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(380)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        stats_grp = QGroupBox("Canli Durum")
        stats_lay = QHBoxLayout(stats_grp)
        photo_col = QVBoxLayout()
        self._lbl_photo_count = QLabel("0")
        self._lbl_photo_count.setObjectName("statValue")
        photo_col.addWidget(self._lbl_photo_count)
        photo_col.addWidget(QLabel("ISLENEN FOTO"))
        panel_col = QVBoxLayout()
        self._lbl_panel_count = QLabel("0")
        self._lbl_panel_count.setObjectName("statValue")
        panel_col.addWidget(self._lbl_panel_count)
        panel_col.addWidget(QLabel("HASARLI PANEL"))
        stats_lay.addLayout(photo_col)
        stats_lay.addLayout(panel_col)
        layout.addWidget(stats_grp)

        legend_grp = QGroupBox("Hasar Listesi")
        legend_lay = QVBoxLayout(legend_grp)
        legend_lay.setContentsMargins(6, 6, 6, 6)
        self._legend = QListWidget()
        self._legend.setMinimumHeight(160)
        self._legend.setSpacing(2)
        self._legend.setUniformItemSizes(True)
        legend_lay.addWidget(self._legend)
        layout.addWidget(legend_grp, stretch=1)

        panel_detail_grp = QGroupBox("Panel Detay")
        panel_detail_lay = QVBoxLayout(panel_detail_grp)
        panel_detail_lay.setContentsMargins(6, 6, 6, 6)
        self._panel_detail_list = QListWidget()
        self._panel_detail_list.setMaximumHeight(100)
        self._panel_detail_list.itemClicked.connect(self._on_panel_detail_clicked)
        panel_detail_lay.addWidget(self._panel_detail_list)
        layout.addWidget(panel_detail_grp)

        report_grp = QGroupBox("Rapor")
        report_lay = QHBoxLayout(report_grp)
        btn_png = QPushButton("PNG Kaydet")
        btn_png.clicked.connect(self._on_export_png)
        btn_pdf = QPushButton("PDF Kaydet")
        btn_pdf.clicked.connect(self._on_export_pdf)
        report_lay.addWidget(btn_png)
        report_lay.addWidget(btn_pdf)
        layout.addWidget(report_grp)

        log_grp = QGroupBox("Log")
        log_lay = QVBoxLayout(log_grp)
        log_lay.setContentsMargins(6, 6, 6, 6)
        self._log_output = QPlainTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setMinimumHeight(120)
        log_lay.addWidget(self._log_output)
        layout.addWidget(log_grp)

        scroll.setWidget(container)
        return scroll

    def _check_weights(self) -> None:
        missing = []
        if not PARTS_WEIGHT.is_file():
            missing.append(str(PARTS_WEIGHT))
        if not DAMAGE_WEIGHT.is_file():
            missing.append(str(DAMAGE_WEIGHT))
        if missing:
            self._status_badge.setText("MODEL: EKSIK")
            self._log("Eksik agirlik dosyalari: " + ", ".join(missing))
        else:
            self._status_badge.setText("MODEL: HAZIR")

    def _log(self, msg: str) -> None:
        self._log_output.appendPlainText(msg)

    def _set_status_bar(self, msg: str) -> None:
        self._status_bar_label.setText(msg)

    def _on_panel_wave_test(self) -> None:
        self._btn_panel_wave.setEnabled(False)
        self._schema.start_wave()

    def _on_panel_wave_finished(self) -> None:
        self._btn_panel_wave.setEnabled(True)

    def _on_conf_changed(self, val: int) -> None:
        self._conf = val / 100.0
        self._lbl_conf.setText(f"{self._conf:.2f}")

    def _on_overlap_changed(self, val: int) -> None:
        self._min_overlap = val / 100.0
        self._lbl_overlap.setText(f"{self._min_overlap:.2f}")

    def _update_session_ui(self) -> None:
        self._lbl_session.setText(f"Oturum: {self._session.session_id}")
        self._lbl_photo_count.setText(str(self._session.photo_count))
        self._lbl_panel_count.setText(str(len(self._session.panels)))
        self._set_status_bar(
            f"{self._session.photo_count} foto islendi | "
            f"{len(self._session.panels)} hasarli panel"
        )
        self._refresh_legend()

    def _status_color(self, status: str) -> str:
        return self._config["status_colors"].get(status, "#B0BEC5")

    def _refresh_legend(self) -> None:
        self._legend.clear()
        self._legend_panel_ids.clear()
        panel_labels = self._config.get("panel_labels", {})
        damage_labels = self._config.get("damage_labels", {})

        if not self._session.panels and self._session.photo_count == 0:
            item = QListWidgetItem()
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            widget = DamageRowWidget("#666", "Henuz veri yok")
            item.setSizeHint(QSize(0, DAMAGE_ROW_H))
            self._legend.addItem(item)
            self._legend.setItemWidget(item, widget)
            return

        for panel_id in sorted(self._session.panels.keys()):
            worst = self._session.get_worst_damage(panel_id)
            status = self._session.get_panel_status(panel_id)
            plabel = panel_labels.get(panel_id, panel_id)
            color = self._status_color(status)
            if worst:
                dlabel = damage_labels.get(worst.damage_type, worst.damage_type)
                text = f"{plabel} · {dlabel} · {worst.confidence:.2f}"
            else:
                text = plabel
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, panel_id)
            widget = DamageRowWidget(color, text)
            item.setSizeHint(QSize(0, DAMAGE_ROW_H))
            self._legend.addItem(item)
            self._legend.setItemWidget(item, widget)
            self._legend_panel_ids.append(panel_id)

        if self._session.photo_count > 0 and not self._session.panels:
            item = QListWidgetItem()
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            widget = DamageRowWidget("#4CAF50", "Hasar tespit edilmedi")
            item.setSizeHint(QSize(0, DAMAGE_ROW_H))
            self._legend.addItem(item)
            self._legend.setItemWidget(item, widget)

    def _select_panel(self, panel_id: str | None) -> None:
        if panel_id is None:
            self._schema.set_highlight(None)
            self._panel_detail_list.clear()
            return

        self._nav_lock = True
        self._schema.set_highlight(panel_id)

        row = -1
        for i, pid in enumerate(self._legend_panel_ids):
            if pid == panel_id:
                row = i
                break
        if row >= 0:
            self._legend.setCurrentRow(row)

        self._refresh_panel_detail(panel_id)

        worst = self._session.get_worst_damage(panel_id)
        if worst:
            self._show_photo_preview(worst.source_photo, use_annotated=True)
        elif panel_id in self._session.panels_seen:
            for path in reversed(self._session.photos_processed):
                self._show_photo_preview(path, use_annotated=True)
                break

        self._nav_lock = False

    def _refresh_panel_detail(self, panel_id: str) -> None:
        self._panel_detail_list.clear()
        damage_labels = self._config.get("damage_labels", {})
        records = self._session.get_damages_for_panel(panel_id)
        if not records:
            self._panel_detail_list.addItem("Kayit yok")
            return
        for rec in records:
            dlabel = damage_labels.get(rec.damage_type, rec.damage_type)
            fname = Path(rec.source_photo).name
            self._panel_detail_list.addItem(
                f"{dlabel} · {rec.confidence:.2f} · {fname}"
            )
            item = self._panel_detail_list.item(self._panel_detail_list.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, rec.source_photo)

    def _on_legend_selected(self, row: int) -> None:
        if self._nav_lock or row < 0 or row >= len(self._legend_panel_ids):
            return
        self._select_panel(self._legend_panel_ids[row])

    def _on_schema_panel_clicked(self, panel_id: str) -> None:
        if panel_id in self._session.panels or panel_id in self._session.panels_seen:
            self._select_panel(panel_id)
        else:
            self._schema.set_highlight(panel_id)
            self._panel_detail_list.clear()
            self._panel_detail_list.addItem("Bu panel icin veri yok")

    def _on_panel_detail_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._show_photo_preview(str(path), use_annotated=True)

    def _thumb_for_path(self, path: str) -> QPixmap:
        if path in self._photo_thumbs:
            return self._photo_thumbs[path]
        frame = cv2.imread(path)
        if frame is None:
            pix = QPixmap(GALLERY_THUMB_W, GALLERY_THUMB_H)
            pix.fill(Qt.GlobalColor.darkGray)
        else:
            pix = scale_pixmap(frame_to_pixmap(frame), GALLERY_THUMB_W, GALLERY_THUMB_H)
        self._photo_thumbs[path] = pix
        return pix

    def _update_sidebar_photo_label(self) -> None:
        n = len(self._pending_paths)
        if n == 0:
            self._sidebar_photo_label.setText("Henuz fotograf yok")
        elif n == 1:
            self._sidebar_photo_label.setText("1 fotograf eklendi")
        else:
            self._sidebar_photo_label.setText(f"{n} fotograf eklendi")

    def _clear_gallery(self) -> None:
        while self._gallery_layout.count():
            item = self._gallery_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._gallery_widgets.clear()

    def _append_gallery_thumb(self, index: int, path: str) -> None:
        status = self._photo_status.get(path, PHOTO_PENDING)
        widget = GalleryThumb(index, self._thumb_for_path(path), status)
        widget.clicked.connect(self._on_gallery_thumb_clicked)
        widget.double_clicked.connect(self._open_detail_at)
        self._gallery_layout.addWidget(widget)
        self._gallery_widgets.append(widget)

    def _update_gallery_status(self, path: str, status: str) -> None:
        self._photo_status[path] = status
        if path not in self._pending_paths:
            return
        idx = self._pending_paths.index(path)
        if idx < len(self._gallery_widgets):
            self._gallery_widgets[idx].set_status(status)

    def _update_gallery_pixmap(self, path: str, pixmap: QPixmap) -> None:
        if path not in self._pending_paths:
            return
        idx = self._pending_paths.index(path)
        if idx < len(self._gallery_widgets):
            self._gallery_widgets[idx].set_pixmap(pixmap)

    def _on_gallery_thumb_clicked(self, index: int) -> None:
        self._select_preview_index(index)

    def _on_new_session(self) -> None:
        if self._session.photo_count > 0:
            reply = QMessageBox.question(
                self,
                "Yeni oturum",
                "Mevcut oturum verisi silinecek. Devam?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._reset_all()

    def _on_reset_session(self) -> None:
        if self._session.photo_count > 0:
            reply = QMessageBox.question(
                self,
                "Oturumu sifirla",
                "Mevcut oturum verisi silinecek. Devam?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._reset_all()

    def _reset_all(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._session.reset()
        self._pending_paths.clear()
        self._photo_status.clear()
        self._photo_thumbs.clear()
        self._clear_gallery()
        self._last_annotated.clear()
        self._preview_index = -1
        self._schema.set_highlight(None)
        self._schema.apply_session(self._session)
        self._preview.setText("Fotograf secin veya ekleyin")
        self._preview.setPixmap(QPixmap())
        self._preview_filename.setText("")
        self._btn_detail.setEnabled(False)
        self._panel_detail_list.clear()
        self._update_sidebar_photo_label()
        self._set_processing_ui(False)
        self._update_session_ui()
        self._log("Oturum sifirlandi — yeni arac icin hazir")

    def _add_paths(self, paths: list[str]) -> None:
        first_add = len(self._pending_paths) == 0
        for p in paths:
            if p not in self._pending_paths:
                self._pending_paths.append(p)
                self._photo_status[p] = PHOTO_PENDING
                self._append_gallery_thumb(len(self._pending_paths) - 1, p)
        self._update_sidebar_photo_label()
        if first_add and self._pending_paths:
            self._select_preview_index(0)

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Fotograf sec",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if paths:
            self._add_paths(paths)

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Klasor sec")
        if not folder:
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(
            str(p) for p in Path(folder).iterdir()
            if p.suffix.lower() in exts
        )
        if paths:
            self._add_paths(paths)
        else:
            QMessageBox.warning(self, "Uyari", "Klasorde goruntu bulunamadi.")

    def _on_camera_capture(self) -> None:
        cameras = list_cameras()
        if not cameras:
            QMessageBox.warning(self, "Kamera", "Kamera bulunamadi.")
            return
        if platform.system() == "Darwin":
            cap = cv2.VideoCapture(cameras[0].index, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(cameras[0].index)
        if not cap.isOpened():
            QMessageBox.warning(self, "Kamera", "Kamera acilamadi.")
            return

        save_dir = PROJECT_ROOT / "output" / "session_captures"
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"capture_{self._session.session_id}_{len(self._pending_paths):04d}.jpg"

        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            QMessageBox.warning(self, "Kamera", "Kare alinamadi.")
            return
        cv2.imwrite(str(fname), frame)
        self._add_paths([str(fname)])
        self._log(f"Kameradan kaydedildi: {fname.name}")

    def _set_processing_ui(self, active: bool) -> None:
        self._btn_process.setEnabled(not active)
        self._btn_cancel.setVisible(active)
        self._progress.setVisible(active)
        if not active:
            self._progress.setValue(0)

    def _on_process_photos(self) -> None:
        if not self._pending_paths:
            QMessageBox.information(self, "Bilgi", "Once foto ekleyin.")
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Bilgi", "Islem devam ediyor.")
            return
        if not PARTS_WEIGHT.is_file() or not DAMAGE_WEIGHT.is_file():
            QMessageBox.critical(
                self,
                "Model eksik",
                f"Agirlik dosyalari bulunamadi:\n{PARTS_WEIGHT}\n{DAMAGE_WEIGHT}\n\n"
                "Once modelleri egitip weights/ altina kopyalayin.",
            )
            return

        unprocessed = [
            p for p in self._pending_paths
            if p not in self._session.photos_processed
        ]
        if not unprocessed:
            QMessageBox.information(self, "Bilgi", "Tum fotograflar zaten islendi.")
            return

        parts = UltralyticsBackend("car-parts-seg", str(PARTS_WEIGHT), color_slot=0)
        damage = UltralyticsBackend("car-damage-seg-v2", str(DAMAGE_WEIGHT), color_slot=1)

        self._processing_total = len(unprocessed)
        self._processing_done = 0
        self._progress.setMaximum(self._processing_total)
        self._progress.setValue(0)
        self._set_processing_ui(True)
        self._set_status_bar(f"Isleniyor: 0/{self._processing_total}")

        self._worker = InspectionWorker(
            unprocessed, parts, damage, self._conf, self._min_overlap
        )
        self._worker.status.connect(self._log)
        self._worker.error.connect(lambda e: self._log(f"HATA: {e}"))
        self._worker.photo_done.connect(self._on_photo_processed)
        self._worker.photo_failed.connect(self._on_photo_failed)
        self._worker.progress.connect(self._on_processing_progress)
        self._worker.finished_all.connect(self._on_processing_finished)
        self._worker.start()

    def _on_cancel_processing(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._log("Islem iptal ediliyor...")

    def _on_processing_progress(self, done: int, total: int) -> None:
        self._processing_done = done
        self._progress.setValue(done)
        self._set_status_bar(f"Isleniyor: {done}/{total}")

    def _on_photo_processed(
        self,
        path: str,
        annotated: np.ndarray,
        assignments: list,
        seen_panels: set,
    ) -> None:
        self._session.add_photo_result(path, assignments, seen_panels)
        self._last_annotated[path] = annotated
        self._update_gallery_status(path, PHOTO_DONE)
        ann_pix = scale_pixmap(frame_to_pixmap(annotated), GALLERY_THUMB_W, GALLERY_THUMB_H)
        self._update_gallery_pixmap(path, ann_pix)
        self._update_session_ui()
        self._schema.apply_session(self._session)

    def _on_photo_failed(self, path: str) -> None:
        self._update_gallery_status(path, PHOTO_ERROR)

    def _on_processing_finished(self) -> None:
        self._set_processing_ui(False)
        self._log(f"Islem tamamlandi. {self._session.photo_count} foto islendi.")
        self._set_status_bar(
            f"{self._session.photo_count} foto islendi | "
            f"{len(self._session.panels)} hasarli panel"
        )

    def _render_preview(self, path: str, use_annotated: bool = False) -> None:
        ann = self._last_annotated.get(path) if use_annotated else None
        if ann is not None:
            pix = scale_pixmap(frame_to_pixmap(ann), 640, 360)
        else:
            frame = cv2.imread(path)
            if frame is None:
                return
            pix = scale_pixmap(frame_to_pixmap(frame), 640, 360)
        self._preview.setPixmap(pix)
        self._preview.setText("")
        self._preview_filename.setText(Path(path).name)
        self._btn_detail.setEnabled(True)

    def _select_preview_index(self, index: int) -> None:
        if index < 0 or index >= len(self._pending_paths):
            self._btn_detail.setEnabled(False)
            return
        self._preview_index = index
        path = self._pending_paths[index]
        use_ann = path in self._last_annotated
        self._render_preview(path, use_annotated=use_ann)
        for i, widget in enumerate(self._gallery_widgets):
            widget.set_selected(i == index)
        if index < len(self._gallery_widgets):
            self._gallery_scroll.ensureWidgetVisible(self._gallery_widgets[index])

    def _show_photo_preview(self, path: str, use_annotated: bool = False) -> None:
        if path in self._pending_paths:
            self._select_preview_index(self._pending_paths.index(path))
        else:
            self._render_preview(path, use_annotated=use_annotated)

    def _open_detail_for_current_photo(self) -> None:
        if self._preview_index >= 0:
            self._open_detail_at(self._preview_index)

    def _open_detail_at(self, index: int) -> None:
        if index < 0 or index >= len(self._pending_paths):
            return
        self._detail_index = index
        self._select_preview_index(index)
        self._update_detail_view()
        self._view_stack.setCurrentIndex(self.PAGE_DETAIL)

    def _close_detail(self) -> None:
        self._view_stack.setCurrentIndex(self.PAGE_MAIN)

    def _detail_prev(self) -> None:
        if self._detail_index > 0:
            self._detail_index -= 1
            self._update_detail_view()

    def _detail_next(self) -> None:
        if self._detail_index < len(self._pending_paths) - 1:
            self._detail_index += 1
            self._update_detail_view()

    def _update_detail_view(self) -> None:
        if self._detail_index < 0 or self._detail_index >= len(self._pending_paths):
            return
        path = self._pending_paths[self._detail_index]
        total = len(self._pending_paths)
        self._detail_filename_lbl.setText(Path(path).name)
        self._detail_pos_lbl.setText(f"{self._detail_index + 1} / {total}")
        self._detail_prev_btn.setEnabled(self._detail_index > 0)
        self._detail_next_btn.setEnabled(self._detail_index < total - 1)
        self._select_preview_index(self._detail_index)

        ann = self._last_annotated.get(path)
        if ann is not None:
            pix = frame_to_pixmap(ann)
        else:
            frame = cv2.imread(path)
            if frame is None:
                return
            pix = frame_to_pixmap(frame)
        label_size = self._detail_img.size()
        w = max(label_size.width() - 20, 400)
        h = max(label_size.height() - 20, 300)
        self._detail_img.setPixmap(scale_pixmap(pix, w, h))

    def _on_export_png(self) -> None:
        if self._session.photo_count == 0:
            QMessageBox.information(self, "Bilgi", "Once fotograflari isleyin.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "PNG kaydet", f"rapor_{self._session.session_id}.png", "PNG (*.png)"
        )
        if path and self._schema.export_png(path):
            self._log(f"PNG kaydedildi: {path}")

    def _on_export_pdf(self) -> None:
        if self._session.photo_count == 0:
            QMessageBox.information(self, "Bilgi", "Once fotograflari isleyin.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "PDF kaydet", f"rapor_{self._session.session_id}.pdf", "PDF (*.pdf)"
        )
        if path and self._schema.export_pdf(path):
            self._log(f"PDF kaydedildi: {path}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
