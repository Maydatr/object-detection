from __future__ import annotations

import os
import platform
import re
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from backends import detections_to_legend

PanelCallback = Callable[[str, np.ndarray, float, int, dict], None]
FocusCallback = Callable[[dict[str, Any] | None], None]
TextCallback = Callable[[str], None]
FinishedCallback = Callable[[], None]


class CaptureEngine:
    def __init__(
        self,
        on_panel: PanelCallback,
        on_focus: FocusCallback,
        on_status: TextCallback,
        on_error: TextCallback,
        on_device: TextCallback,
        on_finished: FinishedCallback,
    ) -> None:
        self._on_panel = on_panel
        self._on_focus = on_focus
        self._on_status = on_status
        self._on_error = on_error
        self._on_device = on_device
        self._on_finished = on_finished

        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lock = threading.Lock()

        self._backends: list[Any] = []
        self._source = "0"
        self._skip_frames = 1
        self._conf = 0.25
        self._save_frames = False
        self._save_dir = "./output"

        self._focus_tracker_id: int | None = None
        self._focus_requested_at: tuple[int, int] | None = None
        self._focus_started_at: float | None = None
        self._last_focus_center: tuple[float, float] | None = None

    def configure(
        self,
        backends: list[Any],
        source: str,
        skip_frames: int,
        conf: float,
        save_frames: bool,
        save_dir: str,
    ) -> None:
        if self.is_running():
            raise RuntimeError("Yakalama çalışırken yapılandırılamaz.")
        self._backends = backends
        self._source = source
        self._skip_frames = max(1, int(skip_frames))
        self._conf = float(conf)
        self._save_frames = bool(save_frames)
        self._save_dir = save_dir
        self.set_focus(None)

    def start(self) -> None:
        if self.is_running():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="capture-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def is_running(self) -> bool:
        thread = self._thread
        return self._running.is_set() and thread is not None and thread.is_alive()

    def set_focus(self, tracker_id: int | None) -> None:
        with self._lock:
            self._focus_tracker_id = tracker_id
            self._focus_requested_at = None
            self._focus_started_at = time.monotonic() if tracker_id is not None else None
            self._last_focus_center = None
        if tracker_id is None:
            self._on_focus(None)

    def request_focus_at(self, x: int, y: int) -> None:
        with self._lock:
            self._focus_requested_at = (int(x), int(y))

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            if not self._backends:
                raise RuntimeError("Model seçilmedi.")

            if self._save_frames:
                os.makedirs(self._save_dir, exist_ok=True)

            for backend in self._backends:
                self._on_status(f"Model yükleniyor: {backend.model_id}")
                device = backend.load()
                backend.reset_tracker()
                self._on_device(device)
                self._on_status(f"Model hazır: {backend.model_id} ({device})")

            cap = self._open_capture(self._source)
            if not cap.isOpened():
                raise RuntimeError(f"Kaynak açılamadı: {self._source}")

            frame_index = 0
            self._on_status("Yakalama başladı.")

            while self._running.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame_index += 1
                if frame_index % self._skip_frames != 0:
                    continue

                for backend_index, backend in enumerate(self._backends):
                    if not self._running.is_set():
                        break

                    input_frame = frame.copy()
                    started = time.perf_counter()
                    detections = backend.infer(input_frame, self._conf)
                    detections = backend.update(detections)

                    focus_id = self._resolve_focus_request(detections)
                    annotated = backend.annotate(input_frame, detections, focus_id)
                    latency_ms = (time.perf_counter() - started) * 1000.0

                    if backend_index == 0:
                        self._publish_focus_info(detections, input_frame, focus_id)

                    if self._save_frames and len(detections) > 0:
                        self._save_detected_frame(annotated, backend.model_id, frame_index)

                    accent = getattr(backend, "accent_color_hex", "#2F6DF6")
                    legend_data = detections_to_legend(detections, focus_id, accent)
                    self._on_panel(backend.model_id, annotated, latency_ms, len(detections), legend_data)
        except Exception as exc:
            self._on_error(str(exc))
        finally:
            self._running.clear()
            if cap is not None:
                cap.release()
            self._on_finished()

    def _open_capture(self, source: str) -> cv2.VideoCapture:
        stripped = source.strip()
        if stripped.isdigit():
            camera_index = int(stripped)
            if platform.system() == "Darwin":
                return cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
            return cv2.VideoCapture(camera_index)
        return cv2.VideoCapture(stripped)

    def _resolve_focus_request(self, detections: Any) -> int | None:
        with self._lock:
            requested_at = self._focus_requested_at
            focus_id = self._focus_tracker_id

        if requested_at is None:
            return focus_id

        selected_id = _tracker_id_at_point(detections, requested_at)
        with self._lock:
            self._focus_requested_at = None
            if selected_id is not None:
                self._focus_tracker_id = selected_id
                self._focus_started_at = time.monotonic()
                self._last_focus_center = None
                focus_id = selected_id

        if selected_id is not None:
            self._on_status(f"Odak kilitlendi: ID {selected_id}")
        else:
            self._on_status("Tıklanan noktada takip edilebilir nesne yok.")
        return focus_id

    def _publish_focus_info(self, detections: Any, frame: np.ndarray, focus_id: int | None) -> None:
        if focus_id is None:
            return
        if detections.tracker_id is None or len(detections) == 0:
            self._on_focus({"status": "kayip", "track_id": focus_id})
            return

        matches = np.where(detections.tracker_id == focus_id)[0]
        if len(matches) == 0:
            self._on_focus({"status": "kayip", "track_id": focus_id})
            return

        idx = int(matches[0])
        x1, y1, x2, y2 = [int(v) for v in detections.xyxy[idx]]
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        center_x = x1 + width / 2.0
        center_y = y1 + height / 2.0

        with self._lock:
            started_at = self._focus_started_at
            previous_center = self._last_focus_center
            self._last_focus_center = (center_x, center_y)

        self._on_focus(
            {
                "label": _label_for_detection(detections, idx),
                "confidence": _confidence_for_detection(detections, idx),
                "track_id": focus_id,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 2) if height else "-",
                "center_x": int(center_x),
                "center_y": int(center_y),
                "color_bgr": _mean_color_bgr(frame, x1, y1, x2, y2),
                "duration_s": time.monotonic() - started_at if started_at is not None else None,
                "direction": _direction(previous_center, (center_x, center_y)),
            }
        )

    def _save_detected_frame(self, frame: np.ndarray, model_id: str, frame_index: int) -> None:
        safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_id))
        filename = f"{frame_index:06d}_{safe_model_id}.jpg"
        cv2.imwrite(os.path.join(self._save_dir, filename), frame)


def _tracker_id_at_point(detections: Any, point: tuple[int, int]) -> int | None:
    if detections.tracker_id is None or len(detections) == 0:
        return None

    px, py = point
    boxes = detections.xyxy
    hits = np.where(
        (boxes[:, 0] <= px)
        & (px <= boxes[:, 2])
        & (boxes[:, 1] <= py)
        & (py <= boxes[:, 3])
    )[0]
    if len(hits) == 0:
        return None

    # If boxes overlap, choose the smallest area: usually the most precise target.
    areas = (boxes[hits, 2] - boxes[hits, 0]) * (boxes[hits, 3] - boxes[hits, 1])
    hit_index = int(hits[int(np.argmin(areas))])
    return int(detections.tracker_id[hit_index])


def _label_for_detection(detections: Any, idx: int) -> str:
    names = detections.data.get("class_name") if detections.data else None
    if names is not None:
        return str(names[idx])
    if detections.class_id is not None:
        return str(detections.class_id[idx])
    return "-"


def _confidence_for_detection(detections: Any, idx: int) -> float | None:
    if detections.confidence is None:
        return None
    return float(detections.confidence[idx])


def _mean_color_bgr(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int] | None:
    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    bgr = crop.reshape(-1, 3).mean(axis=0)
    return tuple(int(v) for v in bgr)


def _direction(
    previous: tuple[float, float] | None,
    current: tuple[float, float],
    threshold: float = 4.0,
) -> str:
    if previous is None:
        return "sabit"

    dx = current[0] - previous[0]
    dy = current[1] - previous[1]
    parts: list[str] = []
    if abs(dx) >= threshold:
        parts.append("sağ" if dx > 0 else "sol")
    if abs(dy) >= threshold:
        parts.append("aşağı" if dy > 0 else "yukarı")
    return " + ".join(parts) if parts else "sabit"
