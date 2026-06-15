from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass

import cv2


@dataclass
class CameraDevice:
    index: int
    name: str

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name}"


def _macos_avf_names_modern() -> list[str]:
    """AVCaptureDeviceDiscoverySession ile cihaz isimlerini dondurur.

    Bu API macOS 10.15+ icin OpenCV'nin CAP_AVFOUNDATION backend'inin
    dahili olarak kullandigi yontemle ayni siralamayi verir.
    """
    try:
        import AVFoundation as AVF  # type: ignore
    except ImportError:
        return []
    try:
        device_types = [
            AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera,
            AVF.AVCaptureDeviceTypeExternalUnknown,
        ]
        session = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
            device_types,
            AVF.AVMediaTypeVideo,
            AVF.AVCaptureDevicePositionUnspecified,
        )
        return [str(d.localizedName()) for d in session.devices()]
    except Exception:
        return []


def _macos_avf_names_legacy() -> list[str]:
    """Eski API fallback — macOS 13 oncesi veya modern API basarisiz olursa."""
    try:
        import AVFoundation as AVF  # type: ignore
    except ImportError:
        return []
    try:
        devices = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)
        return [str(d.localizedName()) for d in devices]
    except Exception:
        return []


def _macos_profiler_names() -> list[str]:
    try:
        output = subprocess.run(
            ["system_profiler", "-json", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if output.returncode != 0:
            return []
        data = json.loads(output.stdout)
        return [cam.get("_name", "Bilinmeyen kamera") for cam in data.get("SPCameraDataType", [])]
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return []


def _probe_working_indices(max_probe: int, backend: int) -> list[int]:
    """cv2.VideoCapture ile hangi index'lerin gercekten acildigini dondurur.

    Bu, AVFoundation siralama farkliliklarindan bagimsiz ground-truth haritalama
    saglar: dondurdugunuz index her zaman OpenCV'de dogru kamerayi acar.
    """
    working: list[int] = []
    consecutive_miss = 0
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            working.append(i)
            cap.release()
            consecutive_miss = 0
        else:
            cap.release()
            consecutive_miss += 1
            if consecutive_miss >= 2 and working:
                break
    return working


def list_cameras(max_probe: int = 6) -> list[CameraDevice]:
    """Cihazdaki kameralari OpenCV index sirasiyla listeler.

    macOS: Gercek OpenCV index'lerini cv2.VideoCapture probe ile bulur,
    isimleri modern AVFoundation API'sinden (DiscoverySession) alir.
    Diger platformlar: jenerik 0..N listesi.
    """
    if platform.system() == "Darwin":
        backend = cv2.CAP_AVFOUNDATION
        working = _probe_working_indices(max_probe, backend)
        if not working:
            working = list(range(min(max_probe, 2)))

        names = (
            _macos_avf_names_modern()
            or _macos_avf_names_legacy()
            or _macos_profiler_names()
        )

        devices: list[CameraDevice] = []
        for i in working:
            name = names[i] if i < len(names) else f"Kamera {i}"
            devices.append(CameraDevice(i, name))
        return devices

    return [CameraDevice(i, f"Kamera {i}") for i in range(max_probe)]
