"""
Detection backends: unified sv.Detections pipeline.

Her backend:
  - infer(frame) -> sv.Detections   (ham inference, zamanlama disaridan yapilir)
  - update(detections) -> sv.Detections  (ByteTrack guncelleme + tracker_id)
  - annotate(frame, detections) -> np.ndarray  (Supervision annotator'lar)

MODEL_REGISTRY: gosterim adi -> (model_cls_key, weight_path)
"""
from __future__ import annotations

import time
import warnings
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
import torch

# sv.ByteTrack 0.28'de deprecated (kaldirilma: 0.30); hala calisir, uyariyi sustur.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import supervision as sv

from ultralytics import YOLO, RTDETR


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _patch_dfine_mps_float64() -> None:
    """D-FINE position embedding uses float64; Apple MPS only supports float32."""
    try:
        import transformers.models.d_fine.modeling_d_fine as dfine_mod
    except ImportError:
        return
    if getattr(dfine_mod, "_mps_float64_patched", False):
        return

    def build_2d_sinusoidal_position_embedding(
        height: int,
        width: int,
        embed_dim: int = 256,
        temperature: float = 10000.0,
        cls_token: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if embed_dim % 4 != 0:
            raise ValueError(f"`embed_dim` must be divisible by 4, got {embed_dim}")

        internal = (
            torch.float32
            if device is not None and torch.device(device).type == "mps"
            else torch.float64
        )
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=internal, device=device) / pos_dim
        omega = 1.0 / temperature**omega

        grid_h = torch.arange(height, dtype=internal, device=device)
        grid_w = torch.arange(width, dtype=internal, device=device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

        emb_h = grid_h.flatten().outer(omega)
        emb_w = grid_w.flatten().outer(omega)
        pos_embed = torch.cat(
            [emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=1
        )

        if cls_token:
            pos_embed = torch.cat(
                [
                    torch.zeros(1, embed_dim, dtype=internal, device=device),
                    pos_embed,
                ],
                dim=0,
            )

        return pos_embed.to(dtype)

    dfine_mod.build_2d_sinusoidal_position_embedding = (
        build_2d_sinusoidal_position_embedding
    )
    dfine_mod._mps_float64_patched = True


# ---------------------------------------------------------------------------
# Renk paleti (her backend ayri renk; birden fazla acikken karmasa olmaz)
# ---------------------------------------------------------------------------
_PALETTE_HEX = ["#2F6DF6", "#F6652F", "#2FF6A0"]  # mavi, turuncu, yesil
_PALETTE = [sv.Color.from_hex(h) for h in _PALETTE_HEX]


def _make_annotators(color: sv.Color) -> tuple[sv.BoxAnnotator, sv.LabelAnnotator]:
    box = sv.BoxAnnotator(color=color, thickness=2)
    label = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.WHITE,
        text_scale=0.5,
        text_thickness=1,
        text_padding=4,
    )
    return box, label


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class DetectionBackend(Protocol):
    model_id: str

    def infer(self, frame: np.ndarray, conf: float) -> sv.Detections: ...
    def update(self, detections: sv.Detections) -> sv.Detections: ...
    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        focus_tracker_id: int | None = None,
    ) -> np.ndarray: ...
    def reset_tracker(self) -> None: ...


# ---------------------------------------------------------------------------
# Ultralytics backend: YOLO ve RT-DETR ikisi de buradan
# ---------------------------------------------------------------------------
class UltralyticsBackend:
    def __init__(self, model_id: str, weight: str, color_slot: int = 0) -> None:
        self.model_id = model_id
        self._weight = weight
        self._device = resolve_device()
        self._model: YOLO | RTDETR | None = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._tracker = sv.ByteTrack()
        slot = color_slot % len(_PALETTE)
        self.accent_color_hex: str = _PALETTE_HEX[slot]
        color = _PALETTE[slot]
        self._box_ann, self._label_ann = _make_annotators(color)
        self._focus_box_ann = sv.BoxAnnotator(
            color=sv.Color.from_hex("#2F6DF6"), thickness=3
        )
        self._focus_label_ann = sv.LabelAnnotator(
            color=sv.Color.from_hex("#2F6DF6"),
            text_color=sv.Color.WHITE,
            text_scale=0.55,
            text_thickness=2,
            text_padding=5,
        )
        self._dim_box_ann = sv.BoxAnnotator(
            color=sv.Color.from_hex("#505870"), thickness=1
        )
        self._mask_ann = sv.MaskAnnotator(
            color=sv.ColorPalette.DEFAULT,
            opacity=0.4,
            color_lookup=sv.ColorLookup.CLASS,
        )

    def load(self) -> str:
        """Modeli cihaza yukle, cihaz adini dondur."""
        weight_lower = self._weight.lower()
        if "rtdetr" in weight_lower or "rt-detr" in weight_lower or "resnet" in weight_lower:
            self._model = RTDETR(self._weight)
        else:
            self._model = YOLO(self._weight)
        self._model.to(self._device)
        return self._device

    def infer(self, frame: np.ndarray, conf: float = 0.25) -> sv.Detections:
        if self._model is None:
            return sv.Detections.empty()
        results = self._model.predict(
            source=frame,
            conf=conf,
            device=self._device,
            verbose=False,
        )
        return sv.Detections.from_ultralytics(results[0])

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self._tracker.update_with_detections(detections)

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        focus_tracker_id: int | None = None,
        show_boxes: bool = True,
        show_masks: bool = True,
    ) -> np.ndarray:
        annotated = frame.copy()

        if focus_tracker_id is not None and detections.tracker_id is not None:
            focus_mask = detections.tracker_id == focus_tracker_id
            dim_mask = ~focus_mask

            dim_dets = detections[dim_mask]
            focus_dets = detections[focus_mask]

            if len(dim_dets) > 0 and show_boxes:
                annotated = self._dim_box_ann.annotate(annotated, dim_dets)

            if len(focus_dets) > 0:
                if show_masks and focus_dets.mask is not None:
                    annotated = self._mask_ann.annotate(annotated, focus_dets)
                if show_boxes:
                    labels = _build_labels(focus_dets)
                    annotated = self._focus_box_ann.annotate(annotated, focus_dets)
                    annotated = self._focus_label_ann.annotate(
                        annotated, focus_dets, labels=labels
                    )
        else:
            if show_masks and detections.mask is not None:
                annotated = self._mask_ann.annotate(annotated, detections)
            if show_boxes:
                labels = _build_labels(detections)
                annotated = self._box_ann.annotate(annotated, detections)
                annotated = self._label_ann.annotate(
                    annotated, detections, labels=labels
                )

        return annotated

    def reset_tracker(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._tracker = sv.ByteTrack()


# ---------------------------------------------------------------------------
# HuggingFace Transformers backend: Meta DETR, USTC D-FINE vb.
# ---------------------------------------------------------------------------
class TransformersBackend:
    def __init__(
        self, model_id: str, checkpoint: str, color_slot: int = 2
    ) -> None:
        self.model_id = model_id
        self._checkpoint = checkpoint
        self._device = resolve_device()
        self._processor = None
        self._model = None
        self._id2label: dict[int, str] | None = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._tracker = sv.ByteTrack()
        slot = color_slot % len(_PALETTE)
        self.accent_color_hex: str = _PALETTE_HEX[slot]
        color = _PALETTE[slot]
        self._box_ann, self._label_ann = _make_annotators(color)
        self._dim_box_ann = sv.BoxAnnotator(
            color=sv.Color.from_hex("#505870"), thickness=1
        )

    def load(self) -> str:
        try:
            from transformers import (
                AutoImageProcessor,
                AutoModelForObjectDetection,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers paketi kurulu degil. "
                "'pip install transformers timm pillow' ile yukle."
            ) from exc

        if self._device == "mps" and "dfine" in self._checkpoint.lower():
            _patch_dfine_mps_float64()

        self._processor = AutoImageProcessor.from_pretrained(self._checkpoint)
        self._model = AutoModelForObjectDetection.from_pretrained(
            self._checkpoint
        )
        self._model.to(self._device)
        self._model.eval()
        self._id2label = {
            int(k): v for k, v in self._model.config.id2label.items()
        }
        return self._device

    def infer(self, frame: np.ndarray, conf: float = 0.25) -> sv.Detections:
        if self._model is None or self._processor is None:
            return sv.Detections.empty()

        from PIL import Image as PILImage  # type: ignore

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        h, w = frame.shape[:2]
        target = torch.tensor([[h, w]], device=self._device)
        results = self._processor.post_process_object_detection(
            outputs, threshold=conf, target_sizes=target
        )[0]
        results = {k: v.detach().cpu() for k, v in results.items()}
        return sv.Detections.from_transformers(
            results, id2label=self._id2label
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self._tracker.update_with_detections(detections)

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        focus_tracker_id: int | None = None,
        show_boxes: bool = True,
        show_masks: bool = True,
    ) -> np.ndarray:
        annotated = frame.copy()
        if focus_tracker_id is not None and detections.tracker_id is not None:
            dim_dets = detections[detections.tracker_id != focus_tracker_id]
            focus_dets = detections[detections.tracker_id == focus_tracker_id]
            if len(dim_dets) > 0 and show_boxes:
                annotated = self._dim_box_ann.annotate(annotated, dim_dets)
            if len(focus_dets) > 0 and show_boxes:
                labels = _build_labels(focus_dets)
                annotated = self._box_ann.annotate(annotated, focus_dets)
                annotated = self._label_ann.annotate(
                    annotated, focus_dets, labels=labels
                )
        else:
            if show_boxes:
                labels = _build_labels(detections)
                annotated = self._box_ann.annotate(annotated, detections)
                annotated = self._label_ann.annotate(
                    annotated, detections, labels=labels
                )
        return annotated

    def reset_tracker(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._tracker = sv.ByteTrack()


# ---------------------------------------------------------------------------
# RF-DETR backend (Roboflow)
# ---------------------------------------------------------------------------
class RFDetrBackend:
    def __init__(self, model_id: str, weight: str, color_slot: int = 2) -> None:
        self.model_id = model_id
        self._weight = weight
        self._device = resolve_device()
        self._model = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._tracker = sv.ByteTrack()
        slot = color_slot % len(_PALETTE)
        self.accent_color_hex: str = _PALETTE_HEX[slot]
        color = _PALETTE[slot]
        self._box_ann, self._label_ann = _make_annotators(color)
        self._dim_box_ann = sv.BoxAnnotator(
            color=sv.Color.from_hex("#505870"), thickness=1
        )

    def load(self) -> str:
        try:
            from rfdetr import RFDETRBase, RFDETRLarge  # type: ignore
            if "large" in self._weight.lower():
                self._model = RFDETRLarge(pretrain_weights=self._weight)
            else:
                self._model = RFDETRBase(pretrain_weights=self._weight)
        except ImportError as exc:
            raise ImportError(
                "rfdetr paketi kurulu degil. 'pip install rfdetr' ile yukle."
            ) from exc
        return self._device

    def infer(self, frame: np.ndarray, conf: float = 0.25) -> sv.Detections:
        if self._model is None:
            return sv.Detections.empty()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage  # type: ignore
        pil_img = PILImage.fromarray(rgb)
        dets = self._model.predict(pil_img, threshold=conf)
        return dets

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self._tracker.update_with_detections(detections)

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        focus_tracker_id: int | None = None,
        show_boxes: bool = True,
        show_masks: bool = True,
    ) -> np.ndarray:
        annotated = frame.copy()
        if focus_tracker_id is not None and detections.tracker_id is not None:
            dim_dets = detections[detections.tracker_id != focus_tracker_id]
            focus_dets = detections[detections.tracker_id == focus_tracker_id]
            if len(dim_dets) > 0 and show_boxes:
                annotated = self._dim_box_ann.annotate(annotated, dim_dets)
            if len(focus_dets) > 0 and show_boxes:
                labels = _build_labels(focus_dets)
                annotated = self._box_ann.annotate(annotated, focus_dets)
                annotated = self._label_ann.annotate(
                    annotated, focus_dets, labels=labels
                )
        else:
            if show_boxes:
                labels = _build_labels(detections)
                annotated = self._box_ann.annotate(annotated, detections)
                annotated = self._label_ann.annotate(
                    annotated, detections, labels=labels
                )
        return annotated

    def reset_tracker(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._tracker = sv.ByteTrack()


# ---------------------------------------------------------------------------
# Label builder yardimcisi
# ---------------------------------------------------------------------------
def _build_labels(detections: sv.Detections) -> list[str]:
    labels = []
    names = (
        detections.data.get("class_name")
        if detections.data
        else None
    )
    for i in range(len(detections)):
        cls_name = names[i] if names is not None else str(detections.class_id[i] if detections.class_id is not None else "?")
        conf = f"{detections.confidence[i]:.2f}" if detections.confidence is not None else ""
        tid = f" ID:{detections.tracker_id[i]}" if detections.tracker_id is not None else ""
        labels.append(f"{cls_name} {conf}{tid}")
    return labels


def detections_to_legend(
    detections: sv.Detections,
    focus_tracker_id: int | None = None,
    accent_hex: str = "#2F6DF6",
) -> dict:
    """Serialize sv.Detections to a plain dict for the Qt legend overlay.

    Returns:
        {
            "accent_hex": str,
            "items": [{"name", "confidence", "track_id", "is_focused"}, ...],
            "summary": [{"name", "count", "max_conf"}, ...],
        }
    Items are sorted by confidence descending; summary by count descending.
    """
    names = detections.data.get("class_name") if detections.data else None

    items: list[dict] = []
    for i in range(len(detections)):
        cls_name = (
            str(names[i])
            if names is not None
            else str(detections.class_id[i] if detections.class_id is not None else "?")
        )
        conf = float(detections.confidence[i]) if detections.confidence is not None else None
        tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else None
        items.append({
            "name": cls_name,
            "confidence": conf,
            "track_id": tid,
            "is_focused": tid is not None and tid == focus_tracker_id,
        })

    items.sort(key=lambda x: x["confidence"] or 0.0, reverse=True)

    # per-class summary
    groups: dict[str, list[float]] = {}
    for item in items:
        groups.setdefault(item["name"], []).append(item["confidence"] or 0.0)

    summary = sorted(
        [{"name": n, "count": len(c), "max_conf": max(c)} for n, c in groups.items()],
        key=lambda x: (-x["count"], -x["max_conf"]),
    )

    return {"accent_hex": accent_hex, "items": items, "summary": summary}


# ---------------------------------------------------------------------------
# MODEL_REGISTRY
# format: display_name -> callable factory (lazy; cagrildiginda backend olusturulur)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, callable] = {
    "Arac Hasar  (YOLO26n-seg)": lambda: UltralyticsBackend(
        model_id="car-damage-seg",
        weight="weights/car-damage-seg.pt",
        color_slot=0,
    ),
}


def create_backend(
    display_name: str,
) -> UltralyticsBackend | TransformersBackend | RFDetrBackend:
    factory = MODEL_REGISTRY[display_name]
    return factory()
