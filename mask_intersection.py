"""Pixel-level overlap-ratio mask intersection for damage-to-panel assignment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv


@dataclass(frozen=True)
class MaskDetection:
    class_name: str
    confidence: float
    mask: np.ndarray  # H x W bool


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def overlap_ratio(damage_mask: np.ndarray, part_mask: np.ndarray) -> float:
    """Fraction of damage pixels that lie inside the part mask."""
    damage_area = int(damage_mask.sum())
    if damage_area == 0:
        return 0.0
    return float((damage_mask & part_mask).sum()) / damage_area


def assign_damage_to_panel(
    damage_mask: np.ndarray,
    part_masks: dict[str, np.ndarray],
    min_ratio: float = 0.10,
) -> tuple[str | None, float]:
    damage_area = int(damage_mask.sum())
    if damage_area == 0:
        return None, 0.0

    best_name: str | None = None
    best_ratio = 0.0
    for name, pmask in part_masks.items():
        ratio = overlap_ratio(damage_mask, pmask)
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = name

    if best_name is None or best_ratio < min_ratio:
        return "unknown", best_ratio
    return best_name, best_ratio


def detections_to_masks(
    detections: sv.Detections,
    ignore_classes: set[str] | None = None,
) -> list[MaskDetection]:
    if len(detections) == 0 or detections.mask is None:
        return []

    ignore = ignore_classes or set()
    names = detections.data.get("class_name") if detections.data else None
    out: list[MaskDetection] = []

    for i in range(len(detections)):
        cls_name = (
            normalize_class_name(str(names[i]))
            if names is not None
            else str(detections.class_id[i])
        )
        if cls_name in ignore:
            continue
        mask = detections.mask[i].astype(bool)
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        out.append(MaskDetection(class_name=cls_name, confidence=conf, mask=mask))

    return out


def build_part_mask_dict(part_detections: list[MaskDetection]) -> dict[str, np.ndarray]:
    """Merge multiple detections of the same panel class into one union mask."""
    merged: dict[str, np.ndarray] = {}
    for det in part_detections:
        if det.class_name in merged:
            merged[det.class_name] |= det.mask
        else:
            merged[det.class_name] = det.mask.copy()
    return merged


def assign_damages_to_panels(
    damage_detections: list[MaskDetection],
    part_masks: dict[str, np.ndarray],
    min_ratio: float = 0.10,
) -> list[tuple[str, MaskDetection, float]]:
    """Return list of (panel_name, damage_detection, overlap_ratio)."""
    results: list[tuple[str, MaskDetection, float]] = []
    for dmg in damage_detections:
        panel, ratio = assign_damage_to_panel(dmg.mask, part_masks, min_ratio=min_ratio)
        if panel is None:
            continue
        results.append((panel, dmg, ratio))
    return results
