"""Single-vehicle inspection session: accumulates panel -> damage assignments."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from mask_intersection import MaskDetection, normalize_class_name

# Higher index = more severe (worst wins on merge).
DAMAGE_SEVERITY: dict[str, int] = {
    "scratch": 0,
    "dent": 1,
    "lamp_broken": 2,
    "glass_shatter": 3,
    "crack": 4,
    "tire_flat": 5,
}

PANEL_STATUS_OK = "ok"
PANEL_STATUS_NO_DATA = "no_data"
PANEL_STATUS_LIGHT = "light"
PANEL_STATUS_HEAVY = "heavy"

LIGHT_DAMAGE = {"scratch", "dent"}
HEAVY_DAMAGE = {"crack", "glass_shatter", "lamp_broken", "tire_flat"}


@dataclass
class DamageRecord:
    damage_type: str
    confidence: float
    overlap_ratio: float
    source_photo: str

    def __post_init__(self) -> None:
        self.damage_type = normalize_class_name(self.damage_type)


@dataclass
class VehicleSession:
    """One vehicle per session; reset before starting the next vehicle."""

    session_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    panels: dict[str, list[DamageRecord]] = field(default_factory=dict)
    panels_seen: set[str] = field(default_factory=set)
    photos_processed: list[str] = field(default_factory=list)
    photo_count: int = 0

    def reset(self) -> None:
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.panels.clear()
        self.panels_seen.clear()
        self.photos_processed.clear()
        self.photo_count = 0

    def add_photo_result(
        self,
        photo_path: str,
        assignments: list[tuple[str, MaskDetection, float]],
        seen_panels: set[str] | None = None,
    ) -> None:
        self.photos_processed.append(photo_path)
        self.photo_count += 1
        if seen_panels:
            self.panels_seen.update(seen_panels)

        for panel_name, dmg, ratio in assignments:
            record = DamageRecord(
                damage_type=dmg.class_name,
                confidence=dmg.confidence,
                overlap_ratio=ratio,
                source_photo=photo_path,
            )
            self._merge_panel_damage(panel_name, record)

    def _merge_panel_damage(self, panel_name: str, record: DamageRecord) -> None:
        existing = self.panels.get(panel_name, [])
        if not existing:
            self.panels[panel_name] = [record]
            return

        worst = max(existing, key=lambda r: DAMAGE_SEVERITY.get(r.damage_type, -1))
        new_sev = DAMAGE_SEVERITY.get(record.damage_type, -1)
        old_sev = DAMAGE_SEVERITY.get(worst.damage_type, -1)

        if new_sev > old_sev:
            self.panels[panel_name] = [record]
        elif new_sev == old_sev and record.confidence > worst.confidence:
            self.panels[panel_name] = [record]
        else:
            self.panels[panel_name].append(record)

    def get_worst_damage(self, panel_name: str) -> DamageRecord | None:
        records = self.panels.get(panel_name)
        if not records:
            return None
        return max(records, key=lambda r: (DAMAGE_SEVERITY.get(r.damage_type, -1), r.confidence))

    def get_damages_for_panel(self, panel_name: str) -> list[DamageRecord]:
        return list(self.panels.get(panel_name, []))

    def get_panel_status(self, panel_name: str, all_panel_ids: list[str] | None = None) -> str:
        worst = self.get_worst_damage(panel_name)
        if worst is not None:
            if worst.damage_type in HEAVY_DAMAGE:
                return PANEL_STATUS_HEAVY
            if worst.damage_type in LIGHT_DAMAGE:
                return PANEL_STATUS_LIGHT
            return PANEL_STATUS_HEAVY

        if panel_name in self.panels_seen:
            return PANEL_STATUS_OK
        return PANEL_STATUS_NO_DATA

    def summary(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for panel, records in self.panels.items():
            worst = self.get_worst_damage(panel)
            out[panel] = {
                "status": self.get_panel_status(panel),
                "worst_type": worst.damage_type if worst else None,
                "worst_conf": worst.confidence if worst else None,
                "count": len(records),
            }
        return out
