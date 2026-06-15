"""Map car_svg/car.svg paths to panels by view + position rules."""
from __future__ import annotations

import json
import re
import yaml
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATH_ANALYSIS = PROJECT_ROOT / "car_svg" / "path_analysis.json"
PANEL_CONFIG = PROJECT_ROOT / "panel_config.yaml"

OUTLINE_MIN = 1800


def is_panel_path(entry: dict) -> bool:
    if entry["path_id"] == "path1" or entry.get("stroke_only"):
        return False
    if entry.get("fill") in ("none", "transparent"):
        return False
    w, h = entry["width"], entry["height"]
    if entry.get("fill") == "#000000" and (w > OUTLINE_MIN or h > OUTLINE_MIN):
        return False
    return True


def assign_front(cx: float, cy: float, fill: str) -> str:
    if cy < 2220:
        return "wheel"
    if fill == "#000000":
        return "front_light" if cy < 2520 else "front_bumper"
    if cx > 900 and cy < 2470:
        return "front_right_light"
    if fill == "#fefefe" and cy < 2480:
        return "front_glass"
    if fill == "#fefefe" and cy < 2540:
        return "hood"
    if fill == "#a1a1a1":
        return "front_bumper"
    if cx < 350:
        return "front_left_light"
    if cy >= 2540:
        return "front_bumper"
    return "front_light"


def assign_side(cx: float, cy: float, fill: str) -> str:
    if cy > 3040:
        return "wheel"
    if cx < 190:
        return "front_bumper" if fill == "#fefefe" else "front_left_light"
    if cx < 310:
        return "hood"
    if cx < 500:
        return "front_left_door"
    if cx < 680:
        return "back_left_door"
    if cx < 800:
        return "trunk"
    if cx < 920:
        return "back_glass"
    if cx < 1020:
        return "back_left_light"
    return "back_bumper"


def assign_rear(cx: float, cy: float, fill: str) -> str:
    if cy > 1990:
        return "wheel"
    if cx < 280:
        return "back_bumper"
    if cx < 420:
        return "tailgate"
    if cx < 560:
        return "trunk" if fill != "#fefefe" or cy < 1880 else "back_door"
    if cx < 700:
        return "back_door"
    if cx < 860:
        return "back_glass"
    if cx < 980:
        return "back_left_light"
    return "back_right_light"


def assign_top(cx: float, cy: float, fill: str) -> str:
    if cy > 2460:
        return "front_bumper"
    if cy > 2380:
        return "hood"
    if cy > 2320:
        return "front_light"
    if cy > 2240:
        return "wheel"
    if cy > 2120:
        return "front_left_door" if cx < 1330 else "front_right_door"
    if cy > 2020:
        return "wheel"
    if cy > 1920:
        return "back_left_door" if cx < 1330 else "back_right_door"
    if cy > 1820:
        if cx < 1280:
            return "tailgate"
        if cx < 1380:
            return "trunk"
        return "back_glass"
    if cy > 1760:
        return "back_light"
    return "back_bumper"


def assign_panel(entry: dict) -> str | None:
    view = entry["view"]
    cx, cy = entry["cx"], entry["cy"]
    fill = entry.get("fill", "")
    if view == "side":
        return assign_side(cx, cy, fill)
    if view == "front":
        return assign_front(cx, cy, fill)
    if view == "rear":
        return assign_rear(cx, cy, fill)
    if view == "top":
        return assign_top(cx, cy, fill)
    return None


def main() -> None:
    entries = json.loads(PATH_ANALYSIS.read_text(encoding="utf-8"))
    panel_paths: dict[str, list[str]] = defaultdict(list)

    for entry in entries:
        if not is_panel_path(entry):
            continue
        panel = assign_panel(entry)
        if panel:
            panel_paths[panel].append(entry["path_id"])

    for panel_id in panel_paths:
        panel_paths[panel_id].sort(key=lambda x: int(re.sub(r"\D", "", x) or 0))

    config = yaml.safe_load(PANEL_CONFIG.read_text(encoding="utf-8"))
    config.pop("panel_dots", None)
    config["schema_svg"] = "car_svg/car.svg"
    config["panel_elements"] = dict(sorted(panel_paths.items()))

    with open(PANEL_CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    total = sum(len(v) for v in panel_paths.values())
    print(f"Mapped {total} paths -> {len(panel_paths)} panels")
    for pid in config.get("schema_panels", []):
        paths = panel_paths.get(pid, [])
        print(f"  {pid}: {paths}")


if __name__ == "__main__":
    main()
