#!/usr/bin/env python3
"""CarDD COCO instance segmentation -> YOLO-seg format converter."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARDD_ANNOTATIONS = Path(r"C:\datasets\CarDD_release\CarDD_COCO\annotations")
CARDD_ROOT = Path(r"C:\datasets\CarDD_release\CarDD_COCO")
SAVE_DIR = PROJECT_ROOT / "datasets" / "cardd-seg"

SPLITS = ("train2017", "val2017", "test2017")

DATA_YAML = f"""\
path: {CARDD_ROOT.as_posix()}
train: images/train2017
val: images/val2017
test: images/test2017
nc: 6
names:
  0: dent
  1: scratch
  2: crack
  3: glass_shatter
  4: lamp_broken
  5: tire_flat
"""


def _ensure_yolo_layout() -> None:
    """Ultralytics path.resolve() junction bozar; images/ + labels/ gercek klasor olmali."""
    images_root = CARDD_ROOT / "images"
    labels_root = CARDD_ROOT / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        legacy_img = CARDD_ROOT / split
        target_img = images_root / split
        if legacy_img.is_dir() and not target_img.exists():
            legacy_img.rename(target_img)
            print(f"Moved {split} -> images/{split}")

        src_lbl = SAVE_DIR / "labels" / split
        dst_lbl = labels_root / split
        if not src_lbl.is_dir():
            raise FileNotFoundError(f"Label split missing: {src_lbl}")
        if dst_lbl.exists():
            shutil.rmtree(dst_lbl)
        shutil.copytree(src_lbl, dst_lbl)
        print(f"Copied labels -> {dst_lbl}")

    for cache in CARDD_ROOT.rglob("*.cache"):
        cache.unlink()
        print(f"Removed cache: {cache}")


def _write_data_yaml() -> None:
    yaml_path = SAVE_DIR / "data.yaml"
    yaml_path.write_text(DATA_YAML, encoding="utf-8")
    print(f"Wrote {yaml_path}")


def main() -> int:
    if not CARDD_ANNOTATIONS.is_dir():
        print(f"ERROR: annotations dir not found: {CARDD_ANNOTATIONS}", file=sys.stderr)
        return 1

    from ultralytics.data.converter import convert_coco

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Converting COCO -> YOLO-seg into {SAVE_DIR}")
    convert_coco(
        labels_dir=str(CARDD_ANNOTATIONS),
        save_dir=str(SAVE_DIR),
        use_segments=True,
        cls91to80=False,
    )
    _ensure_yolo_layout()
    _write_data_yaml()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
