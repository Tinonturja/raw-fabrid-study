"""Check every released patch against its recorded crop and quantify missed mask regions."""
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"train_good": "train/good", "test_good": "test/good", "test_defect": "test/defect"}


def gray(archive, member):
    with Image.open(io.BytesIO(archive.read(member))) as im:
        return np.array(im.convert("L"))


def main():
    meta = ROOT / "data" / "remote_metadata"
    with (meta / "RAW_FABRID_HighRes_Metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        info = {r["filename"]: r for r in csv.DictReader(f)}
    groups = defaultdict(list)
    for im in json.loads((meta / "RAW_FABRID_Patches_COCO.json").read_text())["images"]:
        groups[im["parent"]].append(im)
    assert set(groups) == set(info)
    mismatches = []
    checked = 0
    component_rows = {}
    with zipfile.ZipFile(ROOT / "data" / "RAW_FABRID.zip") as archive:
        for parent, patches in sorted(groups.items()):
            raw = gray(archive, "RAW_FABRID/images/" + parent)
            seen = np.zeros(raw.shape, dtype=bool)
            for patch in patches:
                y, x = patch["r_offset"], patch["c_offset"]
                subset = np.s_[y:y+256, x:x+256]
                seen[subset] = True
                member = f"RAW_FABRID/MVTec/{FOLDERS[patch['type']]}/{patch['file_name']}"
                if not np.array_equal(raw[subset], gray(archive, member)):
                    mismatches.append(patch["file_name"])
                checked += 1
            if info[parent]["has_defect"].lower() != "true":
                assert seen.all()
                continue
            mask = gray(archive, "RAW_FABRID/masks/" + parent)
            component_rows[parent] = {}
            for threshold in (1, 128, 255):
                fg = mask >= threshold
                components, count = label(fg, structure=np.ones((3, 3), dtype=np.uint8))
                pixels = np.bincount(components[fg], minlength=count+1)[1:]
                partial = np.bincount(components[fg & seen], minlength=count+1)[1:]
                component_rows[parent][threshold] = {
                    "foreground_pixels": int(pixels.sum()),
                    "unseen_foreground_pixels": int((pixels - partial).sum()),
                    "n_components": count,
                    "partial_components": int(((partial > 0) & (partial < pixels)).sum()),
                    "fully_missed_component_areas": pixels[partial == 0].tolist(),
                }
    result = {"checked_patches": checked, "grayscale_pixel_mismatches": mismatches,
              "mask_connectivity": 8, "by_parent_and_threshold": component_rows}
    (ROOT / "results" / "release_coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Verified crops", checked, "mismatches", len(mismatches))
    for t in (1, 128, 255):
        rows = [r[t] for r in component_rows.values()]
        print(t, "foreground missing/total", sum(r["unseen_foreground_pixels"] for r in rows),
              sum(r["foreground_pixels"] for r in rows), "components entirely missed/total",
              sum(len(r["fully_missed_component_areas"]) for r in rows), sum(r["n_components"] for r in rows))


if __name__ == "__main__":
    main()
