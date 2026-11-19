"""Illustrate the largest fully omitted 8-connected region at mask value 255."""
import io
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
coverage = json.loads((ROOT / "results/release_coverage.json").read_text())
size, parent = max((a, p) for p, rows in coverage["by_parent_and_threshold"].items()
                   for a in rows["255"]["fully_missed_component_areas"])
patches = json.loads((ROOT / "data/remote_metadata/RAW_FABRID_Patches_COCO.json").read_text())
with zipfile.ZipFile(ROOT / "data/RAW_FABRID.zip") as archive:
    with Image.open(io.BytesIO(archive.read("RAW_FABRID/images/" + parent))) as im:
        gray = np.array(im.convert("L"))
    with Image.open(io.BytesIO(archive.read("RAW_FABRID/masks/" + parent))) as im:
        mask = np.array(im.convert("L")) == 255
seen = np.zeros(gray.shape, dtype=bool)
for row in patches["images"]:
    if row["parent"] == parent:
        y, x = row["r_offset"], row["c_offset"]
        seen[y:y+256, x:x+256] = True
parts, n = label(mask, structure=np.ones((3, 3)))
selected = next(i for i in range(1, n+1) if (parts == i).sum() == size and not seen[parts == i].any())
ys, xs = np.nonzero(parts == selected)
pad = 42
x0, x1 = max(0, xs.min()-pad), min(gray.shape[1], xs.max()+pad+1)
y0, y1 = max(0, ys.min()-pad), min(gray.shape[0], ys.max()+pad+1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
for ax, x, y, xx, yy, title in zip(axes, [0, x0], [0, y0], [gray.shape[1], x1],
                                      [gray.shape[0], y1], ["Whole source frame", "Marked region, enlarged"]):
    field = np.s_[y:yy, x:xx]
    ax.imshow(gray[field], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    tint = np.zeros((*gray[field].shape, 4))
    tint[~seen[field]] = [0.08, 0.62, 0.88, 0.35]
    ax.imshow(tint, interpolation="nearest")
    ax.contour(mask[field], levels=[.5], colors=["#ffbe46"], linewidths=1)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
axes[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor="none",
                            edgecolor="white", linewidth=1.4, linestyle="--"))
fig.suptitle(f"All 28 released crops omit this {size:,}-pixel marked region", fontsize=14)
fig.legend(handles=[Patch(facecolor="#54a8cf", label="Area outside all released crops"),
                    Line2D([0], [0], color="#de9312", label="Source mask outline")],
           loc="lower center", bbox_to_anchor=(.5, .09), ncol=2, frameon=False)
fig.text(.5, .038, parent + "  ·  RAW-FABRID v1, Pérez-Llorens et al., CC BY 4.0",
         ha="center", fontsize=8)
fig.subplots_adjust(top=.84, bottom=.22, wspace=.08)
fig.savefig(ROOT / "reports/missed_region_example.png", dpi=175, bbox_inches="tight")
print(parent, size)
