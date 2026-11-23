"""Methods figure: the Protocol 5 evaluation pipeline, drawn in the shared style with real thumbnails from saved
results (RAW-FABRID Rollo7A first defective frame; the same frame as the score-map figure). No new computation.
Usage: python3 fig_pipeline.py"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S

S.apply()
HERE = Path(__file__).resolve().parents[1]
Q = HERE / 'results_p5' / 'qualitative'
cz = np.load(Q / 'coverage_examples.npz')
cm = json.loads((Q / 'coverage_examples.json').read_text())['schematic']
mz = np.load(Q / 'maps.npz')
FILL, EDGE = '#f6f4ef', '#d6d0c2'

fig = plt.figure(figsize=(S.DOUBLE, 3.05))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis('off')


def box(x, y, w, h, n, title, fc=FILL, ec=EDGE, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=fc, ec=ec, lw=0.7, ls=ls))
    if n:
        ax.add_patch(Circle((x + 1.9, y + h - 2.0), 1.25, fc=S.COBALT, ec='none'))
        ax.text(x + 1.9, y + h - 2.05, n, color='white', fontsize=6.5, fontweight='bold', ha='center', va='center')
        ax.text(x + 3.7, y + h - 2.05, title, fontsize=7, fontweight='bold', ha='left', va='center')
    else:
        ax.text(x + 1.2, y + h - 2.05, title, fontsize=7, fontweight='bold', ha='left', va='center')


def thumb(img, x, y, w, cmap=None, mask=None, outline=None):
    h = w * img.shape[0] / img.shape[1] * (100 / 42) * (3.05 / S.DOUBLE)   # keep the image aspect in figure units
    a = ax.inset_axes((x, y, w, h), transform=ax.transData)
    a.imshow(img, cmap=cmap or ('gray' if img.ndim == 2 else None), interpolation='lanczos')
    if mask is not None:
        ring = binary_dilation(mask, iterations=max(2, round(0.01 * mask.shape[1])))
        a.contour(ring, levels=[0.5], colors=outline or S.COPPER, linewidths=0.6)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(S.INK2); sp.set_linewidth(0.4)
    return a, h


def arrow(x0, y0, x1, y1, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=0.8, color=S.INK2, ls=ls,
                                 shrinkA=0, shrinkB=0))


def body(x, y, text, size=6, color=None):
    ax.text(x, y, text, fontsize=size, color=color or S.INK2, ha='left', va='top', linespacing=1.35)


# frozen-before-scoring band
ax.add_patch(FancyBboxPatch((0.6, 37.6), 98.8, 3.6, boxstyle='round,pad=0,rounding_size=1.0', fc='#eef1f8',
                            ec='none'))
ax.text(1.8, 39.4, 'Fixed before any score was seen:', fontsize=6.5, fontweight='bold', va='center', color=S.COBALT)
ax.text(22.5, 39.4, 'protocol, amendment and implementation hashed; files checked by SHA-256 (TSFabrics: run outputs only); '
        'one seed (13017)', fontsize=6.3, va='center', color=S.INK)

Y, HB = 11.5, 24.5            # main row: bottom and height of the stage boxes
xs = [0.6, 17.4, 39.4, 57.0, 72.4, 85.2]
ws = [15.6, 20.8, 16.4, 14.2, 11.6, 14.2]

# 1 test frame
box(xs[0], Y, ws[0], HB, '1', 'Test frame')
img, mask = mz['rawfabrid|image'], mz['rawfabrid|mask'].astype(bool)
_, th = thumb(img, xs[0] + 1.0, Y + HB - 5.0 - 9.0 * 0.8, ws[0] - 2.0, mask=mask)
body(xs[0] + 1.0, Y + 8.6, 'native resolution;\nmask (copper) for\ngrading; Practice A\nalso uses it')

# 2 cutting layouts
box(xs[1], Y, ws[1], HB, '2', 'Cut into detector input')
im2 = cz['schematic|image']
s = cm['scale']
grid = [tuple(c) for c in cz['schematic|grid']]
rel = [tuple(c) for c in cz['schematic|released']]
moved = [c for c in rel if c not in set(grid)]
tw = (ws[1] - 3.0) / 2
labels = ['R1 native frame', 'R2 256-px grid', 'A released crops', 'C4 ×4 downscale']
for k, lab in enumerate(labels):
    cx = xs[1] + 1.0 + (k % 2) * (tw + 1.0)
    cy = Y + HB - 3.6 - (k // 2 + 1) * 8.9
    if k == 3:
        from PIL import Image
        small = np.asarray(Image.fromarray(im2).resize((im2.shape[1] // 4, im2.shape[0] // 4), Image.BOX))
        a, h = thumb(small, cx, cy, tw)
    else:
        a, h = thumb(im2, cx, cy, tw)
    if k == 1:
        for y0, x0 in grid:
            a.add_patch(Rectangle((x0 * s, y0 * s), 256 * s, 256 * s, fill=False, ec='white', lw=0.35))
    if k == 2:
        for y0, x0 in moved:
            a.add_patch(Rectangle((x0 * s, y0 * s), 256 * s, 256 * s, fill=False, ec=S.COPPER, lw=0.6))
    ax.text(cx, cy - 0.6, lab, fontsize=5.6, color=S.INK, va='top')

# 3 frozen detectors
box(xs[2], Y, ws[2], HB, '3', 'Frozen detector')
for k, (fam, name) in enumerate([('patchcore', 'PatchCore-style,\nWRN-50 (primary)'), ('gauss', 'Shared Gaussian,\nWRN-50'),
                                 ('dino', 'DINOv2 ViT-S/14\ntoken kNN')]):
    yy = Y + HB - 5.0 - k * 4.3
    ax.plot(xs[2] + 1.9, yy - 0.9, {'patchcore': 'o', 'gauss': '^', 'dino': 's'}[fam], ms=7, color=S.DET[fam], mec='none')  # same shapes as the result figures
    body(xs[2] + 3.4, yy, name, size=6, color=S.INK)
body(xs[2] + 1.0, Y + 7.0, 'fitted on normal tiles\nonly, no defect image\nseen; R1 and R2 share\none bank')

# 4 stitched score map
box(xs[3], Y, ws[3], HB, '4', 'Stitch scores')
m = mz['rawfabrid|patchcore|R2']
lo, hi = np.percentile(m, [1, 99.5])
thumb(np.clip((m - lo) / (hi - lo), 0, 1), xs[3] + 1.0, Y + HB - 5.0 - 9.0 * 0.62, ws[3] - 2.0, cmap=S.CMAP)
body(xs[3] + 1.0, Y + 9.6, 'tiles fused by maximum;\nR1 and C4 maps resized\nto native size;\nuncovered pixels get\nthe lowest score')

# 5 grading
box(xs[4], Y, ws[4], HB, '5', 'Grade')
body(xs[4] + 1.0, Y + HB - 4.6, 'whole-frame\npixel AP on the\nfull native mask,\nper roll, cluster\nor category;\nAITEX pooled', color=S.INK)

# 6 contrast and decision
box(xs[5], Y, ws[5], HB, '6', 'Decide')
body(xs[5] + 1.0, Y + HB - 4.6, 'R2 − R1 per unit;\nbootstrap (2,000\ndraws) of each\ndataset\'s units;\nsign-flip test', color=S.INK)
ax.add_patch(FancyBboxPatch((xs[5] + 0.8, Y + 0.9), ws[5] - 1.6, 8.2, boxstyle='round,pad=0,rounding_size=0.8',
                            fc='white', ec=S.COBALT, lw=0.6))
body(xs[5] + 1.4, Y + 8.5, 'Rule: same sign on\n≥ 3 of 4 datasets,\neach |est| ≥ 0.01,\n95% CI excludes 0', size=5.7, color=S.COBALT)

for i in range(5):
    arrow(xs[i] + ws[i] + 0.05, Y + HB / 2, xs[i + 1] - 0.05, Y + HB / 2)

# label-only branch
box(17.4, 1.2, 66.4, 7.0, None, 'Label-only audit (no detector), from the same layouts', fc='white', ls=(0, (3, 2)))
body(18.6, 5.0, 'coverage of marked pixels; regions that no window shows whole; sliver patches under 1% defect;\n'
     'fate of each 8-connected defect region (whole, split, never seen)', size=6)
arrow(xs[1] + ws[1] / 2, Y - 0.05, xs[1] + ws[1] / 2, 8.85, ls='--')

S.save(fig, HERE / 'figures' / 'fig05_pipeline', keep_aspect=True)
