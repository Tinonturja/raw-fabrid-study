"""Figure 1 (methods schematic: the four ways of cutting a frame) and Figure 1b-style coverage illustration, drawn
from results_p5/qualitative/coverage_examples.npz in the shared style. Frames chosen by rule (see fig_coverage_example.py).
Usage: python3 fig_methods_coverage.py"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S

S.apply()
HERE = Path(__file__).resolve().parents[1]
Q = HERE / 'results_p5' / 'qualitative'
OUT = HERE / 'manuscript_v2' / 'figures'
z = np.load(Q / 'coverage_examples.npz')
meta = json.loads((Q / 'coverage_examples.json').read_text())
T = 256


def overlay(ax, img, mask, unseen=None, alpha=0.9):
    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
    ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([S.JADE]), alpha=alpha, interpolation='nearest')
    if unseen is not None and unseen.any():
        ax.imshow(np.ma.masked_where(~unseen, unseen), cmap=ListedColormap([S.COPPER]), alpha=1, interpolation='nearest')


def boxes(ax, coords, s, color, lw=0.8, ls='-'):
    for y, x in coords:
        ax.add_patch(Rectangle((x * s, y * s), T * s, T * s, fill=False, ec=color, lw=lw, ls=ls))


def clean(ax, title):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=7.5)


def fig1():
    k = 'maxunseen'
    img, mask, uns = z[f'{k}|image'], z[f'{k}|mask'], z[f'{k}|unseen']
    s = meta[k]['scale']
    grid = [tuple(c) for c in z[f'{k}|grid']]
    rel = [tuple(c) for c in z[f'{k}|released']]
    moved = [c for c in rel if c not in set(grid)]
    h, w = mask.shape
    fig, axes = plt.subplots(2, 2, figsize=(S.DOUBLE, 4.3), gridspec_kw={'wspace': 0.04, 'hspace': 0.22})
    a = axes.ravel()
    overlay(a[0], img, mask)
    a[0].add_patch(Rectangle((0, 0), w - 1, h - 1, fill=False, ec=S.COBALT, lw=1.6))
    clean(a[0], '(a)  R1: native full frame')
    overlay(a[1], img, mask)
    boxes(a[1], grid, s, 'white', lw=0.7)
    clean(a[1], f'(b)  R2: blind grid, {len(grid)} tiles of 256 px')
    overlay(a[2], img, mask, uns)
    boxes(a[2], [c for c in rel if c in set(grid)], s, 'white', lw=0.5, ls=(0, (2, 2)))
    boxes(a[2], moved, s, S.COPPER, lw=1.1)
    clean(a[2], f'(c)  A: released defect-aware crops ({len(moved)} moved)')
    from PIL import Image
    nw, nh = meta[k]['size']
    small = np.asarray(Image.fromarray(img).resize((nw // 4, nh // 4), Image.BOX))   # true x4 of the native frame
    a[3].imshow(small, extent=(0, w, h, 0), interpolation='nearest')
    a[3].imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([S.JADE]), alpha=0.9, interpolation='nearest')
    clean(a[3], '(d)  C4: whole frame downscaled ×4')
    handles = [Patch(color=S.JADE, label='marked defect pixels'), Patch(color=S.COPPER, label='defect pixels outside every released crop'),
               Patch(fill=False, ec=S.COPPER, lw=1.1, label='released crop moved off the grid')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(OUT / 'fig01_layouts.png'); fig.savefig(OUT / 'fig01_layouts.pdf'); plt.close(fig)


def fig_cov():
    k = 'maxunseen'
    img, mask, uns = z[f'{k}|image'], z[f'{k}|mask'], z[f'{k}|unseen']
    s = meta[k]['scale']
    rel = [tuple(c) for c in z[f'{k}|released']]
    grid = set(tuple(c) for c in z[f'{k}|grid'])
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.25), gridspec_kw={'width_ratios': [1.75, 1], 'wspace': 0.05})
    ax = axes[0]
    overlay(ax, img, mask, uns)
    boxes(ax, [c for c in rel if c in grid], s, 'white', lw=0.45, ls=(0, (2, 2)))
    boxes(ax, [c for c in rel if c not in grid], s, S.COPPER, lw=1.0)
    ys, xs = np.nonzero(uns)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    r = 70
    y0, y1 = max(0, cy - r), min(mask.shape[0], cy + r)
    x0, x1 = max(0, cx - int(r * 1.6)), min(mask.shape[1], cx + int(r * 1.6))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=S.INK, lw=1.0))
    clean(ax, f"(a)  {100 * meta[k]['unseen_px'] / meta[k]['defect_px']:.0f}% of this frame's defect pixels lie outside every released crop")
    ax2 = axes[1]
    overlay(ax2, img[y0:y1, x0:x1], mask[y0:y1, x0:x1], uns[y0:y1, x0:x1])
    clean(ax2, '(b)  detail')
    for sp in ax2.spines.values():
        sp.set_visible(True); sp.set_color(S.INK); sp.set_linewidth(1.0)
    handles = [Patch(color=S.JADE, label='defect pixels seen by a released crop'),
               Patch(color=S.COPPER, label='defect pixels no released crop sees'),
               Patch(fill=False, ec=S.COPPER, lw=1.0, label='crop moved off the grid')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(OUT / 'fig02_coverage_gap.png'); fig.savefig(OUT / 'fig02_coverage_gap.pdf'); plt.close(fig)


if __name__ == '__main__':
    fig1(); fig_cov()
    print('ok', meta)
