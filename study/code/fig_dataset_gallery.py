"""Dataset gallery: one defective test image per dataset, all drawn at ONE pixel scale with the edge-aligned
256-pixel grid, so frame geometry (exact tiling vs overlapping last row/column) is visible at a glance.

Image choice is by rule, not by appearance:
  RAW-FABRID and VisA use the images of the score-map figure (Rollo7A first defective frame; VisA cashew 000.JPG);
  AITEX and MVTec AD 2 Fabric use the first defective evaluation image in sorted order.
  TSFabrics frames are not stored on this machine (the run used Kaggle), so it is drawn as geometry only.

Two steps:
  python fig_dataset_gallery.py --compute --aitex <dir> --visa <dir> --raw <zip> --mvtec <dir>   (Mac, frozen manifests)
  python fig_dataset_gallery.py --plot
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'results_p5' / 'qualitative'
TSF_SIZE = (800, 640)   # TSFabrics frame size (w, h), Table 2


def compute(aitex, visa, raw, mvtec):
    import p5_core as C
    import p5_run as R
    man = lambda d: json.loads((HERE / 'manifests' / f'{d}_manifest.json').read_text())
    z, meta = {}, {}

    def put(tag, item, released=None):
        img = np.asarray(item['image']().convert('L'))
        m = np.asarray(item['mask']()).astype(bool)
        assert img.shape == m.shape, (tag, img.shape, m.shape)
        z[f'{tag}|image'], z[f'{tag}|mask'] = img, m
        if released is not None:
            z[f'{tag}|released'] = np.array(released)
        h, w = m.shape
        meta[tag] = {'id': item['id'], 'size': [w, h], 'tiles': len(C.grid_layout(h, w)),
                     'exact': w % 256 == 0 and h % 256 == 0, 'defect_px': int(m.sum())}
        print(tag, meta[tag], flush=True)

    _, _, items = next(R.aitex_units(Path(aitex), man('aitex'), 0))
    it = next(i for i in items if i['defect'])
    put('aitex', it, C.aitex_original_layout(np.asarray(it['mask']()).astype(bool)))

    for roll, _, items, _ in R.rawfabrid_units(Path(raw), man('rawfabrid'), ['Rollo7A'], 0):
        it = next(i for i in items if i['defect'])
        put('rawfabrid', it, it['A'])

    for cat, _, items in R.visa_units(Path(visa), man('visa'), ['cashew'], 0):
        put('visa', next(i for i in items if i['id'].endswith('Anomaly/000.JPG')))

    mv = HERE / 'manifests' / 'mvtec_fabric_manifest.json'
    if mv.exists():
        _, _, items = next(R.mvtec_units(Path(mvtec), json.loads(mv.read_text()), 0))
        put('mvtec', next(i for i in items if i['defect']))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'gallery.npz', **z)
    (OUT / 'gallery.json').write_text(json.dumps(meta, indent=1))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import ListedColormap
    from scipy.ndimage import binary_dilation
    import figstyle as S
    import p5_core as C
    S.apply()
    z = np.load(OUT / 'gallery.npz')
    meta = json.loads((OUT / 'gallery.json').read_text())
    names = {'aitex': 'AITEX', 'rawfabrid': 'RAW-FABRID', 'visa': 'VisA (cashew)', 'mvtec': 'MVTec AD 2 Fabric',
             'tsf': 'TSFabrics'}
    meta['tsf'] = {'size': list(TSF_SIZE), 'tiles': len(C.grid_layout(TSF_SIZE[1], TSF_SIZE[0])), 'exact': False}

    # layout in native pixels: AITEX strip on top; MVTec left; RAW-FABRID over VisA + TSFabrics on the right
    g = 150                      # gap between frames, native pixels
    hdr = 330                    # room for each frame's two-line title
    pos = {'aitex': (0, 0)}
    y2 = 256 + g + hdr
    pos['mvtec'] = (0, y2)
    xr = 2448 + g * 2
    pos['rawfabrid'] = (xr, y2)
    y3 = y2 + 1024 + g + hdr
    pos['visa'] = (xr, y3)
    pos['tsf'] = (xr + 1274 + g, y3)
    W = max(xr + 1274 + g + 800, 4096)
    H = max(y3 + 1176, y2 + 2048)
    fig = plt.figure(figsize=(S.DOUBLE, S.DOUBLE * (H + hdr) / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, -hdr + 40); ax.set_aspect('equal'); ax.axis('off')

    for k, (x0, y0) in pos.items():
        if k not in meta:
            continue
        w, h = meta[k]['size']
        if k in z.files or f'{k}|image' in z.files:
            img, m = z[f'{k}|image'], z[f'{k}|mask']
            ax.imshow(img, cmap='gray', extent=(x0, x0 + w, y0 + h, y0), interpolation='lanczos', zorder=1)
            # masks are dilated for visibility at this scale (stated in the caption)
            md = binary_dilation(m, iterations=max(3, round(0.004 * max(w, h))))
            ax.imshow(np.ma.masked_where(~md, md), cmap=ListedColormap([S.JADE]), alpha=0.7,
                      extent=(x0, x0 + w, y0 + h, y0), interpolation='nearest', zorder=2)
        else:
            ax.add_patch(Rectangle((x0, y0), w, h, fc='#efece6', ec='none', zorder=1))
            ax.text(x0 + w / 2, y0 + 128, 'geometry only;\nframes not stored\nlocally', ha='center',
                    va='center', fontsize=5.3, color=S.INK2, zorder=6,
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2))
        # edge-aligned grid; the band where the last row / column overlaps its neighbour is shaded copper
        ys, xs = C.grid_positions(h, 256), C.grid_positions(w, 256)
        for xx in xs[1:]:
            ax.plot([x0 + xx, x0 + xx], [y0, y0 + h], color='white', lw=0.45, zorder=3)
        for yy in ys[1:]:
            ax.plot([x0, x0 + w], [y0 + yy, y0 + yy], color='white', lw=0.45, zorder=3)
        if w % 256:
            ax.add_patch(Rectangle((x0 + xs[-1], y0), (xs[-2] + 256) - xs[-1], h, fc=S.COPPER, alpha=0.35, ec='none', zorder=2.5))
        if h % 256:
            ax.add_patch(Rectangle((x0, y0 + ys[-1]), w, (ys[-2] + 256) - ys[-1], fc=S.COPPER, alpha=0.35, ec='none', zorder=2.5))
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=S.INK2, lw=0.5, zorder=4))
        tiling = 'exact tiling' if meta[k]['exact'] else 'edge overlap'
        ax.text(x0, y0 - 175, f"{names[k]}", fontsize=7.5, fontweight='bold', va='bottom', ha='left')
        ax.text(x0, y0 - 45, f"{w} × {h} px, {meta[k]['tiles']} tiles, {tiling}", fontsize=6.3,
                color=S.INK2, va='bottom', ha='left')
    # scale reference: one 256-pixel tile
    sx, sy = pos['visa'][0] - g * 0, H + 0
    from matplotlib.patches import Patch
    handles = [Patch(fc=S.JADE, alpha=0.7, label='marked defect (dilated for visibility)'),
               Patch(fc=S.COPPER, alpha=0.35, label='overlap of the edge-aligned last row or column'),
               Patch(fc='#8c8a84', ec='white', label='256-pixel tile grid')]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.0), handlelength=1.4)
    S.save(fig, HERE / 'figures' / 'fig01_datasets', keep_aspect=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--compute', action='store_true')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--aitex'); ap.add_argument('--visa'); ap.add_argument('--raw'); ap.add_argument('--mvtec')
    a = ap.parse_args()
    if a.compute:
        compute(a.aitex, a.visa, a.raw, a.mvtec)
    if a.plot:
        plot()
