"""Qualitative score maps for the manuscript: native full frame (R1) vs tiled (R2) for the PatchCore-style and
DINOv2-token kNN detectors, using the frozen Protocol 5 banks and code paths. Selection rule (stated in the caption):
the unit with the largest DINOv2 R2 - R1 gain in each dataset (VisA cashew, RAW-FABRID Rollo7A), first defective
image in sorted order. Step 1 (Mac): --compute writes results_p5/qualitative/maps.npz. Step 2: --plot draws the figure.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'results_p5' / 'qualitative'
PICK = [('visa', 'cashew'), ('rawfabrid', 'Rollo7A')]


def compute(visa, raw):
    import torch
    from PIL import Image
    import p5_core as C, p5_models as M, p5_run as R
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    fams = [M.WRN(device), M.DINO(device)]
    arrays, meta = {}, {}
    for ds, unit in PICK:
        man = json.loads((HERE / 'manifests' / f'{ds}_manifest.json').read_text())
        if ds == 'visa':
            it = next(i for u, _, items in R.visa_units(Path(visa), man, [unit], 0) for i in items if i['defect'])
        else:
            it = next(i for u, _, items, _ in R.rawfabrid_units(Path(raw), man, [unit], 0) for i in items if i['defect'])
        scorers, _ = R.build_scorers(fams, None, HERE / 'results_p5' / ds / 'banks', unit, device)
        image = it['image']()
        mask = it['mask']()
        w, h = image.size
        grid = C.grid_layout(h, w)
        s = 768 / max(w, h)
        small = (max(1, round(w * s)), max(1, round(h * s)))
        arrays[f'{ds}|image'] = np.asarray(image.resize(small, Image.BILINEAR))
        arrays[f'{ds}|mask'] = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(small, Image.NEAREST)) > 0
        for fam in fams:
            scs = scorers[fam.name][:1]                       # PatchCore for WRN-50, kNN for DINOv2
            r1 = R.full_map(fam, scs, image)[scs[0].name]
            tm = R.tile_maps(fam, scs, image, grid)[scs[0].name]
            r2 = C.fuse_max(h, w, tm, grid)
            for lay, m in (('R1', r1), ('R2', r2)):
                arrays[f'{ds}|{scs[0].name}|{lay}'] = np.asarray(
                    Image.fromarray(m.astype(np.float32)).resize(small, Image.BILINEAR))
        meta[ds] = {'unit': unit, 'image_id': it['id'], 'size': [w, h]}
        print(ds, meta[ds], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'maps.npz', **arrays)
    (OUT / 'maps.json').write_text(json.dumps(meta, indent=1))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import figstyle as S
    S.apply()
    z = np.load(OUT / 'maps.npz')
    meta = json.loads((OUT / 'maps.json').read_text())
    cols = [('patchcore', 'R1', 'PatchCore-style, native'), ('patchcore', 'R2', 'PatchCore-style, tiled'),
            ('dino', 'R1', 'DINOv2 kNN, native'), ('dino', 'R2', 'DINOv2 kNN, tiled')]
    fig, axes = plt.subplots(len(PICK), 5, figsize=(S.DOUBLE, 3.2))
    for r, (ds, unit) in enumerate(PICK):
        img, mask = z[f'{ds}|image'], z[f'{ds}|mask']
        ax = axes[r, 0]
        ax.imshow(img)
        ax.contour(mask, levels=[0.5], colors=S.COPPER, linewidths=0.9)
        ax.set_ylabel(f"{'VisA' if ds == 'visa' else 'RAW-FABRID'}\n{unit}", fontsize=8, fontweight='bold')
        if r == 0:
            ax.set_title('Image and mask', fontsize=7.5, loc='center')
        for c, (sc, lay, title) in enumerate(cols, start=1):
            m = z[f'{ds}|{sc}|{lay}']
            pair = np.concatenate([z[f'{ds}|{sc}|R1'].ravel(), z[f'{ds}|{sc}|R2'].ravel()])
            lo, hi = np.percentile(pair, [1, 99.5])           # shared scale per detector and image
            ax = axes[r, c]
            ax.imshow(np.clip((m - lo) / (hi - lo + 1e-12), 0, 1), cmap=S.CMAP)
            ax.contour(mask, levels=[0.5], colors='white', linewidths=0.6, linestyles='--')
            if r == 0:
                ax.set_title(title, fontsize=7.5, loc='center')
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    import matplotlib as mpl
    cax = fig.add_axes([0.92, 0.18, 0.008, 0.6])
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=S.CMAP), cax=cax, ticks=[0, 1])
    cb.ax.set_yticklabels(['low', 'high'], fontsize=6.5)
    cb.outline.set_visible(False)
    cb.set_label('anomaly score, per detector and image', fontsize=6.5)
    fig.subplots_adjust(left=0.06, right=0.9, wspace=0.04, hspace=0.12)
    out = HERE / 'manuscript_v2' / 'figures'
    fig.savefig(out / 'fig7_score_maps.png', dpi=300, bbox_inches='tight')
    fig.savefig(out / 'fig7_score_maps.pdf', bbox_inches='tight')
    print('wrote fig7', meta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--compute', action='store_true')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--visa')
    ap.add_argument('--raw')
    a = ap.parse_args()
    if a.compute:
        compute(a.visa, a.raw)
    if a.plot:
        plot()
