"""Label-only cutting audit for VisA (all 12 categories, all anomalous test images).
Grid: 256-px squares at stride 256, plus an edge-aligned last row/column so the whole image is covered."""
import glob, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cut_audit import audit_frame, summarise, S


def grid_full(h, w):
    ys = sorted(set(list(range(0, h - S + 1, S)) + [h - S]))
    xs = sorted(set(list(range(0, w - S + 1, S)) + [w - S]))
    return [(y, x) for y in ys for x in xs]


def defect_aware(mask, wins):
    """RAW-FABRID-style rule: recentre any square whose defect touches its border onto that defect's centre."""
    h, w = mask.shape
    out = []
    for y, x in wins:
        sub = mask[y:y + S, x:x + S]
        if sub.any() and (sub[0].any() or sub[-1].any() or sub[:, 0].any() or sub[:, -1].any()):
            ys, xs = np.nonzero(sub)
            cy, cx = y + (ys.min() + ys.max()) / 2, x + (xs.min() + xs.max()) / 2
            y = int(min(max(round(cy - S / 2), 0), h - S)); x = int(min(max(round(cx - S / 2), 0), w - S))
        out.append((y, x))
    return out


root, outp = sys.argv[1], sys.argv[2]
res, allA, allB = {}, [], []
for cat in sorted(p.name for p in Path(root).iterdir() if (p / 'Data').is_dir()):
    A, Bg = [], []
    for mp in sorted(glob.glob(f'{root}/{cat}/Data/Masks/Anomaly/*.png')):
        m = np.asarray(Image.open(mp).convert('L')) > 0
        if not m.any():
            continue
        g = grid_full(*m.shape)
        Bg.append(audit_frame(m, g)); A.append(audit_frame(m, defect_aware(m, g)))
    res[cat] = {'B_grid': summarise(Bg), 'A_defect_aware': summarise(A), 'image_size': list(m.shape)}
    allA += A; allB += Bg
res['ALL'] = {'B_grid': summarise(allB), 'A_defect_aware': summarise(allA)}
json.dump(res, open(outp, 'w'), indent=1)
print('category      size         B:split%  B:sliver%  B:median%   A:unseen%  A:split%  A:sliver%')
for c, r in res.items():
    b, a = r['B_grid'], r['A_defect_aware']
    print(f"{c:12s} {str(r.get('image_size','')):12s} {b['regions_split_no_whole_view_pct']:8} {b['sliver_patches_pct']:9} {b['median_defect_fraction_of_positive_patch_pct']:9}   {a['unseen_pct']:9} {a['regions_split_no_whole_view_pct']:8} {a['sliver_patches_pct']:9}")
