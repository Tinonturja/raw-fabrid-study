"""Protocol 5 fragmentation analysis (Amendment 1, A3), label-only. Uses the frozen P5 adapters, so cohort,
manifest checks and mask decoders are identical to the main run. For every 8-connected component of each native
defect mask and each layout (R2 grid; released/guided practice A where it exists):
  - retained fraction: the largest share of the component inside any single window;
  - cut: no single window holds the whole component (component touches a window border inside the frame);
  - oversize: bounding box larger than 256 pixels on either side (no tile can hold it), reported separately.
Positive patches (>= 1 marked pixel) under 1% defect are split into slivers from cut components and natural
small components that are whole inside the patch.
Usage: python p5_fragmentation.py --dataset aitex|visa|rawfabrid --data <path> --out <json>"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import label, find_objects

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_core as C
import p5_run as R

S = 256
EIGHT = np.ones((3, 3), int)


def audit(mask, windows):
    lab, n = label(mask, structure=EIGHT)
    comps = []
    win_counts = []                                   # per window: {component: pixels}
    for y, x in windows:
        sub = lab[y:y + S, x:x + S]
        ids, cnt = np.unique(sub[sub > 0], return_counts=True)
        win_counts.append(dict(zip(ids.tolist(), cnt.tolist())))
    for r, sl in enumerate(find_objects(lab), start=1):
        size = int((lab[sl] == r).sum())
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        best = max((wc.get(r, 0) for wc in win_counts), default=0)
        comps.append({'px': size, 'oversize': bool(bh > S or bw > S), 'retained': best / size, 'cut': best < size})
    patches = []
    for wc in win_counts:
        a = sum(wc.values())
        if a:
            from_cut = any(comps[c - 1]['cut'] and v < comps[c - 1]['px'] for c, v in wc.items())
            patches.append({'frac': a / (S * S), 'from_cut': bool(from_cut)})
    return comps, patches


def summarise(comps, patches):
    fit = [c for c in comps if not c['oversize']]
    cut = [c for c in fit if c['cut']]
    sl = [p for p in patches if p['frac'] < 0.01]
    q = lambda v: [round(float(x), 3) for x in np.quantile(v, [0.25, 0.5, 0.75])] if len(v) else None
    return {'components': len(comps), 'oversize': len(comps) - len(fit),
            'fit_components': len(fit), 'fit_cut': len(cut),
            'fit_cut_pct': round(100 * len(cut) / max(len(fit), 1), 1),
            'fit_cut_pixel_share_pct': round(100 * sum(c['px'] for c in cut) / max(sum(c['px'] for c in fit), 1), 1),
            'retained_fraction_of_cut_q25_q50_q75': q([c['retained'] for c in cut]),
            'positive_patches': len(patches), 'slivers_lt1pct': len(sl),
            'slivers_from_cut': sum(p['from_cut'] for p in sl),
            'slivers_natural_small': sum(not p['from_cut'] for p in sl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    man = json.loads((R.HERE / 'manifests' / f'{a.dataset}_manifest.json').read_text())
    if a.dataset == 'aitex':
        units = ((u, items) for u, _, items in R.aitex_units(Path(a.data), man, 0))
    elif a.dataset == 'visa':
        units = ((u, items) for u, _, items in R.visa_units(Path(a.data), man, None, 0))
    else:
        units = ((u, items) for u, _, items, _ in R.rawfabrid_units(Path(a.data), man, None, 0))
    res = {'dataset': a.dataset, 'units': {}}
    allc = {'R2': ([], []), 'A': ([], [])}
    for unit, items in units:
        acc = {'R2': ([], []), 'A': ([], [])}
        for it in items:
            if not it['defect']:
                continue
            m = it['mask']()
            h, w = m.shape
            lays = {'R2': C.grid_layout(h, w)}
            if isinstance(it['A'], list):
                lays['A'] = it['A']
            elif it['A'] is True:
                lays['A'] = C.aitex_original_layout(m)
            for lay, wins in lays.items():
                c, p = audit(m, wins)
                acc[lay][0].extend(c); acc[lay][1].extend(p)
                allc[lay][0].extend(c); allc[lay][1].extend(p)
        res['units'][unit] = {lay: summarise(*v) for lay, v in acc.items() if v[0]}
        print(unit, json.dumps(res['units'][unit]), flush=True)
    res['all'] = {lay: summarise(*v) for lay, v in allc.items() if v[0]}
    print('ALL', json.dumps(res['all'], indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
