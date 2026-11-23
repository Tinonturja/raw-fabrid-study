"""Label-only audit of cutting practices (no detector). Practice A: defect-aware crops; B: blind grid with per-patch labels.
Usage: python3 cut_audit.py <RAW_FABRID.zip> <aitex_mask_dir> <out.json>"""
import glob, io, json, re, sys, zipfile
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy.ndimage import label

S = 256
EIGHT = np.ones((3, 3), bool)


def grid(h, w):
    return [(y, x) for y in range(0, h - S + 1, S) for x in range(0, w - S + 1, S)]


def audit_frame(mask, windows):
    lab, n = label(mask, structure=EIGHT)
    cov = np.zeros(mask.shape, np.uint8)
    for y, x in windows:
        cov[y:y + S, x:x + S] += 1
    out = {'fg': int(mask.sum()), 'unseen_fg': int((mask & (cov == 0)).sum()), 'regions': n,
           'regions_unseen': 0, 'regions_split': 0, 'patches_pos': 0, 'patches_sliver': 0, 'pos_fractions': []}
    for r in range(1, n + 1):
        reg = lab == r
        if not (reg & (cov > 0)).any():
            out['regions_unseen'] += 1
        hits = sum(reg[y:y + S, x:x + S].any() for y, x in windows)
        whole = any(reg[y:y + S, x:x + S].sum() == reg.sum() for y, x in windows)
        out['regions_split'] += int(hits >= 2 and not whole)
    for y, x in windows:
        a = int(mask[y:y + S, x:x + S].sum())
        if a:
            out['patches_pos'] += 1
            out['patches_sliver'] += int(a < 0.01 * S * S)
            out['pos_fractions'].append(a / (S * S))
    return out


def aitex_guided(mask):
    """Author's constructed AITEX rule: centre each foreground window on the midpoint of its marked columns."""
    wins = []
    for _, x in grid(*mask.shape):
        cols = np.flatnonzero(mask[:, x:x + S].any(0))
        if len(cols):
            mid = x + (cols[0] + cols[-1]) / 2
            x = int(min(max(round(mid - S / 2), 0), mask.shape[1] - S))
        wins.append((0, x))
    return wins


def summarise(rows):
    t = lambda k: sum(r[k] for r in rows)
    fr = [f for r in rows for f in r['pos_fractions']]
    return {'frames': len(rows), 'marked_pixels': t('fg'), 'unseen_pct': round(100 * t('unseen_fg') / t('fg'), 3),
            'regions': t('regions'), 'regions_fully_unseen': t('regions_unseen'),
            'regions_split_no_whole_view_pct': round(100 * t('regions_split') / t('regions'), 1),
            'positive_patches': t('patches_pos'), 'sliver_patches_pct': round(100 * t('patches_sliver') / max(t('patches_pos'), 1), 1),
            'median_defect_fraction_of_positive_patch_pct': round(100 * float(np.median(fr)), 2) if fr else None}


def main(zpath, aitex_dir, out):
    res = {}
    z = zipfile.ZipFile(zpath)
    rel = defaultdict(list)
    for r in json.loads(z.read('RAW_FABRID/RAW_FABRID_Patches_COCO.json'))['images']:
        rel[r['parent']].append((r['r_offset'], r['c_offset']))
    names = [n[len('RAW_FABRID/masks/'):] for n in z.namelist() if n.startswith('RAW_FABRID/masks/') and n.endswith('.png')]
    images = set(n[len('RAW_FABRID/images/'):] for n in z.namelist() if n.startswith('RAW_FABRID/images/'))
    A, Bg = [], []
    for n in sorted(set(names) & images):
        m = np.asarray(Image.open(io.BytesIO(z.read('RAW_FABRID/masks/' + n))).convert('L')) >= 128
        if not m.any():
            continue
        A.append(audit_frame(m, rel[n])); Bg.append(audit_frame(m, grid(*m.shape)))
    res['RAW-FABRID'] = {'A_released_defect_aware': summarise(A), 'B_blind_grid_28': summarise(Bg)}
    by = defaultdict(list)
    for p in glob.glob(aitex_dir + '/**/*.png', recursive=True):
        mm = re.match(r'(\d+_\d+_\d+)_mask', p.split('/')[-1])
        if mm:
            by[mm[1]].append(p)
    A, Bg = [], []
    for k in sorted(by):
        ms = [np.asarray(Image.open(p).convert('L')) for p in by[k]]
        if any(x.shape != (256, 4096) for x in ms):
            continue
        m = np.logical_or.reduce([x > 0 for x in ms])
        if not m.any():
            continue
        A.append(audit_frame(m, aitex_guided(m))); Bg.append(audit_frame(m, grid(*m.shape)))
    res['AITEX'] = {'A_constructed_defect_aware': summarise(A), 'B_blind_grid_16': summarise(Bg)}
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
