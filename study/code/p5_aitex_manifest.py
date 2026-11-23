"""Protocol 5 (Amendment 1, A7): AITEX cohort manifest.

Reproduces the original cohort exactly from the saved lists in aitex_external.json,
re-hashes every file on disk, recomputes the defect-pixel and unseen-pixel totals with
the original decoder (>=128, OR over masks) and the original floor-division crop rule,
and writes a frozen manifest. No detector is run.
Usage: python3 p5_aitex_manifest.py <aitex_data_dir> <aitex_external.json> <out.json>
"""
import hashlib, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

data, saved, out = Path(sys.argv[1]), json.load(open(sys.argv[2])), Path(sys.argv[3])

def gray(p):
    with Image.open(p) as im:
        return np.asarray(im.convert('L'))

def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

masks = defaultdict(list)
for p in sorted((data/'Mask_images').rglob('*.png')):
    m = re.fullmatch(r'(\d+_\d+_\d+)_mask\d*', p.stem, flags=re.I)
    assert m, p.name
    masks[m[1]].append(p)
paths = {p.name: p for d in ('Defect_images', 'NODefect_images') for p in (data/d).rglob('*.png')}

def guided_layout(mask):  # original rule, floor division
    coords = []
    for x in range(0, 4096, 256):
        cols = np.flatnonzero(mask[:, x:x+256].any(axis=0))
        coords.append(int(np.clip(x + (cols[0]+cols[-1])//2 - 128, 0, 3840)) if len(cols) else x)
    return coords

roles = {'bank_train_clean': saved['train_clean'], 'calibration_clean': saved['calibration_clean'],
         'evaluation_clean': saved['evaluation_clean'], 'evaluation_defective': saved['defective']}
names = [n for v in roles.values() for n in v]
assert len(names) == len(set(names)), 'image in more than one role'

rows, fg_total, unseen_total, mismatch = {}, 0, 0, []
for role, lst in roles.items():
    for n in sorted(lst):
        img = gray(paths[n]); assert img.shape == (256, 4096), n
        if hashlib.sha256(img.tobytes()).hexdigest() != saved['source_hashes'][n]:
            mismatch.append(n)
        stem = n[:-4]; fabric = stem.split('_')[2]
        row = {'role': role, 'fabric': fabric, 'path': str(paths[n].relative_to(data)), 'sha256': file_sha(paths[n])}
        if role == 'evaluation_defective':
            fm = masks[stem]; assert fm, n
            fg = np.logical_or.reduce([gray(p) >= 128 for p in fm]); assert fg.any(), n
            seen = np.zeros_like(fg)
            for x in guided_layout(fg):
                seen[:, x:x+256] = True
            assert guided_layout(fg) == [c[1] for c in saved['guided_coordinates'][n]], n
            fg_total += int(fg.sum()); unseen_total += int((fg & ~seen).sum())
            row.update(masks=[{'path': str(p.relative_to(data)), 'sha256': file_sha(p)} for p in fm],
                       defect_pixels=int(fg.sum()), unseen_defect_pixels_original_layout=int((fg & ~seen).sum()))
        rows[n] = row

result = {
    'protocol': 'Protocol 5, Amendment 1 (A7), AITEX cohort',
    'source_record': {'file': 'aitex_external.json', 'protocol_sha256': saved['protocol_sha256'], 'seed': saved['seed']},
    'decoder': 'mask value >= 128, logical OR over _mask1/_mask2; converted to grayscale (L) first',
    'layout_rule_practice_A': 'original guided_layout: left = clip(x + (first+last)//2 - 128, 0, 3840) per fixed 256 column',
    'exclusions': saved['audit_issues'], 'fabrics_omitted': saved['fabrics_omitted'],
    'counts': {r: len(v) for r, v in roles.items()},
    'fabrics_in_cohort': sorted({r['fabric'] for r in rows.values()}),
    'defect_pixels_total': fg_total, 'unseen_defect_pixels_original_layout': unseen_total,
    'pixel_hash_mismatches_vs_saved': mismatch,
    'images': rows,
}
out.write_text(json.dumps(result, indent=1))
print(json.dumps({k: result[k] for k in ('counts', 'fabrics_in_cohort', 'exclusions', 'fabrics_omitted',
      'defect_pixels_total', 'unseen_defect_pixels_original_layout', 'pixel_hash_mismatches_vs_saved')}, indent=1))
