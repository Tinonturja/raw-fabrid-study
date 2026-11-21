"""Protocol 5 (Amendment 1, A7): RAW-FABRID manifest. No detector is run.

Per held-out roll: the 1,000 normal reference tiles of Protocol 1 (baseline_seed_13017.json), the test
frames of the 255-frame pixel cohort (202 defective + 53 clean evaluation), the released crop offsets
(practice A), SHA-256 of every archive member used, and the Protocol 1 WRN-50 bank hash to be reused.
Usage: python3 p5_rawfabrid_manifest.py <RAW_FABRID.zip> <Project files/results> <Protocol 1 results dir> <out.json>
"""
import csv, hashlib, io, json, sys, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

zpath, res, p1, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
FIXED = [(y, x) for y in range(0, 769, 256) for x in range(0, 1537, 256)]


def rows(z, name):
    with z.open('RAW_FABRID/RAW_FABRID_' + name + '_Metadata.csv') as s:
        return list(csv.DictReader(io.TextIOWrapper(s, encoding='utf-8-sig')))


sha = lambda b: hashlib.sha256(b).hexdigest()
folds = json.loads((res / 'baseline_seed_13017.json').read_text())['folds']
loc = json.loads((res / 'localization_study.json').read_text())
defective, ev = set(loc['defective_frames']), set(loc['evaluation_clean'])
assert len(defective) == 202 and len(ev) == 53

with zipfile.ZipFile(zpath) as z:
    meta = {r['filename']: r for r in rows(z, 'HighRes')}
    patches = rows(z, 'Patches')
    released = defaultdict(list)
    for r in json.loads(z.read('RAW_FABRID/RAW_FABRID_Patches_COCO.json'))['images']:
        released[r['parent']].append([r['r_offset'], r['c_offset']])
    rolls = sorted({r['roll_id'] for r in meta.values()})
    result = {'protocol': 'Protocol 5, Amendment 1 (A7), RAW-FABRID', 'archive': str(zpath.name),
              'decoder': 'mask value >= 128', 'cohort': '255 pixel-evaluation frames: 202 defective + 53 clean evaluation',
              'rolls': {}}
    fg_total = unseen_total = 0
    for roll in rolls:
        fold = folds['held_out_' + roll]
        refs = [patches[i] for i in fold['train_patch_indices']]
        assert len(refs) == 1000 and all(r['type'] == 'train_good' and meta[r['parent_raw']]['roll_id'] != roll
                                         and meta[r['parent_raw']]['has_defect'].lower() == 'false' for r in refs)
        frames = sorted(p for p in defective | ev if meta[p]['roll_id'] == roll)
        recs = {}
        for p in frames:
            img = z.read('RAW_FABRID/images/' + p)
            rec = {'defect': p in defective, 'image_sha256': sha(img), 'released': sorted(released[p])}
            assert len(released[p]) == 28
            if p in defective:
                mb = z.read('RAW_FABRID/masks/' + p)
                m = np.asarray(Image.open(io.BytesIO(mb)).convert('L')) >= 128
                cov = np.zeros(m.shape, bool)
                for y, x in released[p]:
                    cov[y:y + 256, x:x + 256] = True
                rec.update(mask_sha256=sha(mb), defect_px=int(m.sum()), unseen_defect_px=int((m & ~cov).sum()))
                fg_total += int(m.sum()); unseen_total += int((m & ~cov).sum())
            else:
                assert sorted(map(tuple, released[p])) == sorted(FIXED)
            recs[p] = rec
        p1rec = json.loads((p1 / f'{roll}.json').read_text())['fit']
        result['rolls'][roll] = {
            'refs': [{'patch_name': r['patch_name'], 'sha256': sha(z.read('RAW_FABRID/MVTec/train/good/' + r['patch_name']))}
                     for r in refs],
            'frames': recs, 'protocol1_wrn50_bank_sha256': p1rec['bank_sha256']}
        print(roll, len(frames), 'frames', flush=True)
    result['defect_pixels_total'] = fg_total
    result['unseen_defect_pixels_released'] = unseen_total
    out.write_text(json.dumps(result, indent=1))
    print('defect pixels', fg_total, 'unseen under released crops', unseen_total, f'({100 * unseen_total / fg_total:.3f}%)')
