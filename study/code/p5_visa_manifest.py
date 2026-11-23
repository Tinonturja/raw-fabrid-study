"""Protocol 5 (Amendment 1, A1/A7): VisA cohort manifest and mask-decoder checks.

Decoder: VisA masks hold category IDs (0 background, >0 defect), so defect = value > 0.
Checks: every anomalous test image has a mask of the same size with at least one defect
pixel; every normal image has no mask; files hashed; 10 random overlays saved.
No detector is run. Usage: python3 p5_visa_manifest.py <visa_dir> <out.json> <overlay_dir>
"""
import csv, hashlib, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

root, out, ov = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = list(csv.DictReader(open(root/'split_csv'/'1cls.csv')))
images, stats, values = {}, defaultdict(Counter), Counter()
fails = []
for r in rows:
    p = root/r['image']
    rec = {'category': r['object'], 'split': r['split'], 'label': r['label'], 'sha256': sha(p)}
    stats[r['object']][f"{r['split']}_{r['label']}"] += 1
    if r['label'] == 'anomaly':
        m = root/r['mask']
        a = np.asarray(Image.open(m)); values.update(np.unique(a).tolist())
        with Image.open(p) as im:
            size = im.size
        if a.shape[::-1] != size: fails.append([r['image'], 'mask size differs', list(a.shape)])
        n = int((a > 0).sum())
        if n == 0: fails.append([r['image'], 'empty mask under >0'])
        if (a >= 128).sum() > 0: fails.append([r['image'], 'unexpected value >=128'])
        rec.update(mask=r['mask'], mask_sha256=sha(m), defect_pixels=n, size=list(size))
        stats[r['object']]['defect_pixels'] += n
    elif r['mask']:
        fails.append([r['image'], 'normal image has a mask'])
    images[r['image']] = rec

random.seed(13017)
for key in random.sample([k for k, v in images.items() if v['label'] == 'anomaly'], 10):
    im = Image.open(root/key).convert('RGB'); a = np.asarray(Image.open(root/images[key]['mask'])) > 0
    edge = a & ~(np.roll(a, 1, 0) & np.roll(a, -1, 0) & np.roll(a, 1, 1) & np.roll(a, -1, 1))
    arr = np.asarray(im).copy(); arr[edge] = [255, 0, 0]
    Image.fromarray(arr).save(ov/(key.replace('/', '__') + '.jpg'), quality=85)

result = {'protocol': 'Protocol 5, Amendment 1 (A1, A7), VisA cohort', 'split': 'split_csv/1cls.csv (official)',
          'decoder': 'mask value > 0 (category IDs); never a grayscale threshold',
          'mask_values_seen': sorted(values), 'checks_failed': fails,
          'per_category': {k: dict(v) for k, v in sorted(stats.items())}, 'images': images}
out.write_text(json.dumps(result, indent=1))
tot = Counter(); [tot.update({k: v for k, v in c.items() if k != 'defect_pixels'}) for c in stats.values()]
print(json.dumps({'mask_values_seen': sorted(values), 'checks_failed': len(fails), 'totals': dict(tot),
                  'defect_pixels_total': sum(c['defect_pixels'] for c in stats.values()),
                  'categories': len(stats)}, indent=1)); print(fails[:5])
