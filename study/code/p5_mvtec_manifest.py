"""Protocol 5 fallback dataset (Amendment 1, A7): MVTec AD 2 Fabric manifest, public test split only. No detector.
Usage: python3 p5_mvtec_manifest.py <.../fabric/fabric> <out.json>
"""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
from PIL import Image

root, out = Path(sys.argv[1]), Path(sys.argv[2])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
imgs, fails = {}, []
for p in sorted((root / 'train' / 'good').glob('*.png')):
    imgs[str(p.relative_to(root))] = {'split': 'train', 'label': 'normal', 'sha256': sha(p)}
for lab in ('good', 'bad'):
    for p in sorted((root / 'test_public' / lab).glob('*.png')):
        k = str(p.relative_to(root))
        obj = re.match(r'(\d+)_', p.name).group(1)
        rec = {'split': 'test', 'label': 'anomaly' if lab == 'bad' else 'normal', 'object': f'{lab}_{obj}', 'sha256': sha(p)}
        if lab == 'bad':
            m = root / 'test_public' / 'ground_truth' / 'bad' / (p.stem + '_mask.png')
            a = np.asarray(Image.open(m).convert('L'))
            with Image.open(p) as im:
                if a.shape != im.size[::-1]:
                    fails.append([k, 'mask size'])
            n = int((a >= 128).sum())
            if n == 0:
                fails.append([k, 'empty mask'])
            rec.update(mask=str(m.relative_to(root)), mask_sha256=sha(m), defect_pixels=n)
        imgs[k] = rec
test = [r for r in imgs.values() if r['split'] == 'test']
res = {'protocol': 'Protocol 5 fallback, MVTec AD 2 Fabric, test_public', 'decoder': 'mask value >= 128 (0/255 masks)',
       'resampling_unit': 'physical sample: all lighting/shift variants of one object ID resampled together',
       'counts': {'train': sum(r['split'] == 'train' for r in imgs.values()),
                  'test_anomaly': sum(r['label'] == 'anomaly' for r in test),
                  'test_normal': sum(r['label'] == 'normal' for r in test),
                  'test_objects': len({r['object'] for r in test})},
       'checks_failed': fails, 'images': imgs}
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ('counts', 'checks_failed')}))
