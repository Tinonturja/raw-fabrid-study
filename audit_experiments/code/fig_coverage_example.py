"""Extract the RAW-FABRID frame with the most defect pixels outside every released crop (rule-based choice, no visual
selection), plus the first defective frame of Rollo7A for the methods schematic. Saves downsampled image, mask and window
coordinates to results_p5/qualitative/coverage_examples.npz (Mac step; frozen adapters and manifest checks)."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_core as C
import p5_run as R

zpath = Path(sys.argv[1])
man = json.loads((R.HERE / 'manifests' / 'rawfabrid_manifest.json').read_text())
best, first = None, None
for roll, _, items, _ in R.rawfabrid_units(zpath, man, None, 0):
    for it in items:
        if not it['defect']:
            continue
        m = it['mask']()
        h, w = m.shape
        cov = C.coverage(h, w, it['A'])
        unseen = int((m & (cov == 0)).sum())
        if roll == 'Rollo7A' and first is None:
            first = (dict(it, image_obj=it['image'](), mask_arr=m), unseen)
        if best is None or unseen > best[1]:
            best = (dict(it, image_obj=it['image'](), mask_arr=m), unseen)
out = {}
meta = {}
for tag, (it, unseen) in (('maxunseen', best), ('schematic', first)):
    im = it['image_obj']
    m = it['mask_arr']
    w, h = im.size
    s = 900 / max(w, h)
    sz = (round(w * s), round(h * s))
    out[f'{tag}|image'] = np.asarray(im.resize(sz, Image.BILINEAR))
    out[f'{tag}|mask'] = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(sz, Image.NEAREST)) > 0
    cov = C.coverage(h, w, it['A'])
    out[f'{tag}|unseen'] = np.asarray(Image.fromarray(((m & (cov == 0)) * 255).astype(np.uint8)).resize(sz, Image.NEAREST)) > 0
    out[f'{tag}|released'] = np.array(it['A'])
    out[f'{tag}|grid'] = np.array(C.grid_layout(h, w))
    meta[tag] = {'id': it['id'], 'group': it['group'], 'size': [w, h], 'scale': s, 'unseen_px': unseen,
                 'defect_px': int(m.sum())}
    print(tag, meta[tag], flush=True)
dst = R.HERE / 'results_p5' / 'qualitative'
dst.mkdir(parents=True, exist_ok=True)
np.savez_compressed(dst / 'coverage_examples.npz', **out)
(dst / 'coverage_examples.json').write_text(json.dumps(meta, indent=1))
