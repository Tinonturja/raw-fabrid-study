#!/usr/bin/env python3
"""Compare released patch masks and coordinates with source-image masks."""
import csv, io, json, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
meta=ROOT/'data'/'remote_metadata'
with (meta/'RAW_FABRID_Patches_Metadata.csv').open(newline='',encoding='utf-8-sig') as f: rows={r['patch_name']:r for r in csv.DictReader(f)}
coco=json.loads((meta/'RAW_FABRID_Patches_COCO.json').read_text())
groups=defaultdict(list)
for im in coco['images']:
    if im['type'] in ('test_good','test_defect'): groups[im['parent']].append(im)

def image(z, path):
    with Image.open(io.BytesIO(z.read(path))) as im:
        im.load(); return np.asarray(im).copy()

counts=Counter(); examples=defaultdict(list); diffpix=Counter(); mask_hist=np.zeros(256,dtype=np.int64)
def note(k,name):
    counts[k]+=1
    if len(examples[k])<8: examples[k].append(name)

archive_audit=json.loads((ROOT/'reports'/'archive_audit.json').read_text())
if archive_audit['errors']: raise SystemExit('Archive audit failed')
with zipfile.ZipFile(ROOT/'data'/'RAW_FABRID.zip') as z:
    for j,(parent,ims) in enumerate(sorted(groups.items()),1):
        src=image(z,'RAW_FABRID/masks/'+parent)
        mask_hist+=np.bincount(src.ravel(),minlength=256)
        if np.any((src!=0)&(src!=255)): note('source_mask_has_intermediate_values',parent)
        for im in ims:
            name=im['file_name']; r=im['r_offset']; c=im['c_offset']; crop=src[r:r+256,c:c+256]
            if im['type']=='test_good':
                if np.any(crop>0): note('good_crop_source_nonzero',name)
                if np.any(crop>=128): note('good_crop_source_ge128',name)
                continue
            actual=image(z,'RAW_FABRID/MVTec/ground_truth/defect/'+name)
            count=int(np.count_nonzero(actual)); declared=int(rows[name]['defect_area'])
            if actual.shape!=(256,256) or not np.isin(actual,[0,255]).all(): note('patch_mask_nonbinary_or_wrong_size',name)
            if count!=declared: note('patch_mask_pixel_count_differs_csv',name)
            for threshold in [1,128,255]:
                n=int(np.count_nonzero((crop>=threshold)!=(actual>0)))
                if n: note(f'patch_mask_differs_source_ge{threshold}',name); diffpix[f'ge{threshold}']+=n
        if j%50==0: print(f'Checked masks for {j}/{len(groups)} source images',flush=True)
result={'scope':'Publisher SHA-256 and individual ZIP-member CRC checked.',
        'defective_source_images':len(groups),'patch_counts':dict(Counter(im['type'] for ims in groups.values() for im in ims)),
        'counts':dict(counts),'examples':dict(examples),'different_pixels_across_defect_patches':dict(diffpix),
        'source_mask_histogram_nonzero':{str(i):int(v) for i,v in enumerate(mask_hist) if i and v}}
path=ROOT/'reports'/'patch_mask_consistency_preliminary.json'; path.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:v for k,v in result.items() if k!='source_mask_histogram_nonzero'},indent=2))
