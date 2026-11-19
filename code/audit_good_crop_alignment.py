#!/usr/bin/env python3
"""Check whether good test patches match source crops and contain marked pixels."""
import io, json, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
meta=ROOT/'data'/'remote_metadata'
items=json.loads((meta/'RAW_FABRID_Patches_COCO.json').read_text())['images']
groups={}
for im in items:
    if im['type']=='test_good': groups.setdefault(im['parent'],[]).append(im)

def get(z,n):
    with Image.open(io.BytesIO(z.read(n))) as im:
        im.load(); return np.asarray(im).copy()

results=[]; errors=[]; all_checked=0
archive_audit=json.loads((ROOT/'reports'/'archive_audit.json').read_text())
if archive_audit['errors']: raise SystemExit('Archive audit failed')
with zipfile.ZipFile(ROOT/'data'/'RAW_FABRID.zip') as z:
    for parent,ims in sorted(groups.items()):
        mask=get(z,'RAW_FABRID/masks/'+parent)
        candidates=[]
        for im in ims:
            r,c=im['r_offset'],im['c_offset']; crop=mask[r:r+256,c:c+256]
            if np.any(crop>0): candidates.append((im,crop))
        if not candidates: continue
        try: original=get(z,'RAW_FABRID/images/'+parent)
        except Exception as e:
            errors.append({'parent':parent,'error':f'{type(e).__name__}: {e}'}); continue
        for im,crop in candidates:
            patch=get(z,'RAW_FABRID/MVTec/test/good/'+im['file_name'])
            r,c=im['r_offset'],im['c_offset']; original_crop=original[r:r+256,c:c+256]
            channels_equal=bool(original_crop.ndim==3 and original_crop.shape[2]==3 and np.array_equal(original_crop[:,:,0],original_crop[:,:,1]) and np.array_equal(original_crop[:,:,1],original_crop[:,:,2]))
            normalized_crop=original_crop[:,:,0] if channels_equal else original_crop
            results.append({'patch':im['file_name'],'parent':parent,'source_rgb_channels_equal':channels_equal,
                            'pixel_identical_to_source_crop':bool(np.array_equal(patch,normalized_crop)),
                            'source_mask_nonzero_pixels':int(np.count_nonzero(crop)),
                            'source_mask_ge128_pixels':int(np.count_nonzero(crop>=128)),
                            'source_mask_255_pixels':int(np.count_nonzero(crop==255))})
            all_checked+=1
out={'scope':'Publisher SHA-256 and individual ZIP-member CRC checked.',
     'count_checked':all_checked,'counts':dict(Counter((x['pixel_identical_to_source_crop'],x['source_mask_ge128_pixels']>0) for x in [])),
     'patches':results,'unreadable_parents':errors}
out.pop('counts')
path=ROOT/'reports'/'good_crop_alignment_preliminary.json'; path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
