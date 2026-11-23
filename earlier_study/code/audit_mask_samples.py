#!/usr/bin/env python3
"""Check source and patch masks directly from ZIP parts, with ZIP CRC validation."""
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
meta=ROOT/'data'/'remote_metadata'
rows=list(csv.DictReader((meta/'RAW_FABRID_Patches_Metadata.csv').open(newline='',encoding='utf-8-sig')))
problem=set(json.loads((ROOT/'reports'/'coco_audit_preliminary.json').read_text())['examples']['defect_patch_without_coco_annotation'])

def imageinfo(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image.load(); histogram=image.histogram()
        return {'mode':image.mode,'size':image.size,'values':[i for i,c in enumerate(histogram) if c],
                'nonzero_pixels':sum(histogram[1:]) if image.mode=='L' else None}

archive_audit=json.loads((ROOT/'reports'/'archive_audit.json').read_text())
if archive_audit['errors']: raise SystemExit('Archive audit failed')
with zipfile.ZipFile(ROOT/'data'/'RAW_FABRID.zip') as z:
    masks=[n for n in z.namelist() if n.startswith('RAW_FABRID/masks/') and n.lower().endswith('.png')]
    summary=Counter(); errors=[]; orphan=[]
    images={Path(n).name for n in z.namelist() if n.startswith('RAW_FABRID/images/') and n.lower().endswith('.png')}
    for n in masks:
        try:
            v=imageinfo(z.read(n)); summary[(v['mode'],tuple(v['size']),tuple(v['values']))]+=1
            if v['nonzero_pixels']==0: errors.append({'file':n,'problem':'empty mask'})
            if Path(n).name not in images: orphan.append(Path(n).name)
        except Exception as e: errors.append({'file':n,'error':f'{type(e).__name__}: {e}'})
    samples=[]
    for row in rows:
        if row['patch_name'] not in problem: continue
        name=row['patch_name']; member='RAW_FABRID/MVTec/ground_truth/defect/'+name
        try:
            v=imageinfo(z.read(member)); samples.append({'name':name,'csv_area':int(row['defect_area']),'decoded_mask':v})
        except Exception as e: errors.append({'file':member,'error':f'{type(e).__name__}: {e}'})
out={'scope':'Publisher SHA-256 and ZIP-member CRC checked.',
     'source_masks_checked':len(masks),'mask_shapes_modes_values':{str(k):v for k,v in summary.items()},
     'orphan_masks':orphan,'small_defect_patch_masks':samples,'errors':errors}
path=ROOT/'reports'/'mask_sample_audit_preliminary.json'; path.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
