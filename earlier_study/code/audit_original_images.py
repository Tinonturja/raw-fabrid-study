#!/usr/bin/env python3
"""Check every original image's decoding, shape and grayscale channel consistency."""
import hashlib,io,json,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
expected=json.loads((ROOT/'reports'/'archive_audit.json').read_text())
if expected['errors']:
    raise SystemExit('Complete archive audit has not passed; refusing to audit original images')
counts=Counter(); errors=[]; bad_channels=[]; read=0; by_pixel_hash=defaultdict(list)
with zipfile.ZipFile(ROOT/'data'/'RAW_FABRID.zip') as z:
    files=[n for n in z.namelist() if n.startswith('RAW_FABRID/images/') and n.lower().endswith('.png')]
    for n in files:
        try:
            with Image.open(io.BytesIO(z.read(n))) as image:
                image.load(); arr=np.asarray(image); mode=image.mode; size=image.size
                counts[(mode,str(size))]+=1
                if size!=(1792,1024): errors.append({'file':n,'error':'unexpected image dimensions','size':size})
                if mode=='RGB':
                    if not (np.array_equal(arr[:,:,0],arr[:,:,1]) and np.array_equal(arr[:,:,1],arr[:,:,2])): bad_channels.append(n)
                    else: by_pixel_hash[hashlib.sha256(arr[:,:,0].tobytes()).hexdigest()].append(n)
                elif mode=='L': by_pixel_hash[hashlib.sha256(arr.tobytes()).hexdigest()].append(n)
                else: errors.append({'file':n,'error':f'unexpected mode {mode}'})
                read+=1
        except Exception as e: errors.append({'file':n,'error':f'{type(e).__name__}: {e}'})
        if read and read%100==0: print(f'Checked {read}/{len(files)} original images',flush=True)
out={'scope':'Complete archive passed publisher SHA-256 and ZIP CRC before this check.','original_image_count':len(files),
     'decoded':read,'modes_and_dimensions':{str(k):v for k,v in counts.items()},
     'rgb_images_with_different_channels':bad_channels,'errors':errors,
     'exact_duplicate_pixel_groups':[names for names in by_pixel_hash.values() if len(names)>1]}
(ROOT/'reports'/'original_image_audit.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
