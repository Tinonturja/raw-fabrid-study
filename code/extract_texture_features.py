#!/usr/bin/env python3
"""Calculate fixed classical grayscale texture descriptors for all patches."""
import csv,io,json,zipfile,time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'results'/'texture_features';out.mkdir(parents=True,exist_ok=True)
with (ROOT/'data'/'remote_metadata'/'RAW_FABRID_Patches_Metadata.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
if json.loads((ROOT/'reports'/'archive_audit.json').read_text())['errors']: raise SystemExit('Archive audit failed')
freq=np.fft.fftfreq(64); fy,fx=np.meshgrid(freq,freq,indexing='ij'); radius=np.sqrt(fx**2+fy**2)
bands=[(0,.03),(.03,.06),(.06,.12),(.12,.2),(.2,.3),(.3,.46),(.46,.72)]
angles=np.arctan2(fy,fx)%np.pi
angle_masks=[(angles>=k*np.pi/4)&(angles<(k+1)*np.pi/4)&(radius>0) for k in range(4)]

def descriptor(im):
    x=np.asarray(im.convert('L').resize((64,64),Image.Resampling.BILINEAR),dtype=np.float32)/255
    gx=np.diff(x,axis=1); gy=np.diff(x,axis=0)
    fft=np.abs(np.fft.fft2(x-x.mean()))**2; fft[0,0]=0; total=float(fft.sum())+1e-12
    power=[float(fft[(radius>=lo)&(radius<hi)].sum()/total) for lo,hi in bands]
    power += [float(fft[m].sum()/total) for m in angle_masks]
    hist=np.histogram(x,bins=16,range=(0,1))[0].astype(float)/x.size
    return np.asarray([x.mean(),x.std(),*np.quantile(x,[.05,.25,.5,.75,.95]),np.mean(abs(gx)),np.mean(abs(gy)),gx.std(),gy.std(),*power,*hist],dtype=np.float32)

features=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype='float32',shape=(len(rows),38))
valid=np.lib.format.open_memmap(out/'valid.npy',mode='w+',dtype='bool',shape=(len(rows),));valid[:]=False
folder={'train_good':'train/good','test_good':'test/good','test_defect':'test/defect'};t=time.time()
with zipfile.ZipFile(ROOT/'data'/'RAW_FABRID.zip') as z:
    for i,row in enumerate(rows):
        n=f"RAW_FABRID/MVTec/{folder[row['type']]}/{row['patch_name']}"
        with Image.open(io.BytesIO(z.read(n))) as im: features[i]=descriptor(im)
        valid[i]=True
        if (i+1)%1000==0: print('Texture features',i+1,'/',len(rows),'seconds',round(time.time()-t,1),flush=True)
features.flush();valid.flush();(out/'feature_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'feature_spec.json').write_text(json.dumps({'model':'classical fixed grayscale texture descriptors','resize':[64,64],
 'features':'mean, standard deviation, intensity quantiles and histogram, gradient moments, radial and directional FFT energy',
 'dimensions':38,'fine_tuning':False,'row_count':len(rows),'completed_rows':int(valid.sum()),'device':'cpu'},indent=2)+'\n')
print('Completed',len(rows),'patches',flush=True)
