#!/usr/bin/env python3
"""Recompute test-source features on a fixed 4x7 grid, without defect masks."""
import io,json,re,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18,ResNet18_Weights
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]; result=ROOT/'results'
rows=json.loads((result/'features'/'feature_rows.json').read_text())
groups=defaultdict(list)
for idx,r in enumerate(rows):
    if r['type']!='train_good':
        m=re.search(r'_(\d+)_(\d+)\.png$',r['patch_name'])
        if not m: raise ValueError(r['patch_name'])
        rr,cc=map(int,m.groups()); groups[r['parent_raw']].append((idx,rr,cc))
if any(len(v)!=28 or len({(r,c) for _,r,c in v})!=28 for v in groups.values()): raise RuntimeError('Unexpected source grid structure')
if json.loads((ROOT/'reports'/'archive_audit.json').read_text())['errors']: raise SystemExit('Archive audit failed')

freq=np.fft.fftfreq(64); fy,fx=np.meshgrid(freq,freq,indexing='ij'); radius=np.sqrt(fx**2+fy**2)
bands=[(0,.03),(.03,.06),(.06,.12),(.12,.2),(.2,.3),(.3,.46),(.46,.72)]
angles=np.arctan2(fy,fx)%np.pi; angle_masks=[(angles>=k*np.pi/4)&(angles<(k+1)*np.pi/4)&(radius>0) for k in range(4)]
def texture(im):
    x=np.asarray(im.convert('L').resize((64,64),Image.Resampling.BILINEAR),dtype=np.float32)/255
    gx=np.diff(x,axis=1);gy=np.diff(x,axis=0);fft=np.abs(np.fft.fft2(x-x.mean()))**2;fft[0,0]=0;total=float(fft.sum())+1e-12
    power=[float(fft[(radius>=lo)&(radius<hi)].sum()/total) for lo,hi in bands]+[float(fft[m].sum()/total) for m in angle_masks]
    hist=np.histogram(x,bins=16,range=(0,1))[0].astype(float)/x.size
    return np.asarray([x.mean(),x.std(),*np.quantile(x,[.05,.25,.5,.75,.95]),np.mean(abs(gx)),np.mean(abs(gy)),gx.std(),gy.std(),*power,*hist],dtype=np.float32)

targets={}
for old,new in [('features','features_fixed_grid'),('texture_features','texture_features_fixed_grid')]:
    directory=result/new;directory.mkdir(parents=True,exist_ok=True)
    arr=np.load(result/old/'features.npy'); np.save(directory/'features.npy',arr)
    dest=np.load(directory/'features.npy',mmap_mode='r+'); np.save(directory/'valid.npy',np.zeros(len(rows),dtype=bool))
    valid=np.load(directory/'valid.npy',mmap_mode='r+'); valid[[i for i,r in enumerate(rows) if r['type']=='train_good']]=True
    (directory/'feature_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    spec=json.loads((result/old/'feature_spec.json').read_text()); spec['test_source_crops']='fixed 4x7 grid from original source images; defect masks never used for cropping';spec['warning']='Original patch labels are invalid for reconstructed crops; evaluate only source-image labels.'
    (directory/'feature_spec.json').write_text(json.dumps(spec,indent=2)+'\n');targets[old]=(dest,valid)

device='mps' if torch.backends.mps.is_available() else 'cpu'; torch.set_num_threads(4)
model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);model.fc=torch.nn.Identity();model.eval().to(device)
transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
with zipfile.ZipFile(ROOT/'data'/'RAW_FABRID.zip') as z,torch.inference_mode():
    for j,(parent,items) in enumerate(sorted(groups.items()),1):
        with Image.open(io.BytesIO(z.read('RAW_FABRID/images/'+parent))) as original:
            im=original.convert('RGB'); crops=[im.crop((c*256,r*256,(c+1)*256,(r+1)*256)) for _,r,c in items]
        tensors=[transform(im) for im in crops]; vectors=[]
        for k in range(0,len(tensors),16): vectors.extend(model(torch.stack(tensors[k:k+16]).to(device)).cpu().numpy())
        for (idx,_,_),vec,crop in zip(items,vectors,crops):
            targets['features'][0][idx]=vec;targets['texture_features'][0][idx]=texture(crop)
            targets['features'][1][idx]=True;targets['texture_features'][1][idx]=True
        if j%25==0: print('Fixed grid sources',j,'/',len(groups),flush=True)
for data,valid in targets.values():
    data.flush();valid.flush()
    if not valid.all(): raise RuntimeError('Incomplete grid feature extraction')
print('Completed',len(groups),'defect-containing sources; fixed grid selected without masks',flush=True)
