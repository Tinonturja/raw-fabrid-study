#!/usr/bin/env python3
"""Extract fixed pretrained features for each released RAW-FABRID patch.

This is an exploratory, image-level baseline, not PatchCore or a segmentation model.
All features are computed independently of labels; no dataset-specific fine-tuning occurs.
ZIP members are checked by zipfile's CRC while read.
"""
import argparse, csv, io, json, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--device',choices=('mps','cpu'),default='mps')
    args=ap.parse_args()
    if args.device=='mps' and not torch.backends.mps.is_available(): raise SystemExit('MPS not available')
    torch.set_num_threads(4)
    md=ROOT/'data'/'remote_metadata'/'RAW_FABRID_Patches_Metadata.csv'
    with md.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    folder={'train_good':'train/good','test_good':'test/good','test_defect':'test/defect'}
    out=ROOT/'results'/'features'; out.mkdir(parents=True,exist_ok=True)
    feature_path=out/'features.npy'; valid_path=out/'valid.npy'
    if feature_path.exists() and valid_path.exists():
        features=np.load(feature_path,mmap_mode='r+'); valid=np.load(valid_path,mmap_mode='r+')
        if features.shape!=(len(rows),512) or valid.shape!=(len(rows),): raise SystemExit('Existing feature cache shape mismatch')
    else:
        features=np.lib.format.open_memmap(feature_path,mode='w+',dtype='float32',shape=(len(rows),512))
        valid=np.lib.format.open_memmap(valid_path,mode='w+',dtype='bool',shape=(len(rows),)); valid[:]=False; valid.flush()
    (out/'feature_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    weights=ResNet18_Weights.IMAGENET1K_V1
    model=resnet18(weights=weights); model.fc=torch.nn.Identity(); model.eval().to(args.device)
    transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    pending=np.flatnonzero(~valid); started=time.time(); done=0
    with zipfile.ZipFile(ROOT/'data'/'RAW_FABRID.zip') as z, torch.inference_mode():
        for begin in range(0,len(pending),args.batch_size):
            ix=pending[begin:begin+args.batch_size]; batch=[]
            for i in ix:
                row=rows[int(i)]; member=f"RAW_FABRID/MVTec/{folder[row['type']]}/{row['patch_name']}"
                with Image.open(io.BytesIO(z.read(member))) as im: batch.append(transform(im.convert('RGB')))
            x=torch.stack(batch).to(args.device); y=model(x).cpu().numpy().astype('float32')
            features[ix]=y; features.flush(); valid[ix]=True; valid.flush(); done+=len(ix)
            if done%256 < args.batch_size or done==len(pending):
                print(f'Feature extraction: {int(valid.sum())}/{len(rows)} patches, {time.time()-started:.1f}s in this run',flush=True)
    info={'model':'torchvision ResNet18 ImageNet1K V1 pretrained','resize':[128,128],
          'input':'released patches converted to RGB','normalization':'ImageNet mean and standard deviation',
          'features':'512-dimensional global average pooled, pre-classifier output','fine_tuning':False,
          'row_count':len(rows),'completed_rows':int(valid.sum()),'device':args.device,
          'note':'Archive member CRCs are checked while reading. Verify the publisher SHA-256 separately.'}
    (out/'feature_spec.json').write_text(json.dumps(info,indent=2)+'\n'); print(json.dumps(info,indent=2))

if __name__=='__main__': main()
