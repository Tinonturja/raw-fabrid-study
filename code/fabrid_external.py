"""Evaluate fixed RAW-FABRID detectors on the independent FabriD v3 test split."""
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score
from torchvision.models import ResNet18_Weights, Swin_T_Weights, resnet18, swin_t

from audit_zip import PUBLISHER_SHA256, sha256
from localization_study import pixel_ap
from region_followup import extend_add, field_bins, integrate
from unseen_roll import (BATCH, CHANNELS, DATA, FIXED, OUT, RESULTS, ROOT, SEED, csv_rows,
                         fit_bank, fit_gaussian, read_image, read_json, tile_scores,
                         transformer_features, transformer_tiles)

SOURCE = DATA/'fabrid_v3.zip'
DIGEST = '4944cec7f54ce3f9120ed243ea8c33f3ff3a3d4c39556fd06e7cfe3a5b3a05c3'
OUTPUT = RESULTS/'fabrid_external.json'
PROTOCOL = ROOT/'reports/fabrid_external_protocol.md'


def positions(length):
    result = list(range(0, length-255, 256))
    if result[-1] != length-256:
        result.append(length-256)
    return result


def source_id(name):
    return Path(name).name.split('.rf.')[0]


def polygon_mask(archive, path, size):
    contents = archive.read(path).decode('utf-8-sig')
    bitmap = Image.new('1', size)
    draw = ImageDraw.Draw(bitmap)
    lines = [line for line in contents.splitlines() if line.strip()]
    for line in lines:
        vals = line.split()
        if len(vals)<7 or (len(vals)-1)%2:
            raise ValueError(f'Malformed polygon: {path}')
        category = int(vals[0])
        points = np.array([float(v) for v in vals[1:]]).reshape(-1,2)
        if category not in range(4) or not np.isfinite(points).all() or (points<0).any() or (points>1).any():
            raise ValueError(f'Invalid polygon: {path}')
        draw.polygon([(min(size[0]-1,int(x*size[0])),min(size[1]-1,int(y*size[1])))
                      for x,y in points],fill=1)
    mask = np.asarray(bitmap,dtype=bool)
    if lines and not mask.any():
        raise ValueError(f'Empty rasterized positive mask: {path}')
    return mask


def guided_windows(mask, fixed):
    h,w=mask.shape
    guided=[]
    for y,x in fixed:
        yy,xx=np.nonzero(mask[y:y+256,x:x+256])
        if len(yy):
            yc = y + (int(yy.min())+int(yy.max()))//2
            xc = x + (int(xx.min())+int(xx.max()))//2
            guided.append((min(max(yc-128,0),h-256),min(max(xc-128,0),w-256)))
        else:
            guided.append((y,x))
    return guided


def fit_models(archive, patches, meta, device):
    fold=read_json(RESULTS/'baseline_seed_13017.json')['folds']['held_out_Rollo3A']
    refs=[patches[i] for i in fold['train_patch_indices']]
    assert len(refs)==1000 and all(r['type']=='train_good'
        and meta[r['parent_raw']]['roll_id']!='Rollo3A'
        and meta[r['parent_raw']]['has_defect'].lower()=='false' for r in refs)
    cnn=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    bank,_=fit_bank(archive,refs,cnn,device,SEED)
    cnn_hash=hashlib.sha256(bank.cpu().numpy().tobytes()).hexdigest()
    assert cnn_hash==read_json(OUT/'resnet18_Rollo3A.json')['fit']['bank_sha256']
    swin=swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).eval().to(device)
    channels=torch.tensor(CHANNELS,device=device)
    chunks=[]
    for i in range(0,len(refs),BATCH):
        images=[read_image(archive,'MVTec/train/good/'+r['patch_name'],'RGB')
                for r in refs[i:i+BATCH]]
        chunks.append(transformer_features(images,swin,channels,device).cpu().numpy())
    mean,inv=fit_gaussian(np.concatenate(chunks))
    mean_hash=hashlib.sha256(mean.tobytes()).hexdigest()
    assert mean_hash==read_json(OUT/'swin_t_Rollo3A.json')['fit']['mean_sha256']
    params=tuple(torch.from_numpy(a).to(device) for a in (mean,inv))
    return {'resnet18':lambda image,pos:tile_scores(image,pos,cnn,bank,device),
            'swin_t':lambda image,pos:transformer_tiles(image,pos,swin,channels,params,device)}, {
            'resnet_bank_sha256':cnn_hash,'swin_mean_sha256':mean_hash}


def pack(hist):
    width=max(len(a) for a in hist.values())
    hist={k:np.pad(a,(0,width-len(a))) for k,a in hist.items()}
    total=hist['background']+hist['foreground']
    indices=np.flatnonzero(total)
    return {'bins':indices.tolist(),'total':total[indices].tolist(),
            'foreground':hist['foreground'][indices].tolist(),
            'region_mass':hist['region_mass'][indices].tolist()}


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    assert sha256(SOURCE)==DIGEST and sha256(DATA/'RAW_FABRID.zip')==PUBLISHER_SHA256
    with zipfile.ZipFile(DATA/'RAW_FABRID.zip') as raw:
        patches=csv_rows(raw,'Patches')
        meta={r['filename']:r for r in csv_rows(raw,'HighRes')}
        models,fit=fit_models(raw,patches,meta,device)
    output={'protocol_sha256':sha256(PROTOCOL),'archive_sha256':DIGEST,'device':device,
            'fit':fit,'frames':{},'models':{},'exclusions':Counter()}
    hist={m:{p:{k:np.zeros(1,dtype=np.float64 if k=='region_mass' else np.int64)
                 for k in ('background','foreground','region_mass')} for p in ('fixed','guided')}
          for m in models}
    regions=0
    with zipfile.ZipFile(SOURCE) as archive:
        names=archive.namelist()
        test_images=sorted(n for n in names if n.startswith('test/images/') and n.lower().endswith(('.jpg','.jpeg','.png')))
        test_labels=[n for n in names if n.startswith('test/labels/') and n.endswith('.txt')]
        assert len(test_images)==len(set(test_images)) and len(test_labels)==len(set(test_labels))
        group_splits=defaultdict(set)
        for n in names:
            parts=n.split('/')
            if len(parts)==3 and parts[0] in ('train','valid','test') and parts[1]=='images':
                group_splits[source_id(n)].add(parts[0])
        output['source_ids_shared_between_splits']=sum(len(v)>1 for v in group_splits.values())
        seen_sources=set()
        seen_images={}
        for path in test_images:
            sid=source_id(path)
            if sid in seen_sources:
                output['exclusions']['duplicate_test_source_id']+=1
                continue
            seen_sources.add(sid)
            stem=Path(path).stem
            label_path='test/labels/'+stem+'.txt'
            if label_path not in names:
                raise ValueError('Missing test label: '+label_path)
            with Image.open(io.BytesIO(archive.read(path))) as im:
                im.load()
                if im.width<256 or im.height<256:
                    output['exclusions']['smaller_than_window']+=1
                    continue
                image=im.convert('L').convert('RGB')
            image_hash=hashlib.sha256(image.tobytes()).hexdigest()
            mask=polygon_mask(archive,label_path,image.size)
            label_hash=hashlib.sha256(mask.tobytes()).hexdigest()
            if image_hash in seen_images:
                if seen_images[image_hash]!=label_hash:
                    raise ValueError('Identical image has conflicting masks: '+path)
                output['exclusions']['exact_duplicate_image']+=1
                continue
            seen_images[image_hash]=label_hash
            h,w=mask.shape
            fixed=[(y,x) for y in positions(h) for x in positions(w)]
            guided=guided_windows(mask,fixed)
            allpos=list(dict.fromkeys(fixed+guided))
            components,n=label(mask,structure=np.ones((3,3)))
            regions+=n
            sizes=np.bincount(components[mask],minlength=n+1)
            rec={'source_id':sid,'image_sha256':image_hash,'shape':[h,w],
                 'defect':bool(mask.any()),'foreground':int(mask.sum()),
                 'windows':len(fixed),'scores':{},'unseen_foreground':{}}
            for name,score in models.items():
                values=score(image,allpos)
                rec['scores'][name]={}
                for policy,loc in [('fixed',fixed),('guided',guided)]:
                    bins=field_bins(allpos,values,loc,mask.shape)
                    p=hist[name][policy]
                    for key,array in [('background',np.bincount(bins[~mask],minlength=1)),
                                      ('foreground',np.bincount(bins[mask],minlength=1)),
                                      ('region_mass',np.bincount(bins[mask],weights=1./sizes[components[mask]],minlength=1))]:
                        p[key]=extend_add(p[key],array)
                    scoremax=max(float(values[allpos.index(x)].max()) for x in loc)
                    rec['scores'][name][policy]=scoremax
                    unseen=int(((bins==0)&mask).sum())
                    if policy in rec['unseen_foreground']:
                        assert rec['unseen_foreground'][policy]==unseen
                    rec['unseen_foreground'][policy]=unseen
            output['frames'][path]=rec
            if len(output['frames'])%50==0:
                print('FabriD scored',len(output['frames']),flush=True)
    labels=[v['defect'] for v in output['frames'].values()]
    assert sum(labels)>0 and sum(not x for x in labels)>0 and regions>0
    output['regions']=regions
    for model,policies in hist.items():
        output['models'][model]={}
        for policy,v in policies.items():
            h=pack(v)
            bins=np.array(h['bins']);width=bins.max()+1
            total=np.zeros(width,dtype=np.int64);fg=np.zeros(width,dtype=np.int64);mass=np.zeros(width)
            total[bins]=h['total'];fg[bins]=h['foreground'];mass[bins]=h['region_mass']
            ap_all=pixel_ap(total,fg);ap_observed=pixel_ap(total[1:],fg[1:])
            au_pro,_,_=integrate(total-fg,mass,regions)
            auc=roc_auc_score(labels,[r['scores'][model][policy] for r in output['frames'].values()])
            output['models'][model][policy]={'whole_ap':ap_all,'observed_ap':ap_observed,'aupro_005':au_pro,
                'frame_auroc':float(auc),'unseen_foreground':int(fg[0]),'histogram':h}
    OUTPUT.write_text(json.dumps(output,separators=(',',':'),allow_nan=False)+'\n')
    print('FabriD completed',len(labels),'frames',sum(labels),'defective',regions,'regions',flush=True)


if __name__=='__main__':
    main()
