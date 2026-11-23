"""Replay held-out detectors and measure component overlap below 5% pixel FPR."""
import argparse
import hashlib
import json
import time
import zipfile
from collections import defaultdict

import numpy as np
import torch
from scipy.ndimage import label
from torchvision.models import resnet18, ResNet18_Weights, swin_t, Swin_T_Weights

from audit_zip import PUBLISHER_SHA256, sha256
from unseen_roll import (BATCH, CHANNELS, DATA, FIXED, OUT, RESULTS, ROOT, SEED,
                         csv_rows, fit_bank, fit_gaussian, read_image, read_json,
                         tile_scores, transformer_features, transformer_tiles)

FPR_LIMIT = .05
PROTOCOL = ROOT/'reports/region_analysis_protocol.md'
DEST = RESULTS/'region_followup'
POLICIES = ('released', 'fixed')


def extend_add(current, addition):
    if len(current) < len(addition):
        current = np.pad(current, (0, len(addition)-len(current)))
    current[:len(addition)] += addition
    return current


def field_bins(coords, values, positions, shape):
    index = {pos:i for i,pos in enumerate(coords)}
    field = np.zeros(shape, dtype=np.int32)
    for y,x in positions:
        cells = 1 + np.floor(values[index[(y,x)]]*1000).astype(np.int32)
        expanded = cells.repeat(16, 0).repeat(16, 1)
        np.maximum(field[y:y+256, x:x+256], expanded, out=field[y:y+256, x:x+256])
    return field


def integrate(background, region_mass, region_count):
    fpr = np.r_[0., np.cumsum(background[::-1], dtype=np.float64)/sum(background)]
    pro = np.r_[0., np.cumsum(region_mass[::-1], dtype=np.float64)/region_count]
    i = np.searchsorted(fpr, FPR_LIMIT, side='right')
    if i < len(fpr):
        y = np.interp(FPR_LIMIT, fpr[i-1:i+1], pro[i-1:i+1])
        fpr, pro = np.r_[fpr[:i],FPR_LIMIT], np.r_[pro[:i],y]
    area = np.trapezoid(pro, fpr)/FPR_LIMIT
    return float(area), fpr.tolist(), pro.tolist()


@torch.inference_mode()
def run(model_name, roll, archive, meta, patches, coords, orig, device):
    start = time.perf_counter()
    fold = read_json(RESULTS/'baseline_seed_13017.json')['folds']['held_out_'+roll]
    refs = [patches[i] for i in fold['train_patch_indices']]
    assert len(refs)==1000 and all(r['type']=='train_good'
        and meta[r['parent_raw']]['roll_id']!=roll
        and meta[r['parent_raw']]['has_defect'].lower()=='false' for r in refs)
    if model_name == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
        bank, selected = fit_bank(archive, refs, model, device, SEED)
        fit_hash = hashlib.sha256(bank.cpu().numpy().tobytes()).hexdigest()
        def score(image, positions):
            return tile_scores(image, positions, model, bank, device)
    else:
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).eval().to(device)
        channels = torch.tensor(CHANNELS, device=device)
        chunks = []
        for start_i in range(0, len(refs), BATCH):
            tiles = [read_image(archive, 'MVTec/train/good/'+r['patch_name'],'RGB')
                     for r in refs[start_i:start_i+BATCH]]
            chunks.append(transformer_features(tiles,model,channels,device).cpu().numpy())
        mean, inv = fit_gaussian(np.concatenate(chunks))
        fit_hash = hashlib.sha256(mean.tobytes()).hexdigest()
        params = tuple(torch.from_numpy(a).to(device) for a in (mean,inv))
        def score(image,positions):
            return transformer_tiles(image,positions,model,channels,params,device)
    saved = read_json(OUT/f'{model_name}_{roll}.json')
    assert fit_hash == saved['fit']['bank_sha256' if model_name=='resnet18' else 'mean_sha256']
    positives = [p for p,r in saved['frames'].items() if r['defect']]
    chosen = set()
    if roll == 'Rollo3A':
        chosen = {min(positives,key=lambda p:(saved['frames'][p]['unseen_foreground'],p)),
                  min(positives,key=lambda p:(-saved['frames'][p]['unseen_foreground'],p))}
        assert len(chosen)==2
    data = {p:{'background':np.zeros(1,dtype=np.int64),
               'region_mass':np.zeros(1,dtype=np.float64),
               'foreground':np.zeros(1,dtype=np.int64)} for p in POLICIES}
    region_count = 0
    primary = set(orig['defective_frames']) | set(orig['evaluation_clean'])
    for parent in sorted(set(fold['test_parents']) & primary):
        image = read_image(archive,'images/'+parent,'RGB')
        positive = parent in orig['defective_frames']
        mask = (np.asarray(read_image(archive,'masks/'+parent,'L'))>=128 if positive
                else np.zeros((1024,1792),dtype=bool))
        components, n = label(mask,structure=np.ones((3,3)))
        region_count += n
        sizes = np.bincount(components[mask], minlength=n+1)
        policies = {'released':coords[parent], 'fixed':FIXED}
        positions = list(dict.fromkeys(coords[parent]+FIXED))
        values = score(image,positions)
        maps = {}
        for policy, locations in policies.items():
            bins = field_bins(positions,values,locations,mask.shape)
            expected = saved['frames'][parent]['scores'][policy]
            used = [positions.index(p) for p in locations]
            assert abs(float(values[used].max())-expected) < 1e-4
            bg = np.bincount(bins[~mask],minlength=1)
            fg = np.bincount(bins[mask],minlength=1)
            mass = np.bincount(bins[mask],weights=1./sizes[components[mask]],minlength=1)
            for key, array in [('background',bg),('foreground',fg),('region_mass',mass)]:
                data[policy][key] = extend_add(data[policy][key],array)
            if parent in chosen:
                maps[policy] = bins
        if parent in chosen:
            np.savez_compressed(DEST/f'{model_name}_{parent}.npz',
                                image=np.asarray(image.convert('L')), mask=mask, **maps)
    out = {'model':model_name,'roll':roll,'protocol_sha256':sha256(PROTOCOL),
           'fit_sha256':fit_hash,'regions':region_count,'device':device,'examples':sorted(chosen),
           'total_seconds':time.perf_counter()-start,'policies':{}}
    for p, hist in data.items():
        assert abs(sum(hist['region_mass'])-region_count)<1e-6
        previous=saved['histograms']['primary_pixels'][p]
        indices=np.array(previous['bins'])
        width=max(len(v) for v in hist.values())
        hist={k:np.pad(v,(0,width-len(v))) for k,v in hist.items()}
        total=hist['background']+hist['foreground']
        assert np.array_equal(total[indices],previous['total'])
        assert np.array_equal(hist['foreground'][indices],previous['foreground'])
        assert int(total.sum())==sum(previous['total'])
        auc,x,y=integrate(hist['background'],hist['region_mass'],region_count)
        out['policies'][p]={'aupro_005':auc,'background':hist['background'].tolist(),
                            'region_mass':hist['region_mass'].tolist(),'fpr':x,'pro':y}
    target=DEST/f'{model_name}_{roll}.json'
    target.write_text(json.dumps(out,separators=(',',':'),allow_nan=False)+'\n')
    print(model_name, roll, 'regions',region_count,'seconds',round(out['total_seconds'],1),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--roll')
    parser.add_argument('--model',choices=('resnet18','swin_t'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    archive_path=DATA/'RAW_FABRID.zip'
    assert sha256(archive_path)==PUBLISHER_SHA256
    DEST.mkdir(exist_ok=True)
    orig=read_json(RESULTS/'localization_study.json')
    with zipfile.ZipFile(archive_path) as archive:
        meta={r['filename']:r for r in csv_rows(archive,'HighRes')}
        patches=csv_rows(archive,'Patches')
        coords=defaultdict(list)
        for r in json.loads(archive.read('RAW_FABRID/RAW_FABRID_Patches_COCO.json'))['images']:
            coords[r['parent']].append((r['r_offset'],r['c_offset']))
        rolls=sorted({r['roll_id'] for r in meta.values()})
        if args.roll and args.roll not in rolls:
            parser.error('Unknown roll')
        for model in ('resnet18','swin_t'):
            if args.model and args.model != model:
                continue
            for roll in rolls:
                if args.roll and roll != args.roll:
                    continue
                run(model,roll,archive,meta,patches,coords,orig,device)


if __name__=='__main__':
    main()
