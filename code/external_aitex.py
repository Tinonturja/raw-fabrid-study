"""Run the specified within-AITEX replication on the publisher's native strips."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

from check_crop_publication_scope import DONORS, POLICIES
from localization_study import (BATCH, ROOT, TRANSFORM, embeddings,
                                score_parent, select_bank, summarize_policy)
from audit_zip import sha256

SEED = 13017
DATA = ROOT/'data'/'aitex'
NAMES = ('Defect_images','NODefect_images','Mask_images')
FIXED = [(0, x) for x in range(0, 4096, 256)]
OVERLAP = [(0, x) for x in range(0, 3841, 128)]


def prepare_archives(download, temp):
    checks = json.loads((ROOT/'reports/aitex_archives.json').read_text())
    tar = shutil.which('bsdtar')
    if tar is None:
        raise SystemExit('Install bsdtar (libarchive-tools) to read the AITEX 7z files.')
    DATA.mkdir(exist_ok=True, parents=True)
    folders = {}
    for prefix in NAMES:
        name = prefix + '.7z'
        archive = DATA/name
        if not archive.is_file() and download:
            url = 'https://www.aitex.es/wp-content/uploads/2019/07/' + name
            with tempfile.NamedTemporaryFile(dir=DATA, delete=False) as f:
                pending = Path(f.name)
            try:
                with urlopen(Request(url, headers={'User-Agent':'Mozilla/5.0'}),timeout=120) as resp, pending.open('wb') as f:
                    shutil.copyfileobj(resp, f, 1024*1024)
                if sha256(pending) != checks[name]['sha256']:
                    raise ValueError(f'Archive fingerprint differs for {name}')
                pending.replace(archive)
            finally:
                pending.unlink(missing_ok=True)
        if not archive.is_file() or sha256(archive) != checks[name]['sha256']:
            raise ValueError(f'Provide the study-fingerprinted publisher archive {name} or use --download')
        entries = subprocess.check_output([tar, '-tf', str(archive)], text=True).splitlines()
        listing = subprocess.check_output([tar, '-tvf', str(archive)], text=True).splitlines()
        if (len(entries) != len(set(entries)) or len(listing) != len(entries) or
                any(row[0] not in ('-', 'd') for row in listing) or
                any(PurePosixPath(e).is_absolute() or '..' in PurePosixPath(e).parts or '\\' in e for e in entries)):
            raise ValueError(f'Unexpected archive members in {name}')
        folder = temp/prefix
        folder.mkdir()
        subprocess.run([tar, '-xf', str(archive), '-C', str(folder)], check=True)
        folders[prefix] = folder
    return folders


def gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L'))


def audit_sources(folders):
    masks = defaultdict(list)
    issues = []
    for path in sorted(folders['Mask_images'].rglob('*.png')):
        match = re.fullmatch(r'(\d+_\d+_\d+)_mask\d*', path.stem, flags=re.I)
        if match:
            masks[match[1]].append(path)
        else:
            issues.append([path.name, 'unrecognized mask filename'])
    by_hash = defaultdict(list)
    for prefix in NAMES[:2]:
        for path in sorted(folders[prefix].rglob('*.png')):
            match = re.fullmatch(r'(\d+)_(\d+)_(\d+)', path.stem)
            if not match:
                issues.append([path.name,'unrecognized image filename'])
                continue
            image = gray(path)
            if image.shape != (256,4096):
                issues.append([path.name, 'unsupported native dimensions', list(image.shape)])
                continue
            defect = prefix == 'Defect_images'
            files = masks.get(path.stem, []) if defect else []
            arrays = [gray(m) for m in files]
            if defect and (not files or any(a.shape != image.shape for a in arrays)):
                issues.append([path.name, 'missing or misaligned source mask'])
                continue
            fg = np.logical_or.reduce([a >= 128 for a in arrays]) if defect else None
            if defect and not fg.any():
                issues.append([path.name, 'empty defect mask'])
                continue
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            by_hash[digest].append((path, match[3], defect, files))
    samples = {}
    for digest, copies in by_hash.items():
        copies.sort(key=lambda x: x[0].name)
        if len({(fabric,defect) for _,fabric,defect,_ in copies}) != 1:
            issues.append([[c[0].name for c in copies], 'duplicate across fabric or label'])
            continue
        path, fabric, defect, _ = copies[0]
        if path.name in samples:
            raise ValueError('Conflicting image basename ' + path.name)
        samples[path.name] = {'path':path,'fabric':fabric,'defect':defect,'hash':digest,
                              'masks':sorted({m for c in copies for m in c[3]})}
        if len(copies) > 1:
            issues.append([[c[0].name for c in copies], 'identical pixels, retained first filename'])
    return samples, issues


def source_mask(row):
    if row['defect']:
        return np.logical_or.reduce([gray(p) >= 128 for p in row['masks']])
    return np.zeros((256,4096), dtype=bool)


def guided_layout(mask):
    coords = []
    for _, x in FIXED:
        columns = np.flatnonzero(mask[:,x:x+256].any(axis=0))
        left = int(np.clip(x + (columns[0]+columns[-1])//2 - 128, 0, 3840)) if len(columns) else x
        coords.append((0,left))
    return coords


def split_and_assign(samples):
    groups = sorted({r['fabric'] for r in samples.values()})
    rng = np.random.default_rng(SEED)
    permuter = np.random.default_rng(20261119)
    train, cal, test, donors, omitted = [], [], [], {}, []
    for fabric in groups:
        good = sorted(p for p,r in samples.items() if r['fabric']==fabric and not r['defect'])
        bad = sorted(p for p,r in samples.items() if r['fabric']==fabric and r['defect'])
        if len(good) < 4 or len(bad) < 2:
            omitted.append(fabric)
            continue
        rng.shuffle(good)
        split = len(good)//2
        mid = split + (len(good)-split)//2
        train.extend(good[:split]); cal.extend(good[split:mid]); test.extend(good[mid:])
        permuter.shuffle(bad)
        for i,p in enumerate(bad):
            donors[p] = {k:bad[(i+1+step%(len(bad)-1))%len(bad)] for step,k in enumerate(DONORS)}
            assert p not in donors[p].values()
    return train, cal, test, donors, omitted


def global_vectors(images, backbone, device):
    vectors = backbone(torch.stack([TRANSFORM(im) for im in images]).to(device))
    return torch.nn.functional.normalize(vectors)


@torch.inference_mode()
def experiment(folders):
    samples, issues = audit_sources(folders)
    train, cal, test, donors, omitted = split_and_assign(samples)
    pos = sorted(donors)
    parents = sorted(set(pos+cal+test))
    assert not set(train).intersection(parents)
    spatial_hist = {k:[np.zeros(1,np.int64),np.zeros(1,np.int64)] for k in POLICIES}
    layouts = {p:guided_layout(source_mask(samples[p])) for p in pos}
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    backbone.fc = torch.nn.Identity()
    pool = [(p,x) for p in sorted(train) for _,x in FIXED]
    indices = sorted(np.random.default_rng(SEED).choice(len(pool),min(1000,len(pool)),replace=False))
    refs = [pool[i] for i in indices]
    deep, glob = [], []
    for start in range(0,len(refs),BATCH):
        batch = []
        for p,x in refs[start:start+BATCH]:
            with Image.open(samples[p]['path']) as im:
                batch.append(im.convert('L').crop((x,0,x+256,256)).convert('RGB'))
        deep.append(embeddings(batch,backbone,device).cpu().reshape(-1,256))
        glob.append(global_vectors(batch,backbone,device))
    bank, selected = select_bank(torch.cat(deep),device,SEED)
    del deep
    global_bank = torch.cat(glob)
    spatial, global_scores = {}, {}
    for p in parents:
        row = samples[p]
        img = Image.fromarray(gray(row['path'])).convert('RGB')
        mask = source_mask(row)
        policies = {'released':layouts.get(p,FIXED),'fixed':FIXED,'overlap':OVERLAP}
        policies.update({k:layouts[donors[p][k]] if p in donors else FIXED for k in DONORS})
        spatial[p] = score_parent(img,mask,policies,backbone,bank,device,p not in cal,spatial_hist)
        coords = list(dict.fromkeys(a for positions in policies.values() for a in positions))
        distances = {}
        for start in range(0,len(coords),BATCH):
            subset = coords[start:start+BATCH]
            batch = [img.crop((x,0,x+256,256)) for _,x in subset]
            d = torch.cdist(global_vectors(batch,backbone,device), global_bank).amin(dim=1)
            distances.update(zip(subset,d.cpu().tolist()))
        global_scores[p] = {k:max(distances[a] for a in positions) for k,positions in policies.items()}
    rolls = {p:samples[p]['fabric'] for p in parents}
    spatial_metrics, global_metrics = {}, {}
    for k in POLICIES:
        spatial_metrics[k] = summarize_policy(k,spatial,spatial_hist[k],set(pos),cal,test,rolls)
        y = [p in donors for p in parents]
        global_metrics[k] = float(roc_auc_score(y,[global_scores[p][k] for p in parents]))
        assert spatial_hist[k][0].sum() == (len(pos)+len(test))*256*4096
    return {'dataset':'AITEX; constructed guided policy is named released only for API compatibility',
            'protocol_sha256':sha256(ROOT/'reports/aitex_protocol.md'),'device':device,'seed':SEED,
            'inventory':{k:len(list(v.rglob('*.png'))) for k,v in folders.items()},
            'audit_issues':issues,'fabrics_omitted':omitted,'source_hashes':{p:r['hash'] for p,r in samples.items()},
            'train_clean':sorted(train),'calibration_clean':sorted(cal),'evaluation_clean':sorted(test),
            'defective':pos,'reference_tiles':refs,'selected_spatial_indices':selected,'donors':donors,
            'guided_coordinates':layouts,'spatial_metrics':spatial_metrics,'global_auroc':global_metrics,
            'spatial_frames':spatial,'global_scores':global_scores,
            'histograms':{k:{'all':a.tolist(),'foreground':b.tolist()} for k,(a,b) in spatial_hist.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='aitex-native-') as temp:
        output = experiment(prepare_archives(args.download,Path(temp)))
    result = ROOT/'results/aitex_external.json'
    result.parent.mkdir(exist_ok=True)
    pending = result.with_suffix('.tmp')
    pending.write_text(json.dumps(output, indent=1, allow_nan=False)+'\n')
    pending.replace(result)
    print(output['inventory'], len(output['defective']), len(output['evaluation_clean']))
    for k in POLICIES:
        print(k,output['global_auroc'][k],output['spatial_metrics'][k]['auroc'])


if __name__=='__main__':
    main()
