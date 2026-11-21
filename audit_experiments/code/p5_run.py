"""Protocol 5 runner: R1 (native full frame) vs R2 (tiled, stitched once per unique pixel),
plus practice A (AITEX), practice C (x4 downscale) and the counting analyses from the same predictions.

One "unit" = AITEX (whole cohort), one VisA category, or one RAW-FABRID roll. Each unit writes one
.npz of per-image sparse histograms and one .json of per-image and per-tile records, then is skipped
on restart. Every input file is checked against its frozen manifest before use.

Usage:
  python p5_run.py --dataset aitex --data <aitex dir> [--limit 5] [--families wrn dino]
  python p5_run.py --dataset visa  --data <VisA dir> [--categories candle ...] [--limit 5]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_core as C
import p5_models as M

HERE = Path(__file__).resolve().parents[1]
LAYOUTS = ('R1', 'R2', 'R2occ', 'A_all', 'A_obs', 'C4')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rgb(path):
    with Image.open(path) as im:
        return im.convert('RGB')


# ---------- dataset adapters ----------

def aitex_units(data, man, limit):
    imgs = man['images']
    check = lambda n: fsha(data / imgs[n]['path']) == imgs[n]['sha256'] or sys.exit(f'manifest mismatch {n}')
    train = sorted(n for n, r in imgs.items() if r['role'] == 'bank_train_clean')
    test = sorted(n for n, r in imgs.items() if r['role'] in ('evaluation_defective', 'evaluation_clean'))
    for n in train + test:
        check(n)
        for m in imgs[n].get('masks', []):
            if fsha(data / m['path']) != m['sha256']:
                sys.exit(f'manifest mismatch {m["path"]}')
    tiles = [(lambda p=data / imgs[n]['path'], x=x: rgb(p).crop((x, 0, x + 256, 256)))
             for n in train for _, x in C.grid_layout(256, 4096)]
    if limit:
        d = [n for n in test if imgs[n]['role'] == 'evaluation_defective'][:limit]
        test = d + [n for n in test if imgs[n]['role'] == 'evaluation_clean'][:limit]
    items = []
    for n in test:
        r = imgs[n]
        defect = r['role'] == 'evaluation_defective'
        items.append({'id': n, 'group': r['fabric'], 'defect': defect,
                      'image': (lambda p=data / r['path']: rgb(p)),
                      'mask': (lambda r=r: C.decode_aitex([data / m['path'] for m in r['masks']]))
                      if defect else None,
                      'A': True})
    yield 'aitex', tiles, items


def visa_units(data, man, cats, limit):
    imgs = man['images']
    for cat in sorted(man['per_category']):
        if cats and cat not in cats:
            continue
        train = sorted(k for k, r in imgs.items() if r['category'] == cat and r['split'] == 'train')
        test = sorted(k for k, r in imgs.items() if r['category'] == cat and r['split'] == 'test')
        for k in train + test:
            if fsha(data / k) != imgs[k]['sha256'] or ('mask' in imgs[k] and fsha(data / imgs[k]['mask']) != imgs[k]['mask_sha256']):
                sys.exit(f'manifest mismatch {k}')
        tiles = []
        for k in train:
            with Image.open(data / k) as im:
                w, h = im.size
            tiles += [(lambda p=data / k, y=y, x=x: rgb(p).crop((x, y, x + 256, y + 256)))
                      for y, x in C.grid_layout(h, w)]
        if limit:
            test = [k for k in test if imgs[k]['label'] == 'anomaly'][:limit] + \
                   [k for k in test if imgs[k]['label'] == 'normal'][:limit]
        items = [{'id': k, 'group': cat, 'defect': imgs[k]['label'] == 'anomaly',
                  'image': (lambda p=data / k: rgb(p)),
                  'mask': (lambda p=data / imgs[k]['mask']: C.decode_visa(p)) if imgs[k]['label'] == 'anomaly' else None,
                  'A': False} for k in test]
        yield cat, tiles, items


def rawfabrid_units(zpath, man, rolls, limit):
    import io, zipfile
    z = zipfile.ZipFile(zpath)

    def member(name, want):
        b = z.read('RAW_FABRID/' + name)
        if hashlib.sha256(b).hexdigest() != want:
            sys.exit(f'manifest mismatch {name}')
        return b

    def img(name, want, mode='RGB'):
        with Image.open(io.BytesIO(member(name, want))) as im:
            return im.convert(mode)
    for roll, r in sorted(man['rolls'].items()):
        if rolls and roll not in rolls:
            continue
        tiles = [(lambda t=t: img('MVTec/train/good/' + t['patch_name'], t['sha256'])) for t in r['refs']]
        frames = sorted(r['frames'])
        if limit:
            frames = [f for f in frames if r['frames'][f]['defect']][:limit] + \
                     [f for f in frames if not r['frames'][f]['defect']][:limit]
        items = [{'id': f, 'group': roll, 'defect': r['frames'][f]['defect'],
                  'image': (lambda f=f: img('images/' + f, r['frames'][f]['image_sha256'])),
                  'mask': (lambda f=f: np.asarray(img('masks/' + f, r['frames'][f]['mask_sha256'], 'L')) >= 128)
                  if r['frames'][f]['defect'] else None,
                  'A': [tuple(c) for c in r['frames'][f]['released']]} for f in frames]
        preset = {'wrn50': (HERE / 'results' / 'wrn50' / 'banks' / f'{roll}.pt', r['protocol1_wrn50_bank_sha256'])}
        yield roll, tiles, items, preset


# ---------- banks ----------

def build_scorers(fams, tiles, bank_dir, unit, device, preset=None):
    bank_dir.mkdir(parents=True, exist_ok=True)
    scorers, info = {}, {}
    for fam in fams:
        path = bank_dir / f'{unit}_{fam.name}.pt'
        if preset and fam.name in preset and not path.exists():
            ppath, want = preset[fam.name]
            pbank = torch.load(ppath).float()
            if M.digest(pbank) != want:
                sys.exit(f'Protocol 1 bank {ppath} does not match its recorded hash')
            X, X_info = M.sample_candidates(fam, tiles, log=lambda *a: log(unit, *a))
            mu, prec = M.SharedGauss.fit(X)
            del X
            X_info['bank_from'] = f'Protocol 1 cache {ppath.name}, sha256 {want}'
            torch.save({'info': X_info, 'bank': pbank, 'mu': mu, 'prec': prec}, path)
            log(unit, fam.name, 'Protocol 1 bank reused; Gaussian fitted on re-sampled candidates')
        if path.exists():
            d = torch.load(path)
            X_info, bank, mu, prec = d['info'], d['bank'], d.get('mu'), d.get('prec')
            log(unit, fam.name, 'bank loaded from cache')
        else:
            t0 = time.time()
            X, X_info = M.sample_candidates(fam, tiles, log=lambda *a: log(unit, *a))
            bank = M.greedy_coreset(X, 10240, 'cpu', log=lambda *a: log(unit, fam.name, *a))  # CPU: same formula, about 3.5x faster than MPS
            mu, prec = M.SharedGauss.fit(X) if fam.name == 'wrn50' else (None, None)
            del X
            torch.save({'info': X_info, 'bank': bank, 'mu': mu, 'prec': prec}, path)
            log(unit, fam.name, f'bank built {time.time() - t0:.0f}s', X_info)
        X_info = dict(X_info, bank_sha256=M.digest(bank), coreset=len(bank))
        if fam.name == 'wrn50':
            g = M.SharedGauss(mu, prec, X_info['candidates'], device)
            scorers[fam.name] = [M.PatchCore(bank, device), g]
            X_info['gauss'] = g.info
        else:
            scorers[fam.name] = [M.DinoKNN(bank, device)]
        info[fam.name] = X_info
    return scorers, info


# ---------- one image ----------

def tile_maps(fam, scs, image, coords, batch=16):
    out = {s.name: [] for s in scs}
    for s in range(0, len(coords), batch):
        crops = [image.crop((x, y, x + 256, y + 256)) for y, x in coords[s:s + batch]]
        for k, v in M.score_maps(fam, scs, crops).items():
            out[k].extend(list(v))
    return out


def full_map(fam, scs, image):
    return {k: v[0] for k, v in M.score_maps(fam, scs, [image]).items()}


def down4_map(fam, scs, image):
    w, h = image.size
    small = image.resize((max(w // 4, 64), max(h // 4, 64)), Image.BILINEAR)
    out = {}
    for k, v in M.score_maps(fam, scs, [small]).items():
        t = torch.from_numpy(v)[None]
        out[k] = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)[0, 0].numpy()
    return out


def process(item, fams, scorers, failures):
    image = item['image']()
    w, h = image.size
    mask = item['mask']() if item['defect'] else np.zeros((h, w), bool)
    assert mask.shape == (h, w), (item['id'], mask.shape, (h, w))
    grid = C.grid_layout(h, w)
    if isinstance(item['A'], list):
        a_coords = item['A']                          # released offsets (RAW-FABRID)
    else:
        a_coords = C.aitex_original_layout(mask) if item['A'] else []
    coords = list(dict.fromkeys(grid + a_coords))
    idx = {c: i for i, c in enumerate(coords)}
    hists, tiles_rec, frame = {}, [], {}
    tmask = np.stack([mask[y:y + 256, x:x + 256] for y, x in grid])
    for fam in fams:
        scs = scorers[fam.name]
        tm = tile_maps(fam, scs, image, coords)
        try:
            r1 = full_map(fam, scs, image)
        except RuntimeError as e:                       # memory: R1 for this family moves to a GPU (logged)
            failures.append({'id': item['id'], 'family': fam.name, 'error': str(e)[:200]})
            r1 = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        c4 = down4_map(fam, scs, image)
        for sc in scs:
            k = sc.name
            maps = tm[k]
            fields = {'R2': C.fuse_max(h, w, [maps[idx[c]] for c in grid], grid), 'C4': c4[k]}
            if r1 is not None:
                fields['R1'] = r1[k]
            if a_coords:
                fields['A'] = C.fuse_max(h, w, [maps[idx[c]] for c in a_coords], a_coords)
            for lay, fld in fields.items():
                if lay == 'A':
                    seen = np.isfinite(fld)
                    hists[(k, 'A_all')] = C.hist(C.to_bins(fld), mask)
                    hists[(k, 'A_obs')] = C.hist(C.to_bins(fld[seen]), mask[seen])
                else:
                    hists[(k, lay)] = C.hist(C.to_bins(fld), mask)
                frame[f'{k}|{lay}'] = float(np.max(fld))
            occ = np.stack([maps[idx[c]] for c in grid])
            hists[(k, 'R2occ')] = C.hist(C.to_bins(occ), tmask)
            for j, (y, x) in enumerate(grid):
                if len(tiles_rec) <= j:
                    tiles_rec.append({'y': y, 'x': x, 'defect_px': int(tmask[j].sum())})
                tiles_rec[j][k] = float(maps[idx[(y, x)]].max())
    rec = {'id': item['id'], 'group': item['group'], 'defect': item['defect'], 'h': h, 'w': w,
           'defect_px': int(mask.sum()), 'frame_max': frame, 'tiles': tiles_rec,
           'grid_windows': len(grid), 'grid_pixel_occurrences': int(C.coverage(h, w, grid).sum())}
    if a_coords:
        cov = C.coverage(h, w, a_coords)
        rec['A'] = {'coords': a_coords, 'unseen_defect_px': int((mask & (cov == 0)).sum()),
                    'unseen_px': int((cov == 0).sum())}
    return rec, hists


def pack(all_hists, out_npz):
    arrays = {}
    keys = sorted({k for h in all_hists for k in h})
    for sc, lay in keys:
        b, t, p, off = [], [], [], [0]
        for h in all_hists:
            hb, ht, hp = h.get((sc, lay), (np.zeros(0, np.int32),) * 3)
            b.append(hb); t.append(ht); p.append(hp); off.append(off[-1] + len(hb))
        arrays[f'{sc}|{lay}|bins'] = np.concatenate(b).astype(np.int32)
        arrays[f'{sc}|{lay}|tot'] = np.concatenate(t).astype(np.int64)
        arrays[f'{sc}|{lay}|pos'] = np.concatenate(p).astype(np.int64)
        arrays[f'{sc}|{lay}|off'] = np.asarray(off, np.int64)
    tmp = out_npz.with_name(out_npz.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out_npz)


@torch.inference_mode()
def main(a):
    torch.manual_seed(M.SEED)
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    amend = HERE / 'PROTOCOL_5_AMENDMENT_1.md'
    if fsha(amend) != (HERE / 'PROTOCOL_5_AMENDMENT_1.sha256').read_text().split()[0]:
        sys.exit('PROTOCOL_5_AMENDMENT_1.md does not match its frozen hash')
    impl = HERE / 'PROTOCOL_5_IMPLEMENTATION.md'
    if not a.limit and (not impl.exists() or fsha(impl) != (HERE / 'PROTOCOL_5_IMPLEMENTATION.sha256').read_text().split()[0]):
        sys.exit('full runs need the frozen PROTOCOL_5_IMPLEMENTATION.md')
    man = json.loads((HERE / 'manifests' / f'{a.dataset}_manifest.json').read_text())
    out_dir = HERE / 'results_p5' / a.dataset / ('smoke' if a.limit else 'full')
    out_dir.mkdir(parents=True, exist_ok=True)
    fams = [M.WRN(device) if f == 'wrn' else M.DINO(device) for f in a.families]
    if a.dataset == 'aitex':
        units = ((u, t, i, None) for u, t, i in aitex_units(Path(a.data), man, a.limit))
    elif a.dataset == 'visa':
        units = ((u, t, i, None) for u, t, i in visa_units(Path(a.data), man, a.categories, a.limit))
    else:
        units = rawfabrid_units(Path(a.data), man, a.categories, a.limit)
    tag = '+'.join(sorted(a.families))
    for unit, tiles, items, preset in units:
        out_json = out_dir / f'{unit}__{tag}.json'
        if out_json.exists() and not a.overwrite:
            log(unit, 'done earlier, skipped')
            continue
        t0 = time.time()
        scorers, bank_info = build_scorers(fams, tiles, HERE / 'results_p5' / a.dataset / 'banks', unit, device, preset)
        recs, hists, failures = [], [], []
        for n, it in enumerate(items, 1):
            r, h = process(it, fams, scorers, failures)
            recs.append(r); hists.append(h)
            if n % 10 == 0 or n == len(items):
                log(unit, f'{n}/{len(items)} images', f'{time.time() - t0:.0f}s')
        pack(hists, out_dir / f'{unit}__{tag}.npz')
        out = {'unit': unit, 'dataset': a.dataset, 'device': device, 'torch': torch.__version__,
               'families': a.families, 'banks': bank_info, 'bins_per_log1p': C.BPL,
               'images': recs, 'r1_failures': failures, 'seconds': time.time() - t0,
               'manifest_sha256': fsha(HERE / 'manifests' / f'{a.dataset}_manifest.json')}
        tmp = out_json.with_suffix('.tmp')
        tmp.write_text(json.dumps(out, separators=(',', ':')))
        tmp.replace(out_json)
        log(unit, 'unit done', f'{time.time() - t0:.0f}s', f'R1 failures: {len(failures)}')
    log('ALL UNITS DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=('aitex', 'visa', 'rawfabrid'), required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--families', nargs='+', default=['wrn', 'dino'], choices=('wrn', 'dino'))
    ap.add_argument('--categories', nargs='*', help='VisA categories or RAW-FABRID rolls')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    main(ap.parse_args())
