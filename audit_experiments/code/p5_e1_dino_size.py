"""Protocol 5 exploratory addendum E1: DINOv2 input-size / bank-context ablation on VisA.
Design: P5_EXPLORATORY_DINO_INPUT_SIZE.md (hash-checked at start). Reuses the frozen P5 feature
extractors, restoration, fusion, histogram and bootstrap code.

Usage: python p5_e1_dino_size.py --data <VisA dir> [--categories candle ...] [--limit 2]
       python p5_e1_dino_size.py --analyze
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_core as C
import p5_models as M

HERE = Path(__file__).resolve().parents[1]
DESIGN = HERE / 'P5_EXPLORATORY_DINO_INPUT_SIZE.md'
OUT = HERE / 'results_p5' / 'e1_dino_size'
N_ANOM, N_NORM = 25, 15
N_WIN, N_BANK = 300, 10240
S518 = 518
ARMS = [  # (name, family, bank, inference)
    ('D_b256_t256', 'dino', 'B256', 256), ('D_b256_t512', 'dino', 'B256', 512),
    ('D_b256_t768', 'dino', 'B256', 768), ('D_b256_full', 'dino', 'B256', 'full'),
    ('D_b512_t512', 'dino', 'B512', 512), ('D_b768_t768', 'dino', 'B768', 768),
    ('D_bfull_full', 'dino', 'Bfull', 'full'), ('D_b518_s518', 'dino', 'B518', 's518'),
    ('W_b256_t256', 'wrn50', 'B256', 256), ('W_b256_full', 'wrn50', 'B256', 'full'),
    ('W_bfull_full', 'wrn50', 'Bfull', 'full')]
CONTRASTS = [
    ('1 context: D Bfull/full - B256/full', 'D_bfull_full', 'D_b256_full'),
    ('2 residual tiling: D B256/256 - Bfull/full', 'D_b256_t256', 'D_bfull_full'),
    ('3 matched size: D B512/512 - B256/256', 'D_b512_t512', 'D_b256_t256'),
    ('3 matched size: D B768/768 - B256/256', 'D_b768_t768', 'D_b256_t256'),
    ('3 matched size: D Bfull/full - B256/256', 'D_bfull_full', 'D_b256_t256'),
    ('4 mismatched size: D B256/512 - B256/256', 'D_b256_t512', 'D_b256_t256'),
    ('4 mismatched size: D B256/768 - B256/256', 'D_b256_t768', 'D_b256_t256'),
    ('4 mismatched size: D B256/full - B256/256', 'D_b256_full', 'D_b256_t256'),
    ('5 train res: D B518/518 - B256/256', 'D_b518_s518', 'D_b256_t256'),
    ('5 train res: D B518/518 - Bfull/full', 'D_b518_s518', 'D_bfull_full'),
    ('6 CNN context: W Bfull/full - B256/full', 'W_bfull_full', 'W_b256_full'),
    ('6 CNN tiling: W B256/256 - B256/full', 'W_b256_t256', 'W_b256_full')]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rgb(p):
    with Image.open(p) as im:
        return im.convert('RGB')


def resize518(im):
    w, h = im.size
    s = S518 / max(w, h)
    return im.resize((max(14, round(w * s)), max(14, round(h * s))), Image.BILINEAR)


def windows(kind, paths):
    """Eligible bank windows for one category: list of (path, box or None, mode)."""
    out = []
    for p in paths:
        if kind in ('Bfull', 'B518'):
            out.append((p, None, kind))
        else:
            side = int(kind[1:])
            with Image.open(p) as im:
                w, h = im.size
            out += [(p, (x, y, x + side, y + side), kind) for y, x in C.grid_layout(h, w, side)]
    return out


def load_window(win):
    p, box, kind = win
    im = rgb(p)
    if kind == 'B518':
        return resize518(im)
    return im.crop(box) if box else im


@torch.inference_mode()
def build_bank(fam, kind, paths):
    wins = windows(kind, paths)
    if len(wins) > N_WIN:
        pick = np.sort(np.random.default_rng(13018).choice(len(wins), N_WIN, replace=False))
        wins = [wins[i] for i in pick]
    per = math.ceil(N_BANK * 1.05 / len(wins))
    rng = np.random.default_rng(13017)
    vecs = []
    for win in wins:
        im = load_window(win)
        w, h = im.size
        f = fam.features(M.to_tensor([im]))
        _, fh, fw, d = f.shape
        ok = torch.nonzero(fam.valid(fh, fw, h, w).ravel()).flatten().numpy()
        idx = rng.choice(ok, min(per, len(ok)), replace=False)
        vecs.append(f.reshape(fh * fw, d)[torch.from_numpy(idx).to(f.device)].float().cpu())
    X = torch.cat(vecs)
    keep = np.sort(np.random.default_rng(13019).choice(len(X), N_BANK, replace=False))
    return X[torch.from_numpy(keep)], {'windows': len(wins), 'per_window': per, 'candidates': len(X)}


def scorer(fam, bank, device):
    return M.DinoKNN(bank, device) if fam.name == 'dino' else M.PatchCore(bank, device)


@torch.inference_mode()
def feats_for(fam, image, inf):
    """Features for one image under one inference configuration; reused for every bank."""
    w, h = image.size
    if inf == 'full':
        return ('full', [(fam.features(M.to_tensor([image])), None)], (h, w))
    if inf == 's518':
        small = resize518(image)
        return ('s518', [(fam.features(M.to_tensor([small])), small.size)], (h, w))
    side = int(inf)
    coords = C.grid_layout(h, w, side)
    out = []
    for s in range(0, len(coords), 8):
        cs = coords[s:s + 8]
        crops = [image.crop((x, y, x + side, y + side)) for y, x in cs]
        out.append((fam.features(M.to_tensor(crops)), cs))
    return (side, out, (h, w))


@torch.inference_mode()
def field_from(fam, sc, ft):
    kind, parts, (h, w) = ft
    if kind == 'full':
        f = parts[0][0]
        _, fh, fw, d = f.shape
        return fam.restore(sc(f.reshape(-1, d)).reshape(1, fh, fw), h, w)[0].cpu().numpy()
    if kind == 's518':
        f, (sw, sh) = parts[0]
        _, fh, fw, d = f.shape
        s = sc(f.reshape(-1, d)).reshape(1, fh, fw)
        hp, wp = fh * M.PATCH, fw * M.PATCH
        m = F.interpolate(s[:, None], size=(hp, wp), mode='bilinear', align_corners=False)[:, :, :sh, :sw]
        m = F.interpolate(m, size=(h, w), mode='bilinear', align_corners=False)[:, 0]
        return M.smooth(m)[0].cpu().numpy()
    side = kind
    maps, coords = [], []
    for f, cs in parts:
        n, fh, fw, d = f.shape
        maps += list(fam.restore(sc(f.reshape(-1, d)).reshape(n, fh, fw), side, side).cpu().numpy())
        coords += cs
    return C.fuse_max(h, w, maps, coords, side)


def subset(man, cat):
    imgs = man['images']
    test = sorted(k for k, r in imgs.items() if r['category'] == cat and r['split'] == 'test')
    a = [k for k in test if imgs[k]['label'] == 'anomaly'][:N_ANOM]
    n = [k for k in test if imgs[k]['label'] == 'normal'][:N_NORM]
    train = sorted(k for k, r in imgs.items() if r['category'] == cat and r['split'] == 'train')
    return train, a + n


@torch.inference_mode()
def run(a):
    torch.manual_seed(M.SEED)
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    want = (HERE / 'P5_EXPLORATORY_DINO_INPUT_SIZE.sha256').read_text().split()[0]
    if fsha(DESIGN) != want:
        sys.exit('E1 design file does not match its recorded hash')
    man = json.loads((HERE / 'manifests' / 'visa_manifest.json').read_text())
    data = Path(a.data)
    stage = 'smoke' if a.limit else 'full'
    (OUT / stage).mkdir(parents=True, exist_ok=True)
    (OUT / 'banks').mkdir(parents=True, exist_ok=True)
    fams = {'dino': M.DINO(device), 'wrn50': M.WRN(device)}
    for cat in sorted(man['per_category'])[::2]:            # E1 amendment 1: every other category
        if a.categories and cat not in a.categories:
            continue
        out_npz = OUT / stage / f'{cat}.npz'
        if out_npz.exists():
            log(cat, 'done, skipped')
            continue
        train, test = subset(man, cat)
        if a.limit:
            test = test[:a.limit] + test[N_ANOM:N_ANOM + a.limit]
        for k in train + test:
            r = man['images'][k]
            if fsha(data / k) != r['sha256'] or ('mask' in r and fsha(data / r['mask']) != r['mask_sha256']):
                sys.exit(f'manifest mismatch {k}')
        t0 = time.time()
        scs, binfo = {}, {}
        for fam_name, kind in sorted({(f, b) for _, f, b, _ in ARMS}):
            path = OUT / 'banks' / f'{cat}_{fam_name}_{kind}.pt'
            if path.exists():
                d = torch.load(path)
            else:
                bank, info = build_bank(fams[fam_name], kind, [data / k for k in train])
                d = {'bank': bank, 'info': info}
                torch.save(d, path)
            binfo[f'{fam_name}_{kind}'] = dict(d['info'], sha256=M.digest(d['bank']))
            scs[(fam_name, kind)] = scorer(fams[fam_name], d['bank'], device)
            log(cat, fam_name, kind, 'bank ready', d['info'])
        recs, hists = [], []
        for i, k in enumerate(test):
            r = man['images'][k]
            image = rgb(data / k)
            w, h = image.size
            mask = C.decode_visa(data / r['mask']) if r['label'] == 'anomaly' else np.zeros((h, w), bool)
            hh = {}
            fmax = {}
            cache = {}
            for name, fam_name, kind, inf in ARMS:
                if (fam_name, inf) not in cache:
                    cache[(fam_name, inf)] = feats_for(fams[fam_name], image, inf)
                fld = field_from(fams[fam_name], scs[(fam_name, kind)], cache[(fam_name, inf)])
                assert fld.shape == (h, w)
                hh[name] = C.hist(C.to_bins(fld), mask)
                fmax[name] = float(np.max(fld))
            del cache
            hists.append(hh)
            recs.append({'id': k, 'defect': r['label'] == 'anomaly', 'defect_px': int(mask.sum()), 'frame_max': fmax})
            if (i + 1) % 10 == 0:
                log(cat, f'{i + 1}/{len(test)} images {time.time() - t0:.0f}s')
        arrays = {}
        for name, *_ in ARMS:
            b, t, p, off = [], [], [], [0]
            for hh in hists:
                hb, ht, hp = hh[name]
                b.append(hb); t.append(ht); p.append(hp); off.append(off[-1] + len(hb))
            arrays[f'{name}|bins'] = np.concatenate(b).astype(np.int32)
            arrays[f'{name}|tot'] = np.concatenate(t).astype(np.int64)
            arrays[f'{name}|pos'] = np.concatenate(p).astype(np.int64)
            arrays[f'{name}|off'] = np.asarray(off, np.int64)
        np.savez_compressed(out_npz.with_name(cat + '.tmp.npz'), **arrays)
        (OUT / stage / f'{cat}.json').write_text(json.dumps({'category': cat, 'banks': binfo, 'images': recs}, indent=1))
        out_npz.with_name(cat + '.tmp.npz').replace(out_npz)
        log(cat, 'unit done', f'{time.time() - t0:.0f}s')
    log('ALL UNITS DONE')


def analyze(a):
    stage = 'smoke' if a.limit else 'full'
    d = OUT / stage
    cats = sorted(p.stem for p in d.glob('*.npz') if not p.stem.endswith('.tmp'))
    dens, C_by, per_cat = {}, {}, {}
    for cat in cats:
        z = np.load(d / f'{cat}.npz')
        n = len(z[f'{ARMS[0][0]}|off']) - 1
        dens[cat] = {}
        for name, *_ in ARMS:
            off = z[f'{name}|off']
            hs = [(z[f'{name}|bins'][off[i]:off[i + 1]], z[f'{name}|tot'][off[i]:off[i + 1]],
                   z[f'{name}|pos'][off[i]:off[i + 1]]) for i in range(n)]
            dens[cat][name] = C.dense(hs)
        nb = max(v[0].shape[1] for v in dens[cat].values())
        for name in dens[cat]:
            T, P = dens[cat][name]
            if T.shape[1] < nb:
                pad = ((0, 0), (0, nb - T.shape[1]))
                dens[cat][name] = (np.pad(T, pad), np.pad(P, pad))
        C_by[cat] = C.bootstrap_counts(n, seed=C.unit_seed(cat))   # independent per category, shared across arms
        per_cat[cat] = {name: float(C.ap_from_hist(T.sum(0), P.sum(0))) for name, (T, P) in dens[cat].items()}
    macro = {name: float(np.mean([per_cat[c][name] for c in cats])) for name, *_ in ARMS}
    res = {'stage': stage, 'categories': cats, 'macro_ap': macro, 'per_category_ap': per_cat, 'contrasts': {}}
    print(f'E1 ({stage}), {len(cats)} categories')
    for name, *_ in ARMS:
        print(f'  {name:14s} macro AP {macro[name]:.4f}')
    for label, x, y in CONTRASTS:
        dist, per = C.macro_boot({c: (*dens[c][x], *dens[c][y]) for c in cats}, C_by)
        est = macro[x] - macro[y]
        lo, hi = C.percentile_ci(dist)
        pos = int(sum(per_cat[c][x] > per_cat[c][y] for c in cats))
        res['contrasts'][label] = {'estimate': est, 'ci95': [float(lo), float(hi)], 'categories_positive': pos}
        print(f'  {label:45s} {est:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {pos}/{len(cats)} positive')
    (d / 'analysis.json').write_text(json.dumps(res, indent=1))
    print('wrote', d / 'analysis.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data')
    ap.add_argument('--categories', nargs='*')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyze', action='store_true')
    a = ap.parse_args()
    analyze(a) if a.analyze else run(a)
