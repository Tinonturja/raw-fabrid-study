"""Exploratory analysis E2: memory-bank origin (tile vs full frame) crossed with inference layout (R2 tiles vs R1
native), with paired reference windows, on RAW-FABRID (7 held-out rolls) and VisA (12 categories, 25 + 15 images).
Design: P5_EXPLORATORY_E2_BANK_LAYOUT.md (hash-checked at start). Reuses the frozen Protocol 5 feature extractors,
restoration, max fusion, histogram and bootstrap code.

Usage: python p5_e2_bank_layout.py --rawfabrid <RAW_FABRID.zip> --visa <VisA dir> [--units Rollo1A candle] [--limit 2]
       python p5_e2_bank_layout.py --analyze [--limit 2]
"""
import argparse, hashlib, io, json, math, sys, time, zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_core as C
import p5_models as M

HERE = Path(__file__).resolve().parents[1]
DESIGN = HERE / 'P5_EXPLORATORY_E2_BANK_LAYOUT.md'
OUT = HERE / 'results_p5' / 'e2_bank_layout'
SIDE, N_SRC, N_WIN, N_BANK = 256, 80, 300, 10240
N_ANOM, N_NORM = 25, 15
SEEDS = (13017, 13027, 13037)
FAMS = {'D': 'dino', 'W': 'wrn50'}
BANKS = ('Btile', 'Bfull')
LAYOUTS = ('R2', 'R1')
ARMS = [f'{f}_{b}_{l}_s{s}' for f in FAMS for b in BANKS for l in LAYOUTS for s in SEEDS]
ESTIMANDS = [  # (label, (bank, layout) of x, (bank, layout) of y): estimate = AP(x) - AP(y)
    ('E_bank,full = Bfull/R1 - Btile/R1', ('Bfull', 'R1'), ('Btile', 'R1')),
    ('E_bank,tile = Bfull/R2 - Btile/R2', ('Bfull', 'R2'), ('Btile', 'R2')),
    ('E_layout,matched = Bfull/R1 - Btile/R2', ('Bfull', 'R1'), ('Btile', 'R2')),
    ('E_layout,tilebank = Btile/R2 - Btile/R1', ('Btile', 'R2'), ('Btile', 'R1'))]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arm(f, b, l, s):
    return f'{f}_{b}_{l}_s{s}'


# ---------------- data ----------------

class RawFabrid:
    def __init__(self, zpath, man):
        self.z = zipfile.ZipFile(zpath)
        self.man = man

    def read(self, name, want=None):
        b = self.z.read('RAW_FABRID/' + name)
        if want is not None and hashlib.sha256(b).hexdigest() != want:
            sys.exit(f'manifest mismatch {name}')
        return b

    def img(self, name, want=None, mode='RGB'):
        with Image.open(io.BytesIO(self.read(name, want))) as im:
            return im.convert(mode)

    def units(self):
        for roll, r in sorted(self.man['rolls'].items()):
            yield roll, r

    def bank_sources(self, roll, r):
        """Candidate windows = the unit's released reference tiles, mapped to (source frame, row, col)."""
        by_src = {}
        for t in r['refs']:
            stem, a, b = t['patch_name'][:-4].rsplit('_', 2)
            by_src.setdefault(stem, []).append((int(a), int(b), t))
        stems = sorted(by_src)
        pick = np.sort(np.random.default_rng(13018).choice(len(stems), min(N_SRC, len(stems)), replace=False))
        sources, cands = [], []
        for si, k in enumerate(pick):
            stem = stems[k]
            sources.append(('images/' + stem + '.png', lambda stem=stem: self.img('images/' + stem + '.png')))
            for a, b, t in sorted(by_src[stem], key=lambda v: (v[0], v[1])):
                cands.append((si, SIDE * a, SIDE * b, t))
        return sources, cands

    def check_tile(self, frame, y, x, t):
        tile = np.asarray(self.img('MVTec/train/good/' + t['patch_name'], t['sha256']))
        if not np.array_equal(np.asarray(frame)[y:y + SIDE, x:x + SIDE], tile):
            sys.exit(f'reference tile does not match its source frame: {t["patch_name"]}')

    def test_items(self, roll, r, limit):
        frames = sorted(r['frames'])
        if limit:
            frames = [f for f in frames if r['frames'][f]['defect']][:limit] + \
                     [f for f in frames if not r['frames'][f]['defect']][:limit]
        for f in frames:
            q = r['frames'][f]
            image = self.img('images/' + f, q['image_sha256'])
            if q['defect']:
                mask = np.asarray(self.img('masks/' + f, q['mask_sha256'], 'L')) >= 128
            else:
                mask = np.zeros(image.size[::-1], bool)
            yield f, bool(q['defect']), image, mask


class VisA:
    def __init__(self, root, man):
        self.root = Path(root)
        self.man = man
        self.imgs = man['images']

    def rgb(self, k):
        with Image.open(self.root / k) as im:
            return im.convert('RGB')

    def units(self):
        for cat in sorted(self.man['per_category']):
            yield cat, None

    def split(self, cat):
        test = sorted(k for k, r in self.imgs.items() if r['category'] == cat and r['split'] == 'test')
        a = [k for k in test if self.imgs[k]['label'] == 'anomaly'][:N_ANOM]
        n = [k for k in test if self.imgs[k]['label'] == 'normal'][:N_NORM]
        train = sorted(k for k, r in self.imgs.items() if r['category'] == cat and r['split'] == 'train')
        return train, a, n

    def verify(self, k):
        r = self.imgs[k]
        if fsha(self.root / k) != r['sha256'] or ('mask' in r and fsha(self.root / r['mask']) != r['mask_sha256']):
            sys.exit(f'manifest mismatch {k}')

    def bank_sources(self, cat, _):
        train, _, _ = self.split(cat)
        pick = np.sort(np.random.default_rng(13018).choice(len(train), min(N_SRC, len(train)), replace=False))
        sources, cands = [], []
        for si, k in enumerate(pick):
            key = train[k]
            self.verify(key)
            sources.append((key, lambda key=key: self.rgb(key)))
            with Image.open(self.root / key) as im:
                w, h = im.size
            cands += [(si, y, x, None) for y, x in C.grid_layout(h, w, SIDE)]
        return sources, cands

    def check_tile(self, frame, y, x, t):
        return

    def test_items(self, cat, _, limit):
        _, a, n = self.split(cat)
        keys = (a[:limit] + n[:limit]) if limit else (a + n)
        for k in keys:
            self.verify(k)
            r = self.imgs[k]
            image = self.rgb(k)
            w, h = image.size
            mask = C.decode_visa(self.root / r['mask']) if r['label'] == 'anomaly' else np.zeros((h, w), bool)
            yield k, r['label'] == 'anomaly', image, mask


# ---------------- banks ----------------

def cell_mask_in_box(fam, fh, fw, h, w, y, x):
    """Feature cells of a whole-image map whose centre lies inside the box, among cells wholly inside the image."""
    sy, sx = (M.PATCH, M.PATCH) if fam.name == 'dino' else (h / fh, w / fw)
    cy = (torch.arange(fh) + 0.5) * sy
    cx = (torch.arange(fw) + 0.5) * sx
    inside = ((cy >= y) & (cy < y + SIDE))[:, None] & ((cx >= x) & (cx < x + SIDE))[None, :]
    return inside & fam.valid(fh, fw, h, w)


@torch.inference_mode()
def build_banks(fam, ds, sources, cands):
    """Paired banks per seed. Returns {(kind, seed): bank tensor} and info."""
    plans = {}
    for s in SEEDS:
        ss = np.random.SeedSequence(s).spawn(4)   # window pick, tile sampling, full sampling, final pick
        rngs = [np.random.default_rng(q) for q in ss[:3]]
        idx = np.arange(len(cands))
        if len(cands) > N_WIN:
            idx = np.sort(rngs[0].choice(len(cands), N_WIN, replace=False))
        plans[s] = {'wins': set(idx.tolist()), 'per': math.ceil(N_BANK * 1.05 / len(idx)), 'rngs': rngs,
                    'Btile': [], 'Bfull': [], 'n_win': int(len(idx)), 'final_ss': ss[3]}
    used = sorted({cands[i][0] for p in plans.values() for i in p['wins']})
    for si in used:
        key, load = sources[si]
        frame = load()
        w, h = frame.size
        ff = fam.features(M.to_tensor([frame]))
        _, fh, fw, d = ff.shape
        ff = ff.reshape(fh * fw, d)
        tile_cache = {}
        for ci in [i for i, c in enumerate(cands) if c[0] == si]:
            if not any(ci in p['wins'] for p in plans.values()):
                continue
            _, y, x, t = cands[ci]
            ds.check_tile(frame, y, x, t)
            if ci not in tile_cache:
                ft = fam.features(M.to_tensor([frame.crop((x, y, x + SIDE, y + SIDE))]))
                _, th, tw, _ = ft.shape
                ok_t = torch.nonzero(fam.valid(th, tw, SIDE, SIDE).ravel()).flatten().numpy()
                ok_f = torch.nonzero(cell_mask_in_box(fam, fh, fw, h, w, y, x).ravel()).flatten().numpy()
                tile_cache[ci] = (ft.reshape(th * tw, d), ok_t, ok_f)
            ft, ok_t, ok_f = tile_cache[ci]
            for s, p in plans.items():
                if ci not in p['wins']:
                    continue
                k = min(p['per'], len(ok_t), len(ok_f))
                it = p['rngs'][1].choice(ok_t, k, replace=False)
                jf = p['rngs'][2].choice(ok_f, k, replace=False)
                p['Btile'].append(ft[torch.from_numpy(it).to(ft.device)].float().cpu())
                p['Bfull'].append(ff[torch.from_numpy(jf).to(ff.device)].float().cpu())
        del ff, tile_cache
    banks, info = {}, {}
    for s, p in plans.items():
        for kind in BANKS:
            X = torch.cat(p[kind])
            assert len(X) >= N_BANK, (kind, s, len(X))
            # same final pick for both banks of a seed: rows are paired window by window
            keep = np.sort(np.random.default_rng(p['final_ss']).choice(len(X), N_BANK, replace=False))
            banks[(kind, s)] = X[torch.from_numpy(keep)]
            info[f'{kind}_s{s}'] = {'windows': p['n_win'], 'per_window': p['per'], 'candidates': int(len(X)),
                                    'sha256': M.digest(banks[(kind, s)])}
    info['source_images'] = [sources[i][0] for i in used]
    return banks, info


def scorer(fam, bank, device):
    return M.DinoKNN(bank, device) if fam.name == 'dino' else M.PatchCore(bank, device)


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def mem(device):
    """Tensor memory currently allocated on the accelerator. Peaks are sampled after feature extraction and after
    each bank is scored, and reported above the level at the start of the call (models and banks excluded)."""
    if device == 'mps':
        return int(torch.mps.current_allocated_memory())
    if device == 'cuda':
        return int(torch.cuda.memory_allocated())
    return 0


@torch.inference_mode()
def fields(fam, scs, image, layout, device):
    """All bank fields for one image under one layout, plus timing (feature s, scoring s per bank) and memory."""
    w, h = image.size
    sync(device); base = mem(device); t0 = time.perf_counter()
    if layout == 'R1':
        parts = [(fam.features(M.to_tensor([image])), None)]
    else:
        coords = C.grid_layout(h, w, SIDE)
        parts = []
        for s in range(0, len(coords), 8):
            cs = coords[s:s + 8]
            parts.append((fam.features(M.to_tensor([image.crop((x, y, x + SIDE, y + SIDE)) for y, x in cs])), cs))
    sync(device); t1 = time.perf_counter()
    peak = mem(device)
    out = {}
    for key, sc in scs.items():
        if layout == 'R1':
            f = parts[0][0]
            _, fh, fw, d = f.shape
            fld = fam.restore(sc(f.reshape(-1, d)).reshape(1, fh, fw), h, w)[0].cpu().numpy()
        else:
            maps, coords = [], []
            for f, cs in parts:
                n, fh, fw, d = f.shape
                maps += list(fam.restore(sc(f.reshape(-1, d)).reshape(n, fh, fw), SIDE, SIDE).cpu().numpy())
                coords += cs
            fld = C.fuse_max(h, w, maps, coords, SIDE)
        out[key] = fld
        peak = max(peak, mem(device))
    sync(device); t2 = time.perf_counter()
    return out, {'feat_s': t1 - t0, 'score_s_per_bank': (t2 - t1) / len(scs), 'mem_bytes': peak - base}


@torch.inference_mode()
def run(a):
    torch.manual_seed(M.SEED)
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    want = (HERE / 'P5_EXPLORATORY_E2_BANK_LAYOUT.sha256').read_text().split()[0]
    if fsha(DESIGN) != want:
        sys.exit('E2 design file does not match its recorded hash')
    stage = 'smoke' if a.limit else 'full'
    (OUT / stage).mkdir(parents=True, exist_ok=True)
    fams = {k: (M.DINO(device) if v == 'dino' else M.WRN(device)) for k, v in FAMS.items()}
    sets = []
    if a.rawfabrid:
        sets.append(('rawfabrid', RawFabrid(a.rawfabrid, json.loads((HERE / 'manifests' / 'rawfabrid_manifest.json').read_text()))))
    if a.visa:
        sets.append(('visa', VisA(a.visa, json.loads((HERE / 'manifests' / 'visa_manifest.json').read_text()))))
    for dsname, ds in sets:
        for unit, r in ds.units():
            if a.units and unit not in a.units:
                continue
            if a.limit and a.units is None and unit not in ('Rollo3A', 'candle'):
                continue
            out_npz = OUT / stage / f'{dsname}__{unit}.npz'
            if out_npz.exists():
                log(unit, 'done, skipped')
                continue
            t0 = time.time()
            sources, cands = ds.bank_sources(unit, r)
            scs, binfo = {}, {}
            for fk, fam in fams.items():
                banks, info = build_banks(fam, ds, sources, cands)
                binfo[fk] = info
                scs[fk] = {(kind, s): scorer(fam, b, device) for (kind, s), b in banks.items()}
                log(unit, fam.name, 'banks ready', {k: v for k, v in info.items() if k != 'source_images'},
                    f'{time.time() - t0:.0f}s')
                del banks
            recs, hists = [], []
            for i, (k, defect, image, mask) in enumerate(ds.test_items(unit, r, a.limit)):
                w, h = image.size
                assert mask.shape == (h, w)
                hh, fmax, timing = {}, {}, {}
                for fk, fam in fams.items():
                    for lay in LAYOUTS:
                        flds, tm = fields(fam, scs[fk], image, lay, device)
                        timing[f'{fk}_{lay}'] = tm
                        for (kind, s), fld in flds.items():
                            assert fld.shape == (h, w)
                            name = arm(fk, kind, lay, s)
                            hh[name] = C.hist(C.to_bins(fld), mask)
                            fmax[name] = float(np.max(fld))
                hists.append(hh)
                recs.append({'id': k, 'defect': defect, 'defect_px': int(mask.sum()), 'size': [h, w],
                             'frame_max': fmax, 'timing': timing})
                if (i + 1) % 10 == 0:
                    log(unit, f'{i + 1} images {time.time() - t0:.0f}s')
            arrays = {}
            for name in ARMS:
                b, t, p, off = [], [], [], [0]
                for hh in hists:
                    hb, ht, hp = hh[name]
                    b.append(hb); t.append(ht); p.append(hp); off.append(off[-1] + len(hb))
                arrays[f'{name}|bins'] = np.concatenate(b).astype(np.int32)
                arrays[f'{name}|tot'] = np.concatenate(t).astype(np.int64)
                arrays[f'{name}|pos'] = np.concatenate(p).astype(np.int64)
                arrays[f'{name}|off'] = np.asarray(off, np.int64)
            tmp = out_npz.with_name(out_npz.stem + '.tmp.npz')
            np.savez_compressed(tmp, **arrays)
            (OUT / stage / f'{dsname}__{unit}.json').write_text(json.dumps(
                {'dataset': dsname, 'unit': unit, 'device': device, 'banks': binfo, 'images': recs}, indent=1))
            tmp.replace(out_npz)
            log(unit, 'unit done', f'{time.time() - t0:.0f}s')
    log('ALL UNITS DONE')


# ---------------- analysis ----------------

def recall_at_fpr(T, P, fpr=0.01):
    """Pixel recall at the highest-score threshold whose normal-pixel exceedance is at most `fpr` (pooled bins)."""
    N = (T - P)[::-1].cumsum()
    Pc = P[::-1].cumsum()
    ok = np.flatnonzero(N <= fpr * N[-1])
    return float(Pc[ok[-1]] / Pc[-1]) if len(ok) and Pc[-1] > 0 else 0.0


def load_unit(path):
    z = np.load(path)
    n = len(z[f'{ARMS[0]}|off']) - 1
    dens = {}
    for name in ARMS:
        off = z[f'{name}|off']
        hs = [(z[f'{name}|bins'][off[i]:off[i + 1]], z[f'{name}|tot'][off[i]:off[i + 1]],
               z[f'{name}|pos'][off[i]:off[i + 1]]) for i in range(n)]
        dens[name] = C.dense(hs)
    nb = max(v[0].shape[1] for v in dens.values())
    for name in dens:
        T, P = dens[name]
        if T.shape[1] < nb:
            pad = ((0, 0), (0, nb - T.shape[1]))
            dens[name] = (np.pad(T, pad), np.pad(P, pad))
    return n, dens


def analyze(a):
    stage = 'smoke' if a.limit else 'full'
    d = OUT / stage
    res = {'stage': stage, 'design_sha256': fsha(DESIGN), 'datasets': {}}
    for dsname in ('rawfabrid', 'visa'):
        paths = sorted(p for p in d.glob(f'{dsname}__*.npz') if not p.stem.endswith('.tmp'))
        if not paths:
            continue
        units = [p.stem.split('__', 1)[1] for p in paths]
        ap, rec, dens, C_by = {}, {}, {}, {}
        for u, p in zip(units, paths):
            n, dens[u] = load_unit(p)
            ap[u] = {k: float(C.ap_from_hist(T.sum(0), P.sum(0))) for k, (T, P) in dens[u].items()}
            rec[u] = {k: recall_at_fpr(T.sum(0), P.sum(0)) for k, (T, P) in dens[u].items()}
            C_by[u] = C.bootstrap_counts(n, seed=C.unit_seed(u))
        out = {'units': units, 'ap': ap, 'recall_at_1pct': rec, 'cells': {}, 'estimands': {}}
        print(f'\n== {dsname} ({stage}), {len(units)} units ==')
        for fk in FAMS:
            for b in BANKS:
                for l in LAYOUTS:
                    v = [np.mean([ap[u][arm(fk, b, l, s)] for s in SEEDS]) for u in units]
                    rv = [np.mean([rec[u][arm(fk, b, l, s)] for s in SEEDS]) for u in units]
                    per_seed = [float(np.mean([ap[u][arm(fk, b, l, s)] for u in units])) for s in SEEDS]
                    out['cells'][f'{fk}_{b}_{l}'] = {'ap': float(np.mean(v)), 'ap_per_seed': per_seed,
                                                     'recall_at_1pct': float(np.mean(rv))}
                    print(f'  {fk} {b:5s} {l}: AP {np.mean(v):.4f} (seeds {", ".join(f"{x:.4f}" for x in per_seed)})'
                          f'  recall@1%FPR {np.mean(rv):.4f}')
            for label, (bx, lx), (by, ly) in ESTIMANDS:
                per_unit = np.array([np.mean([ap[u][arm(fk, bx, lx, s)] - ap[u][arm(fk, by, ly, s)] for s in SEEDS])
                                     for u in units])
                per_seed = [float(np.mean([ap[u][arm(fk, bx, lx, s)] - ap[u][arm(fk, by, ly, s)] for u in units]))
                            for s in SEEDS]
                est = float(per_unit.mean())
                if dsname == 'rawfabrid':
                    lo, hi = C.percentile_ci(C.roll_bootstrap(per_unit))
                    pval = C.signflip_p(per_unit)
                else:
                    dists = []
                    for s in SEEDS:
                        x, y = arm(fk, bx, lx, s), arm(fk, by, ly, s)
                        dist, _ = C.macro_boot({u: (*dens[u][x], *dens[u][y]) for u in units}, C_by)
                        dists.append(dist)
                    lo, hi = C.percentile_ci(np.mean(dists, 0))
                    pval = None
                out['estimands'][f'{fk} {label}'] = {'estimate': est, 'ci95': [lo, hi], 'per_seed': per_seed,
                                                     'units_positive': int((per_unit > 0).sum()),
                                                     'signflip_p': pval}
                print(f'  {fk} {label:42s} {est:+.4f} [{lo:+.4f}, {hi:+.4f}] seeds '
                      f'{", ".join(f"{x:+.4f}" for x in per_seed)}  {(per_unit > 0).sum()}/{len(units)} units +'
                      + (f'  p={pval:.4f}' if pval is not None else ''))
            # interaction, seed-averaged
            inter = np.array([np.mean([(ap[u][arm(fk, 'Bfull', 'R1', s)] - ap[u][arm(fk, 'Btile', 'R1', s)])
                                       - (ap[u][arm(fk, 'Bfull', 'R2', s)] - ap[u][arm(fk, 'Btile', 'R2', s)])
                                       for s in SEEDS]) for u in units])
            out['estimands'][f'{fk} interaction'] = {'estimate': float(inter.mean())}
            print(f'  {fk} interaction (E_bank,full - E_bank,tile) {inter.mean():+.4f}')
        # timing and memory
        tm = {}
        for p in paths:
            for r in json.loads(p.with_suffix('.json').read_text())['images']:
                for key, v in r['timing'].items():
                    tm.setdefault(key, []).append(v)
        out['timing'] = {k: {'feat_s_median': float(np.median([x['feat_s'] for x in v])),
                             'score_s_per_bank_median': float(np.median([x['score_s_per_bank'] for x in v])),
                             'mem_bytes_max': int(max(x['mem_bytes'] for x in v))} for k, v in tm.items()}
        for k, v in out['timing'].items():
            print(f'  timing {k}: features {v["feat_s_median"]:.2f}s, scoring {v["score_s_per_bank_median"]:.2f}s/bank,'
                  f' sampled peak tensor memory above baseline {v["mem_bytes_max"] / 2**30:.2f} GiB')
        res['datasets'][dsname] = out
    # reading rule for DINOv2
    verdict = {}
    for dsname, out in res['datasets'].items():
        e = out['estimands']['D E_bank,full = Bfull/R1 - Btile/R1']
        m = out['estimands']['D E_layout,matched = Bfull/R1 - Btile/R2']
        ca = e['estimate'] >= 0.05 and e['ci95'][0] > 0 and all(x > 0 for x in e['per_seed'])
        cb = m['ci95'][1] >= -0.01
        verdict[dsname] = {'a': bool(ca), 'b': bool(cb)}
    n_ok = sum(v['a'] and v['b'] for v in verdict.values())
    res['reading_rule'] = {'per_dataset': verdict,
                           'outcome': ('supported' if n_ok == 2 else 'partly supported' if n_ok == 1 else 'not supported')
                           if len(verdict) == 2 else 'incomplete'}
    print('\nreading rule:', res['reading_rule'])
    (d / 'analysis.json').write_text(json.dumps(res, indent=1))
    print('wrote', d / 'analysis.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rawfabrid')
    ap.add_argument('--visa')
    ap.add_argument('--units', nargs='*')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyze', action='store_true')
    a = ap.parse_args()
    analyze(a) if a.analyze else run(a)
