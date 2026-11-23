"""Held-out-roll localization with stronger PatchCore settings and a denominator-matched AP.

Independent harness written for the audit. Reads the original project's folds,
cohorts and archive; writes only inside results/.
See ../PROTOCOL.md for the fixed design.
"""
import argparse, csv, hashlib, io, json, sys, time, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label, maximum as nd_maximum
from torchvision import transforms
from torchvision.models import (ResNet18_Weights, Wide_ResNet50_2_Weights,
                                resnet18, wide_resnet50_2)

HERE = Path(__file__).resolve().parents[1]
ORIG = HERE.parent / 'Project files'
SEED = 13017
SIDE = 256
FIXED = [(y, x) for y in range(0, 769, SIDE) for x in range(0, 1537, SIDE)]
OVERLAP = [(y, x) for y in range(0, 769, 128) for x in range(0, 1537, 128)]
POLICIES = ('released', 'fixed', 'overlap', 'gapfill')
BPL = 5000          # bins per unit of log1p(score)
BATCH = 16
NATIVE = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def to_bins(v):
    out = np.zeros(v.shape, np.int64)
    m = np.isfinite(v)
    out[m] = (np.log1p(np.maximum(v[m], 0.0)) * BPL).astype(np.int64) + 1
    return out


def add_hist(store, bins, fg):
    tot = np.bincount(bins)
    pos = np.bincount(bins, weights=fg).astype(np.int64)
    for i, arr in enumerate((tot, pos)):
        if store[i].size < arr.size:
            store[i] = np.pad(store[i], (0, arr.size - store[i].size))
        store[i][:arr.size] += arr


def gauss_kernel(sigma=4.0):
    r = int(4 * sigma)
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


KER = gauss_kernel()


def smooth_up(scores, device):
    """(n,h,w) lattice scores -> (n,256,256) bilinear + Gaussian sigma 4."""
    m = F.interpolate(scores[:, None], size=(SIDE, SIDE), mode='bilinear', align_corners=False)
    k = KER.to(device)
    r = len(k) // 2
    m = F.conv2d(F.pad(m, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1))
    return m[:, 0]


class Native:
    """PatchCore-style local features at native 256 px input."""

    def __init__(self, name, device):
        if name == 'r18_native':
            self.net, self.dim = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), None
        else:
            self.net, self.dim = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1), 1024
        self.net = self.net.eval().to(device)
        self.device = device

    def features(self, tiles):
        n = self.net
        x = torch.stack([NATIVE(t) for t in tiles]).to(self.device)
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        l2 = n.layer2(n.layer1(x))
        l3 = n.layer3(l2)
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.avg_pool2d(l3, 3, 1, 1)
        if self.dim:  # WRN-50: average layer-3 channel pairs (1024 -> 512); 512 + 512 = 1024 dims
            l3 = l3.reshape(l3.shape[0], -1, 2, *l3.shape[-2:]).mean(2)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode='bilinear', align_corners=False)
        f = torch.cat([l2, l3], 1)
        b, c, h, w = f.shape
        return f.permute(0, 2, 3, 1).reshape(b, h, w, c)

    def fit(self, archive, refs, per_tile=128, k_total=10240):
        rng = np.random.default_rng(SEED)
        cands = []
        for s in range(0, len(refs), BATCH):
            tiles = [read(archive, 'MVTec/train/good/' + r['patch_name'], 'RGB') for r in refs[s:s + BATCH]]
            f = self.features(tiles)
            b, h, w, d = f.shape
            f = f.reshape(b, h * w, d)
            idx = torch.from_numpy(np.stack([rng.choice(h * w, per_tile, replace=False) for _ in range(b)])).to(self.device)
            cands.append(torch.gather(f, 1, idx[..., None].expand(-1, -1, d)).reshape(-1, d).half().cpu())
            del f
            if s % 160 == 0:
                log('fit features', s, '/', len(refs))
        X = torch.cat(cands)
        del cands
        if hasattr(torch, 'mps') and self.device == 'mps':
            torch.mps.empty_cache()
        k = k_total
        n_cand, dim = len(X), int(X.shape[1])
        g = torch.Generator().manual_seed(SEED)
        proj = torch.randn(dim, 128, generator=g) / np.sqrt(128)
        P = torch.cat([(X[i:i + 32768].float() @ proj) for i in range(0, n_cand, 32768)]).to(self.device)
        start = int(rng.integers(n_cand))
        sel = torch.empty(k, dtype=torch.long, device=self.device)
        sel[0] = start
        mind = ((P - P[start]) ** 2).sum(1)
        for it in range(1, k):
            if it % 2000 == 0:
                log('coreset', it, '/', k)
            i = torch.argmax(mind)
            sel[it] = i
            mind = torch.minimum(mind, ((P - P[i]) ** 2).sum(1))
        sel = sel.cpu()
        assert len(set(sel.tolist())) == k
        self.bank = X[sel].float().to(self.device)
        self.bank_sq = self.bank.square().sum(1)
        del X, P, mind
        return {'candidates': n_cand, 'coreset': k, 'dim': dim,
                'bank_sha256': hashlib.sha256(self.bank.cpu().numpy().tobytes()).hexdigest()}

    def save(self, path):
        torch.save(self.bank.cpu(), path)

    def load(self, path):
        self.bank = torch.load(path).to(self.device)
        self.bank_sq = self.bank.square().sum(1)
        return {'coreset': len(self.bank), 'dim': int(self.bank.shape[1]), 'loaded_from_cache': True,
                'bank_sha256': hashlib.sha256(self.bank.cpu().numpy().tobytes()).hexdigest()}

    def window_maps(self, image, coords):
        out = []
        for s in range(0, len(coords), BATCH):
            crops = [image.crop((x, y, x + SIDE, y + SIDE)) for y, x in coords[s:s + BATCH]]
            f = self.features(crops)
            b, h, w, d = f.shape
            q = f.reshape(-1, d)
            dist = []
            for j in range(0, len(q), 2048):
                qq = q[j:j + 2048]
                dd = qq.square().sum(1, keepdim=True) + self.bank_sq - 2 * (qq @ self.bank.T)
                dist.append(dd.amin(1).clamp_min(0).sqrt())
            sc = torch.cat(dist).reshape(b, h, w)
            out.append(smooth_up(sc, self.device).cpu().numpy())
        return np.concatenate(out)


class Original:
    """The author's own ResNet-18 spatial detector, imported unchanged."""

    def __init__(self, device):
        sys.path.insert(0, str(ORIG / 'code'))
        import localization_study as ls
        self.ls = ls
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
        self.device = device

    def fit(self, archive, refs):
        self.bank, sel = self.ls.fit_bank(archive, refs, self.net, self.device, SEED)
        return {'coreset': 256, 'bank_sha256': hashlib.sha256(self.bank.cpu().numpy().tobytes()).hexdigest()}

    def window_maps(self, image, coords):
        f = self.ls.tile_scores(image, coords, self.net, self.bank, self.device)
        return f.repeat(16, axis=1).repeat(16, axis=2)


def read(archive, member, mode):
    with Image.open(io.BytesIO(archive.read('RAW_FABRID/' + member))) as im:
        return im.convert(mode)


def csv_rows(archive, name):
    with archive.open('RAW_FABRID/RAW_FABRID_' + name + '_Metadata.csv') as s:
        return list(csv.DictReader(io.TextIOWrapper(s, encoding='utf-8-sig')))


def fuse(maps, index, coords):
    field = np.full((1024, 1792), -np.inf, np.float32)
    for y, x in coords:
        np.maximum(field[y:y + SIDE, x:x + SIDE], maps[index[(y, x)]], out=field[y:y + SIDE, x:x + SIDE])
    return field


def run_fold(archive, roll, det, refs, frames, released, cohorts, out_path):
    t0 = time.perf_counter()
    cache = out_path.parent / 'banks' / f'{roll}.pt'
    if hasattr(det, 'load') and cache.exists():
        fit = det.load(cache)
    else:
        fit = det.fit(archive, refs)
        if hasattr(det, 'save'):
            cache.parent.mkdir(exist_ok=True)
            det.save(cache)
    log(roll, 'fit', fit, f'{time.perf_counter() - t0:.0f}s')
    hist = {p: {d: [np.zeros(1, np.int64), np.zeros(1, np.int64)] for d in ('all', 'S')} for p in POLICIES}
    recs = {}
    for n, parent in enumerate(sorted(frames), 1):
        pos = parent in cohorts['defective']
        image = read(archive, 'images/' + parent, 'RGB')
        mask = (np.asarray(read(archive, 'masks/' + parent, 'L')) >= 128) if pos else np.zeros((1024, 1792), bool)
        rel = released[parent]
        coords = list(dict.fromkeys(rel + FIXED + OVERLAP))
        index = {c: i for i, c in enumerate(coords)}
        maps = det.window_maps(image, coords)
        f_rel = fuse(maps, index, rel)
        seen = np.isfinite(f_rel)
        extra = [w for w in FIXED if not seen[w[0]:w[0] + SIDE, w[1]:w[1] + SIDE].all()]
        layouts = {'released': rel, 'fixed': FIXED, 'overlap': OVERLAP,
                   'gapfill': list(dict.fromkeys(rel + extra))}
        fields = {'released': f_rel}
        for p in ('fixed', 'overlap', 'gapfill'):
            fields[p] = fuse(maps, index, layouts[p])
        lab, nreg = label(mask, structure=np.ones((3, 3), bool))
        rec = {'defect': pos, 'foreground': int(mask.sum()), 'seen_pixels': int(seen.sum()),
               'unseen_foreground': int((mask & ~seen).sum()), 'windows': {p: len(set(v)) for p, v in layouts.items()},
               'frame_score': {}, 'region_max': {}}
        in_pixels = parent in cohorts['pixels']
        for p, fld in fields.items():
            rec['frame_score'][p] = float(fld.max())
            if nreg:
                rm = nd_maximum(fld, lab, index=np.arange(1, nreg + 1))
                rec['region_max'][p] = [float(v) if np.isfinite(v) else None for v in np.atleast_1d(rm)]
            if in_pixels:
                b = to_bins(fld)
                add_hist(hist[p]['all'], b.ravel(), mask.ravel())
                add_hist(hist[p]['S'], b[seen], mask[seen])
        recs[parent] = rec
        if n % 10 == 0:
            log(roll, f'{n}/{len(frames)} frames', f'{time.perf_counter() - t0:.0f}s')

    def pack(pair):
        tot, pos = pair
        nz = np.flatnonzero(tot)
        return {'bins': nz.tolist(), 'total': tot[nz].tolist(), 'foreground': pos[nz].tolist()}
    out = {'roll': roll, 'config': args.model, 'device': str(det.device), 'torch': torch.__version__,
           'fit': fit, 'bins_per_log1p': BPL, 'frames': recs,
           'histograms': {p: {d: pack(v) for d, v in hd.items()} for p, hd in hist.items()},
           'seconds': time.perf_counter() - t0}
    tmp = out_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, separators=(',', ':'), allow_nan=False))
    tmp.replace(out_path)
    log(roll, 'done', f'{out["seconds"]:.0f}s')


@torch.inference_mode()
def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    res = ORIG / 'results'
    folds = json.loads((res / 'baseline_seed_13017.json').read_text())['folds']
    loc = json.loads((res / 'localization_study.json').read_text())
    cohorts = {'defective': set(loc['defective_frames']), 'cal': set(loc['calibration_clean']),
               'eval': set(loc['evaluation_clean'])}
    cohorts['pixels'] = cohorts['defective'] | cohorts['eval']
    primary = cohorts['pixels'] | cohorts['cal']
    assert len(cohorts['pixels']) == 255 and len(primary) == 303
    det = Original(device) if args.model == 'orig_r18' else Native(args.model, device)
    out_dir = HERE / 'results' / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = ORIG / 'data' / 'RAW_FABRID.zip'
    with zipfile.ZipFile(zpath) as archive:
        meta = {r['filename']: r for r in csv_rows(archive, 'HighRes')}
        patches = csv_rows(archive, 'Patches')
        released = defaultdict(list)
        for r in json.loads(archive.read('RAW_FABRID/RAW_FABRID_Patches_COCO.json'))['images']:
            released[r['parent']].append((r['r_offset'], r['c_offset']))
        rolls = sorted({r['roll_id'] for r in meta.values()})
        for roll in rolls:
            if args.rolls and roll not in args.rolls:
                continue
            out_path = out_dir / f'{roll}.json'
            if out_path.exists() and not args.overwrite:
                log(roll, 'exists, skipped')
                continue
            fold = folds['held_out_' + roll]
            refs = [patches[i] for i in fold['train_patch_indices']]
            assert len(refs) == 1000 and all(r['type'] == 'train_good' and meta[r['parent_raw']]['roll_id'] != roll
                                             and meta[r['parent_raw']]['has_defect'].lower() == 'false' for r in refs)
            frames = [p for p in primary if meta[p]['roll_id'] == roll]
            if args.limit:
                frames = sorted(frames)[:args.limit]
                out_path = out_dir / f'{roll}_smoke.json'
            for p in frames:
                assert len(released[p]) == 28
                if p not in cohorts['defective']:
                    assert sorted(released[p]) == sorted(FIXED)
            run_fold(archive, roll, det, refs, frames, released, cohorts, out_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=('orig_r18', 'r18_native', 'wrn50'), required=True)
    ap.add_argument('--rolls', nargs='*')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: first N frames only')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    main()
