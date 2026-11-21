"""Protocol 5 on TSFabrics (Kaggle). Requires p5_core.py and p5_models.py next to this file (or concatenated in front).

Design (recorded in PROTOCOL_5_RUNLOG.md before any TSFabrics score):
- Test units: the 7 positive clusters from the frozen gate (C1, C2, C4, C5, C6, C7, C9).
- Per test cluster: all frames with a defect mask (capped at 300, uniform sample, seed 13017), an equal number of
  label-1 normal frames, and up to 50 label-2 cutline frames (normal for the defect metric; cutline mask kept).
- Decoder: defect pixel = value > 0 in any defect mask of that frame (equivalent to >=128 on 0/255 masks).
- Bank, leave-cluster-out: label-1 frames of every OTHER cluster with the same texture (T1 or T2), edge-aligned 256 grid
  tiles, 1,000 tiles drawn with seed 13018, then the standard candidate sampling and 10,240 coreset.
- Layouts: R1 native frame, R2 edge-aligned grid (800 x 640 -> 12 overlapping windows), R2occ, C4. No practice A.
- Outputs per cluster in /kaggle/working/results_p5/tsfabrics/full/<cluster>__<families>.{npz,json}.
"""
import csv, glob, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if 'C' not in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import p5_core as C
    import p5_models as M

CLUSTERS = {  # scenario -> (texture, cluster), from the frozen pilot table
    'T1_S148_I108_1': ('T1', 'C1'), 'T1_S148_I108_2': ('T1', 'C1'), 'T1_S164_I192_1': ('T1', 'C2'),
    'T1_S164_I193_1': ('T1', 'C2'), 'T1_S164_I199_1': ('T1', 'C2'), 'T1_S164_I36_1': ('T1', 'C3'),
    'T1_S179_I35_1': ('T1', 'C3'), 'T1_S174_I108_1': ('T1', 'C4'), 'T1_S174_I111_1': ('T1', 'C5'),
    'T1_S177_I108_1': ('T1', 'C5'), 'T1_S478_I118_1': ('T1', 'C6'), 'T1_S478_I118_2': ('T1', 'C6'),
    'T1_S555_I117_1': ('T1', 'C7'), 'T2_S262_I45_1': ('T2', 'C8'), 'T2_S262_I52_1': ('T2', 'C8'),
    'T2_S303_I44_1': ('T2', 'C8'), 'T2_S287_I184_1': ('T2', 'C9'), 'T2_S287_I185_1': ('T2', 'C9'),
    'T2_S466_I39_1': ('T2', 'C10'), 'T3_S165_I175_1': ('T3', 'C11'), 'T3_S165_I178_1': ('T3', 'C11'),
    'T3_S165_I178_2': ('T3', 'C11')}
TEST = ['C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C9']
CAP_DEFECT, CAP_CUT = 300, 50


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def find_root():
    for base, dirs, _ in os.walk('/kaggle/input'):
        if 'CSV' in dirs and 'IMAGE' in dirs:
            return Path(base)
    sys.exit('TSFabrics root not found')


def index_scenario(root, s):
    with open(root / 'CSV' / f'{s}.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    labels = {int(re.search(r'(\d+)', r['FileName']).group(1)): int(r['Label']) for r in rows}
    imgs = {}
    for p in glob.glob(str(root / 'IMAGE' / s / '**' / '*.*'), recursive=True):
        m = re.fullmatch(r'(\d+)', Path(p).stem)
        if m:
            imgs[int(m.group(1))] = p
    masks = defaultdict(lambda: defaultdict(list))
    adir = root / 'ANNOTATION' / s
    if adir.is_dir():
        for t in os.listdir(adir):
            for p in glob.glob(str(adir / t / '**' / '*.*'), recursive=True):
                m = re.match(r'(\d+)', Path(p).stem)
                if m:
                    masks[t.lower()][int(m.group(1))].append(p)
    return labels, imgs, masks


def gray_mask(paths, shape):
    out = np.zeros(shape, bool)
    for p in paths:
        a = np.asarray(Image.open(p).convert('L'))
        assert a.shape == shape, (p, a.shape)
        out |= a > 0
    return out


def rgb(p):
    with Image.open(p) as im:
        return im.convert('RGB')


def main():
    root = find_root()
    out_dir = Path('/kaggle/working/results_p5/tsfabrics/full')
    out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(M.SEED)
    log('root', root, 'device', device)
    idx = {s: index_scenario(root, s) for s in CLUSTERS}
    by_cluster = defaultdict(list)
    for s, (t, c) in CLUSTERS.items():
        by_cluster[c].append(s)
    rng = np.random.default_rng(M.SEED)
    fams = [M.WRN(device), M.DINO(device)]
    tag = 'dino+wrn'
    for c in TEST:
        if (out_dir / f'{c}__{tag}.json').exists():
            log(c, 'done earlier'); continue
        t0 = time.time()
        tex = CLUSTERS[by_cluster[c][0]][0]
        # bank: label-1 frames of other same-texture clusters
        ref = [(s, i) for s, (t, cc) in CLUSTERS.items() if t == tex and cc != c
               for i, lab in sorted(idx[s][0].items()) if lab == 1 and i in idx[s][1]]
        pick = np.random.default_rng(M.SEED + 1).choice(len(ref), min(len(ref), 1000), replace=False)
        grid = C.grid_layout(640, 800)
        g2 = np.random.default_rng(M.SEED + 2)
        tiles = []
        for k in sorted(pick):
            s, i = ref[k]
            y, x = grid[int(g2.integers(len(grid)))]
            tiles.append((lambda p=idx[s][1][i], y=y, x=x: rgb(p).crop((x, y, x + 256, y + 256))))
        scorers, bank_info = {}, {}
        for fam in fams:
            X, info = M.sample_candidates(fam, tiles, log=lambda *a: log(c, *a))
            bank = M.greedy_coreset(X, 10240, 'cpu', log=lambda *a: log(c, fam.name, *a))
            if fam.name == 'wrn50':
                mu, prec = M.SharedGauss.fit(X)
                scorers[fam.name] = [M.PatchCore(bank, device), M.SharedGauss(mu, prec, len(X), device)]
            else:
                scorers[fam.name] = [M.DinoKNN(bank, device)]
            bank_info[fam.name] = dict(info, bank_sha256=M.digest(bank), bank_scenarios=sorted({s for s, _ in ref}))
            del X
        # test frames
        items = []
        for s in by_cluster[c]:
            labels, imgs, masks = idx[s]
            dfr = sorted(i for i in masks.get('defect', {}) if i in imgs)
            nfr = sorted(i for i, lab in labels.items() if lab == 1 and i in imgs and i not in masks.get('defect', {}))
            cfr = sorted(i for i, lab in labels.items() if lab == 2 and i in imgs and i not in masks.get('defect', {}))
            items += [(s, i, 'defect') for i in dfr]
            items += [(s, i, 'normal') for i in nfr]
            items += [(s, i, 'cutline') for i in cfr]
        d = [x for x in items if x[2] == 'defect']
        n = [x for x in items if x[2] == 'normal']
        cu = [x for x in items if x[2] == 'cutline']
        d = [d[k] for k in sorted(rng.choice(len(d), min(len(d), CAP_DEFECT), replace=False))]
        n = [n[k] for k in sorted(rng.choice(len(n), min(len(n), len(d)), replace=False))]
        cu = [cu[k] for k in sorted(rng.choice(len(cu), min(len(cu), CAP_CUT), replace=False))] if cu else []
        sel = d + n + cu
        log(c, 'test frames', len(d), 'defect', len(n), 'normal', len(cu), 'cutline')
        import p5_run as R  # reuse process() and pack()
        recs, hists, failures = [], [], []
        for k, (s, i, kind) in enumerate(sel, 1):
            p = idx[s][1][i]
            item = {'id': f'{s}/{i}', 'group': s, 'defect': kind == 'defect', 'image': (lambda p=p: rgb(p)),
                    'mask': (lambda s=s, i=i: gray_mask(idx[s][2]['defect'][i], (640, 800))) if kind == 'defect' else None,
                    'A': False}
            r, h = R.process(item, fams, scorers, failures)
            r['kind'] = kind
            if kind == 'cutline' and i in idx[s][2].get('cutline', {}):
                r['cutline_px'] = int(gray_mask(idx[s][2]['cutline'][i], (640, 800)).sum())
            recs.append(r); hists.append(h)
            if k % 50 == 0 or k == len(sel):
                log(c, f'{k}/{len(sel)} frames', f'{time.time() - t0:.0f}s')
        R.pack(hists, out_dir / f'{c}__{tag}.npz')
        out = {'unit': c, 'dataset': 'tsfabrics', 'device': device, 'torch': torch.__version__, 'families': ['wrn', 'dino'],
               'banks': bank_info, 'bins_per_log1p': C.BPL, 'images': recs, 'r1_failures': failures,
               'seconds': time.time() - t0}
        (out_dir / f'{c}__{tag}.json').write_text(json.dumps(out, separators=(',', ':')))
        log(c, 'unit done', f'{time.time() - t0:.0f}s', 'R1 failures', len(failures))
    log('ALL UNITS DONE')


if __name__ == '__main__':
    main()
