"""Protocol 2: gap-fill versus budget-matched extra-look controls (see ../PROTOCOL_2.md)."""
import argparse, hashlib, json, sys, time, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strong_patchcore as sp
from strong_patchcore import FIXED, OVERLAP, SIDE, SEED, fuse, to_bins, add_hist, read, csv_rows, log

HERE = Path(__file__).resolve().parents[1]
DRAWS = 5
FIXED_SET = set(FIXED)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def frame_rng(parent, draw):
    return np.random.default_rng(int(hashlib.sha256(f'{parent}|{draw}'.encode()).hexdigest()[:12], 16))


def windows_touching(mask_bool, windows):
    return [w for w in windows if mask_bool[w[0]:w[0] + SIDE, w[1]:w[1] + SIDE].any()]


def overlaps(a, b):
    return abs(a[0] - b[0]) < SIDE and abs(a[1] - b[1]) < SIDE


def layouts_for(parent, rel, seen):
    unseen = ~seen
    extra = windows_touching(unseen, FIXED)
    gap = list(dict.fromkeys(rel + extra))
    k = len(gap) - len(set(rel))
    rel_set = set(rel)
    moved = [w for w in rel if w not in FIXED_SET]
    clean_ov = [w for w in OVERLAP if w not in rel_set and not unseen[w[0]:w[0] + SIDE, w[1]:w[1] + SIDE].any()]
    near = [w for w in clean_ov if any(overlaps(w, m) for m in moved)]
    lay = {'released': rel, 'fixed': FIXED, 'gapfill': gap}
    info = {'k': k, 'near_candidates': len(near), 'near_shortfall': 0}
    for d in range(DRAWS):
        r = frame_rng(parent, d)
        if k == 0:
            lay[f'C1_{d}'] = rel; lay[f'C2_{d}'] = rel; lay[f'C3_{d}'] = FIXED
            continue
        pick = lambda pool, n: [pool[i] for i in r.choice(len(pool), min(n, len(pool)), replace=False)] if pool else []
        c1 = pick(near, k)
        info['near_shortfall'] = max(info['near_shortfall'], k - len(c1))
        lay[f'C1_{d}'] = list(dict.fromkeys(rel + c1))
        lay[f'C2_{d}'] = list(dict.fromkeys(rel + pick(clean_ov, k)))
        lay[f'C3_{d}'] = list(dict.fromkeys(FIXED + pick([w for w in OVERLAP if w not in FIXED_SET], k)))
    return lay, info


def run_fold(archive, roll, det, refs, frames, released, cohorts, out_path, model, bank_path, provenance):
    t0 = time.perf_counter()
    if bank_path is not None:
        fit = det.load(bank_path)
        rec1 = json.loads((HERE / 'results' / model / f'{roll}.json').read_text())['fit']['bank_sha256']
        if fit['bank_sha256'] != rec1:
            raise RuntimeError(f'{model} {roll}: cached bank hash does not match Protocol 1 record')
        fit['bank_file_sha256'] = sha(bank_path)
    else:
        fit = det.fit(archive, refs)
    log(roll, 'bank ready', f'{time.perf_counter() - t0:.0f}s')
    hist = defaultdict(lambda: {d: [np.zeros(1, np.int64), np.zeros(1, np.int64)] for d in ('all', 'S')})
    recs = {}
    for n, parent in enumerate(sorted(frames), 1):
        pos = parent in cohorts['defective']
        image = read(archive, 'images/' + parent, 'RGB')
        mask = (np.asarray(read(archive, 'masks/' + parent, 'L')) >= 128) if pos else np.zeros((1024, 1792), bool)
        rel = released[parent]
        coords = list(dict.fromkeys(rel + FIXED + OVERLAP))
        index = {c: i for i, c in enumerate(coords)}
        maps = det.window_maps(image, coords)
        seen = np.isfinite(fuse(maps, index, rel))
        lay, info = layouts_for(parent, rel, seen)
        rec = {'defect': pos, 'k': info['k'], 'near_candidates': info['near_candidates'],
               'near_shortfall': info['near_shortfall'], 'frame_score': {}}
        in_pixels = parent in cohorts['pixels']
        cache = {}
        for p, coords_p in lay.items():
            key = tuple(sorted(set(coords_p)))
            if key not in cache:
                fld = fuse(maps, index, coords_p)
                cache[key] = (float(fld.max()), to_bins(fld) if in_pixels else None)
            score, b = cache[key]
            rec['frame_score'][p] = score
            if in_pixels:
                add_hist(hist[p]['all'], b.ravel(), mask.ravel())
                add_hist(hist[p]['S'], b[seen], mask[seen])
        recs[parent] = rec
        if n % 10 == 0:
            log(roll, f'{n}/{len(frames)} frames', f'{time.perf_counter() - t0:.0f}s')

    def pack(pair):
        tot, fg = pair
        nz = np.flatnonzero(tot)
        return {'bins': nz.tolist(), 'total': tot[nz].tolist(), 'foreground': fg[nz].tolist()}
    out = {'protocol': 'PROTOCOL_2.md', 'protocol_sha256': sha(HERE / 'PROTOCOL_2.md'), 'roll': roll, 'model': model,
           'provenance': provenance, 'fit': fit, 'draws': DRAWS, 'bins_per_log1p': sp.BPL, 'frames': recs,
           'histograms': {p: {d: pack(v) for d, v in hd.items()} for p, hd in hist.items()},
           'seconds': time.perf_counter() - t0}
    tmp = out_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, separators=(',', ':'), allow_nan=False))
    tmp.replace(out_path)
    log(roll, 'done', f'{out["seconds"]:.0f}s')


@torch.inference_mode()
def main():
    torch.manual_seed(SEED)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    sp.args = args
    res = sp.ORIG / 'results'
    folds = json.loads((res / 'baseline_seed_13017.json').read_text())['folds']
    loc = json.loads((res / 'localization_study.json').read_text())
    cohorts = {'defective': set(loc['defective_frames']), 'cal': set(loc['calibration_clean']),
               'eval': set(loc['evaluation_clean'])}
    cohorts['pixels'] = cohorts['defective'] | cohorts['eval']
    primary = cohorts['pixels'] | cohorts['cal']
    det = sp.Original(device) if args.model == 'orig_r18' else sp.Native(args.model, device)
    hub = Path(torch.hub.get_dir()) / 'checkpoints'
    provenance = {'device': device, 'torch': torch.__version__,
                  'code_sha256': {f: sha(HERE / 'code' / f) for f in ('coverage_control.py', 'strong_patchcore.py')},
                  'weights_sha256': {p.name: sha(p) for p in sorted(hub.glob('*.pth'))
                                     if ('resnet18' in p.name) == (args.model != 'wrn50')}}
    out_dir = HERE / 'results_p2' / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(sp.ORIG / 'data' / 'RAW_FABRID.zip') as archive:
        meta = {r['filename']: r for r in csv_rows(archive, 'HighRes')}
        patches = csv_rows(archive, 'Patches')
        released = defaultdict(list)
        for r in json.loads(archive.read('RAW_FABRID/RAW_FABRID_Patches_COCO.json'))['images']:
            released[r['parent']].append((r['r_offset'], r['c_offset']))
        for roll in sorted({r['roll_id'] for r in meta.values()}):
            if args.rolls and roll not in args.rolls:
                continue
            out_path = out_dir / (f'{roll}_smoke.json' if args.limit else f'{roll}.json')
            if out_path.exists() and not args.limit:
                log(roll, 'exists, skipped'); continue
            refs = [patches[i] for i in folds['held_out_' + roll]['train_patch_indices']]
            frames = sorted(p for p in primary if meta[p]['roll_id'] == roll)
            if args.limit:
                frames = [p for p in frames if p in cohorts['defective']][:args.limit]
            bank = None if args.model == 'orig_r18' else HERE / 'results' / args.model / 'banks' / f'{roll}.pt'
            run_fold(archive, roll, det, refs, frames, released, cohorts, out_path, args.model, bank, provenance)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=('orig_r18', 'r18_native', 'wrn50'), required=True)
    ap.add_argument('--rolls', nargs='*')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    main()
