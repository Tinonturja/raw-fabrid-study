"""Protocol 3: gap-fill versus proximity-matched extra looks (C4). See ../PROTOCOL_3.md."""
import argparse, json, sys, time, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strong_patchcore as sp
import coverage_control as cc
from strong_patchcore import FIXED, OVERLAP, SIDE, SEED, fuse, to_bins, add_hist, read, csv_rows, log

HERE = Path(__file__).resolve().parents[1]
DRAWS = 5


def seen_fg(win, seen_mask):
    y, x = win
    return int(seen_mask[y:y + SIDE, x:x + SIDE].sum())


def layouts_c4(parent, rel, seen, mask):
    unseen = ~seen
    extras = [w for w in FIXED if unseen[w[0]:w[0] + SIDE, w[1]:w[1] + SIDE].any()]
    gap = list(dict.fromkeys(rel + extras))
    rel_set = set(rel)
    extras = [w for w in extras if w not in rel_set]
    k = len(extras)
    sm = mask & seen
    cand = [w for w in OVERLAP if w not in rel_set and not unseen[w[0]:w[0] + SIDE, w[1]:w[1] + SIDE].any()]
    cf = {w: seen_fg(w, sm) for w in cand}
    ef = sorted((seen_fg(w, sm) for w in extras), reverse=True)
    lay = {'released': rel, 'gapfill': gap}
    info = {'k': k, 'gap_seen_fg': int(sum(ef)), 'c4_seen_fg': []}
    for d in range(DRAWS):
        r = cc.frame_rng(parent, 100 + d)
        pool = list(cand)
        r.shuffle(pool)
        chosen = []
        for target in ef:
            if not pool:
                break
            best = min(range(len(pool)), key=lambda i: abs(cf[pool[i]] - target))
            chosen.append(pool.pop(best))
        lay[f'C4_{d}'] = list(dict.fromkeys(rel + chosen))
        info['c4_seen_fg'].append(int(sum(cf[w] for w in chosen)))
    return lay, info


def run_fold(archive, roll, det, frames, released, cohorts, out_path, model, bank_path, provenance):
    t0 = time.perf_counter()
    fit = det.load(bank_path)
    rec1 = json.loads((HERE / 'results' / model / f'{roll}.json').read_text())['fit']['bank_sha256']
    if fit['bank_sha256'] != rec1:
        raise RuntimeError(f'{model} {roll}: cached bank hash does not match Protocol 1 record')
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
        lay, info = layouts_c4(parent, rel, seen, mask)
        rec = {'defect': pos, **info, 'frame_score': {}}
        in_pixels = parent in cohorts['pixels']
        for p, cp in lay.items():
            fld = fuse(maps, index, cp)
            rec['frame_score'][p] = float(fld.max())
            if in_pixels:
                b = to_bins(fld)
                add_hist(hist[p]['all'], b.ravel(), mask.ravel())
                add_hist(hist[p]['S'], b[seen], mask[seen])
        recs[parent] = rec
        if n % 10 == 0:
            log(roll, f'{n}/{len(frames)} frames', f'{time.perf_counter() - t0:.0f}s')

    def pack(pair):
        tot, fg = pair
        nz = np.flatnonzero(tot)
        return {'bins': nz.tolist(), 'total': tot[nz].tolist(), 'foreground': fg[nz].tolist()}
    out = {'protocol': 'PROTOCOL_3.md', 'protocol_sha256': cc.sha(HERE / 'PROTOCOL_3.md'), 'roll': roll, 'model': model,
           'provenance': provenance, 'fit': fit, 'draws': DRAWS, 'frames': recs,
           'histograms': {p: {d: pack(v) for d, v in hd.items()} for p, hd in hist.items()},
           'seconds': time.perf_counter() - t0}
    tmp = out_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, separators=(',', ':'), allow_nan=False))
    tmp.replace(out_path)
    log(roll, 'done', f'{out["seconds"]:.0f}s')


@torch.inference_mode()
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    sp.args = args
    res = sp.ORIG / 'results'
    loc = json.loads((res / 'localization_study.json').read_text())
    cohorts = {'defective': set(loc['defective_frames']), 'cal': set(loc['calibration_clean']),
               'eval': set(loc['evaluation_clean'])}
    cohorts['pixels'] = cohorts['defective'] | cohorts['eval']
    primary = cohorts['pixels'] | cohorts['cal']
    det = sp.Native(args.model, device)
    hub = Path(torch.hub.get_dir()) / 'checkpoints'
    provenance = {'device': device, 'torch': torch.__version__,
                  'code_sha256': {f: cc.sha(HERE / 'code' / f) for f in ('coverage_control_c4.py', 'coverage_control.py', 'strong_patchcore.py')},
                  'weights_sha256': {p.name: cc.sha(p) for p in sorted(hub.glob('*.pth'))
                                     if ('resnet18' in p.name) == (args.model != 'wrn50')}}
    out_dir = HERE / 'results_p3' / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(sp.ORIG / 'data' / 'RAW_FABRID.zip') as archive:
        meta = {r['filename']: r for r in csv_rows(archive, 'HighRes')}
        released = defaultdict(list)
        for r in json.loads(archive.read('RAW_FABRID/RAW_FABRID_Patches_COCO.json'))['images']:
            released[r['parent']].append((r['r_offset'], r['c_offset']))
        for roll in sorted({r['roll_id'] for r in meta.values()}):
            if args.rolls and roll not in args.rolls:
                continue
            out_path = out_dir / (f'{roll}_smoke.json' if args.limit else f'{roll}.json')
            if out_path.exists() and not args.limit:
                log(roll, 'exists, skipped'); continue
            frames = sorted(p for p in primary if meta[p]['roll_id'] == roll)
            if args.limit:
                frames = [p for p in frames if p in cohorts['defective']][:args.limit]
            run_fold(archive, roll, det, frames, released, cohorts, out_path, args.model,
                     HERE / 'results' / args.model / 'banks' / f'{roll}.pt', provenance)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=('r18_native', 'wrn50'), required=True)
    ap.add_argument('--rolls', nargs='*')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    main()
