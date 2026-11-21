"""Protocol 4: label-free two-pass scanning vs fixed grid at matched budgets. See ../PROTOCOL_4.md."""
import argparse, json, sys, time, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strong_patchcore as sp
import coverage_control as cc
from strong_patchcore import FIXED, OVERLAP, SIDE, fuse, to_bins, add_hist, read, csv_rows, log

HERE = Path(__file__).resolve().parents[1]
DRAWS = 5
BUDGETS = (2, 7)
CAND = [w for w in OVERLAP if w not in set(FIXED)]


def layouts(parent, rel, first_pass):
    lay = {'fixed': FIXED, 'overlap91': OVERLAP, 'released': rel}
    seen = np.isfinite(first_pass[1])
    unseen = ~seen
    lay['gapfill'] = list(dict.fromkeys(rel + [w for w in FIXED if unseen[w[0]:w[0] + SIDE, w[1]:w[1] + SIDE].any()]))
    fmap = first_pass[0]
    score = {w: float(fmap[w[0]:w[0] + SIDE, w[1]:w[1] + SIDE].max()) for w in CAND}
    ranked = sorted(CAND, key=lambda w: (-score[w], w))
    for k in BUDGETS:
        lay[f'targeted{k}'] = FIXED + ranked[:k]
        for d in range(DRAWS):
            r = cc.frame_rng(parent, 200 + 10 * k + d)
            lay[f'random{k}_{d}'] = FIXED + [CAND[i] for i in r.choice(len(CAND), k, replace=False)]
    return lay


def run_fold(archive, roll, det, frames, released, cohorts, out_path, model, bank_path, provenance):
    t0 = time.perf_counter()
    fit = det.load(bank_path)
    rec1 = json.loads((HERE / 'results' / model / f'{roll}.json').read_text())['fit']['bank_sha256']
    if fit['bank_sha256'] != rec1:
        raise RuntimeError(f'{model} {roll}: cached bank hash does not match Protocol 1 record')
    hist = defaultdict(lambda: [np.zeros(1, np.int64), np.zeros(1, np.int64)])
    recs = {}
    for n, parent in enumerate(sorted(frames), 1):
        pos = parent in cohorts['defective']
        image = read(archive, 'images/' + parent, 'RGB')
        mask = (np.asarray(read(archive, 'masks/' + parent, 'L')) >= 128) if pos else np.zeros((1024, 1792), bool)
        rel = released[parent]
        coords = list(dict.fromkeys(rel + OVERLAP))
        index = {c: i for i, c in enumerate(coords)}
        maps = det.window_maps(image, coords)
        lay = layouts(parent, rel, (fuse(maps, index, FIXED), fuse(maps, index, rel)))
        rec = {'defect': pos, 'frame_score': {}, 'targeted_extras': {k: lay[f'targeted{k}'][28:] for k in BUDGETS}}
        in_pixels = parent in cohorts['pixels']
        for p, cp in lay.items():
            fld = fuse(maps, index, cp)
            rec['frame_score'][p] = float(fld.max())
            if in_pixels:
                add_hist(hist[p], to_bins(fld).ravel(), mask.ravel())
        recs[parent] = rec
        if n % 10 == 0:
            log(roll, f'{n}/{len(frames)} frames', f'{time.perf_counter() - t0:.0f}s')

    def pack(pair):
        tot, fg = pair
        nz = np.flatnonzero(tot)
        return {'bins': nz.tolist(), 'total': tot[nz].tolist(), 'foreground': fg[nz].tolist()}
    out = {'protocol': 'PROTOCOL_4.md', 'protocol_sha256': cc.sha(HERE / 'PROTOCOL_4.md'), 'roll': roll, 'model': model,
           'provenance': provenance, 'fit': fit, 'draws': DRAWS, 'budgets': BUDGETS, 'frames': recs,
           'histograms': {p: pack(v) for p, v in hist.items()}, 'seconds': time.perf_counter() - t0}
    tmp = out_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, separators=(',', ':'), allow_nan=False))
    tmp.replace(out_path)
    log(roll, 'done', f'{out["seconds"]:.0f}s')


@torch.inference_mode()
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    sp.args = args
    loc = json.loads((sp.ORIG / 'results' / 'localization_study.json').read_text())
    cohorts = {'defective': set(loc['defective_frames']), 'cal': set(loc['calibration_clean']),
               'eval': set(loc['evaluation_clean'])}
    cohorts['pixels'] = cohorts['defective'] | cohorts['eval']
    primary = cohorts['pixels'] | cohorts['cal']
    det = sp.Native(args.model, device)
    hub = Path(torch.hub.get_dir()) / 'checkpoints'
    provenance = {'device': device, 'torch': torch.__version__,
                  'code_sha256': {f: cc.sha(HERE / 'code' / f) for f in ('labelfree_scan.py', 'coverage_control.py', 'strong_patchcore.py')},
                  'weights_sha256': {p.name: cc.sha(p) for p in sorted(hub.glob('*.pth'))
                                     if ('resnet18' in p.name) == (args.model != 'wrn50')}}
    out_dir = HERE / 'results_p4' / args.model
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
                frames = frames[:args.limit]
            run_fold(archive, roll, det, frames, released, cohorts, out_path, args.model,
                     HERE / 'results' / args.model / 'banks' / f'{roll}.pt', provenance)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=('r18_native', 'wrn50'), required=True)
    ap.add_argument('--rolls', nargs='*')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    main()
