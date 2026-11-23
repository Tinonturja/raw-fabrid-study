"""Protocol 5 three-seed sensitivity check on AITEX (Amendment 1, A5): how much do the primary contrasts move
when the WRN-50 bank is re-sampled with a different seed? Seed 13017 is the frozen main run (its saved .npz is
reused). Seeds 13027 and 13037 rebuild the bank (candidate draw, projection and coreset start all follow
p5_models.SEED) and rerun the frozen p5_run.process on the same cohort. Bootstrap seed stays 13017.
Usage: python p5_aitex_seeds.py --data <aitex dir> | --analyze"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_core as C
import p5_models as M
import p5_run as R

SEEDS = (13017, 13027, 13037)
OUT = R.HERE / 'results_p5' / 'aitex_seeds'
MAIN = R.HERE / 'results_p5' / 'aitex' / 'full' / 'aitex__dino+wrn.npz'


@torch.inference_mode()
def run(data):
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    man = json.loads((R.HERE / 'manifests' / 'aitex_manifest.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS[1:]:
        out = OUT / f'seed{seed}.npz'
        if out.exists():
            continue
        M.SEED = seed
        torch.manual_seed(seed)
        fams = [M.WRN(device)]
        for unit, tiles, items in R.aitex_units(Path(data), man, 0):
            t0 = time.time()
            scorers, info = R.build_scorers(fams, tiles, OUT / f'banks_seed{seed}', unit, device)
            hists, fails = [], []
            for n, it in enumerate(items, 1):
                _, h = R.process(it, fams, scorers, fails)
                hists.append(h)
                if n % 20 == 0:
                    R.log('seed', seed, f'{n}/{len(items)}', f'{time.time() - t0:.0f}s')
            R.pack(hists, out)
            (OUT / f'seed{seed}.json').write_text(json.dumps({'seed': seed, 'banks': info, 'r1_failures': fails,
                                                               'seconds': time.time() - t0}, indent=1))
            R.log('seed', seed, 'done', f'R1 failures {len(fails)}')


def load(npz, sc, lay):
    z = np.load(npz)
    off = z[f'{sc}|{lay}|off']
    hs = [(z[f'{sc}|{lay}|bins'][off[i]:off[i + 1]], z[f'{sc}|{lay}|tot'][off[i]:off[i + 1]],
           z[f'{sc}|{lay}|pos'][off[i]:off[i + 1]]) for i in range(len(off) - 1)]
    return hs


def analyze():
    files = {13017: MAIN, **{s: OUT / f'seed{s}.npz' for s in SEEDS[1:]}}
    res = {}
    for sc in ('patchcore', 'gauss'):
        for s, f in files.items():
            h1, h2, h4 = load(f, sc, 'R1'), load(f, sc, 'R2'), load(f, sc, 'C4')
            nb = max(int(b.max()) for hs in (h1, h2, h4) for b, _, _ in hs if len(b)) + 1
            (T1, P1), (T2, P2), (T4, P4) = C.dense(h1, nb), C.dense(h2, nb), C.dense(h4, nb)
            Cw = C.bootstrap_counts(len(T1))
            ap = lambda T, P: float(C.ap_from_hist(T.sum(0), P.sum(0)))
            d = C.boot_ap_diff(T2, P2, T1, P1, Cw)
            lo, hi = C.percentile_ci(d)
            res[f'{sc}|seed{s}'] = {'R1': ap(T1, P1), 'R2': ap(T2, P2), 'C4': ap(T4, P4),
                                   'R2-R1': ap(T2, P2) - ap(T1, P1), 'R2-R1_ci95': [lo, hi],
                                   'C4-R1': ap(T4, P4) - ap(T1, P1)}
            print(sc, s, json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v])
                                     for k, v in res[f'{sc}|seed{s}'].items()}))
    for sc in ('patchcore', 'gauss'):
        v = [res[f'{sc}|seed{s}']['R2-R1'] for s in SEEDS]
        a = [res[f'{sc}|seed{s}']['R1'] for s in SEEDS]
        res[f'{sc}|spread'] = {'R2-R1_range': [min(v), max(v)], 'R1_AP_range': [min(a), max(a)]}
        print(sc, 'spread', res[f'{sc}|spread'])
    (OUT / 'analysis.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data')
    ap.add_argument('--analyze', action='store_true')
    a = ap.parse_args()
    analyze() if a.analyze else run(a.data)
