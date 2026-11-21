"""Protocol 5 pipeline tests (Amendment 1, A7). Run: python tests_p5.py [--aitex <dir>] [--torch]

NumPy tests always run. --aitex re-checks the AITEX cohort numbers on real masks.
--torch runs the map-restoration and alignment tests (needs PyTorch; DINOv2 weights download on first use).
"""
import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_core as C

HERE = Path(__file__).resolve().parents[1]
RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{detail}]' if detail else ''), flush=True)


def exact_ap(scores, labels):
    o = np.argsort(-scores, kind='mergesort')
    y = labels[o].astype(float)
    cp = np.cumsum(y)
    return float((y * cp / np.arange(1, len(y) + 1)).sum() / y.sum())


def numpy_tests():
    rng = np.random.default_rng(0)
    # 1. edge-aligned grids cover every pixel, stay inside, overlap only where sizes do not divide by 256
    for h, w in [(256, 4096), (1024, 1792), (640, 800), (800, 640), (1000, 1500), (256, 300), (300, 256)]:
        g = C.grid_layout(h, w)
        cov = C.coverage(h, w, g)
        inside = all(0 <= y <= h - 256 and 0 <= x <= w - 256 for y, x in g)
        exact = (h % 256 == 0 and w % 256 == 0)
        check(f'grid {h}x{w} covers all pixels', cov.min() >= 1 and inside, f'{len(g)} windows, max mult {cov.max()}')
        check(f'grid {h}x{w} overlap only if needed', (cov.max() == 1) == exact)
    check('RAW-FABRID grid = 28 fixed windows', len(C.grid_layout(1024, 1792)) == 28)
    check('AITEX grid = 16 windows', len(C.grid_layout(256, 4096)) == 16)

    # 2. AP from stitched non-overlapping tiles equals AP over concatenated tiles (order invariance)
    h, w = 512, 768
    field = rng.gamma(2, 1, (h, w)).astype(np.float32)
    mask = np.zeros((h, w), bool); mask[100:140, 300:420] = True; mask[400:410, 10:30] = True
    field[mask] += 1.5
    g = C.grid_layout(h, w)
    tiles = [field[y:y + 256, x:x + 256] for y, x in g]
    stitched = C.fuse_max(h, w, tiles, g)
    hs = C.hist(C.to_bins(stitched), mask)
    ht = C.hist(C.to_bins(np.stack(tiles)), np.stack([mask[y:y + 256, x:x + 256] for y, x in g]))
    T1, P1 = C.dense([hs]); T2, P2 = C.dense([ht], T1.shape[1])
    check('stitched AP == concatenated-tile AP (no overlap)', C.ap_from_hist(T1[0], P1[0]) == C.ap_from_hist(T2[0], P2[0]))

    # 3. duplicated pixel occurrences change AP (review's counterexample)
    s = np.array([3., 2., 1.]); y = np.array([1, 0, 1], bool)
    a = C.dense([C.hist(C.to_bins(s), y)]); b = C.dense([C.hist(C.to_bins(np.r_[s, 2.]), np.r_[y, False])])
    ua, ub = C.ap_from_hist(a[0][0], a[1][0]), C.ap_from_hist(b[0][0], b[1][0])
    check('unique-pixel AP 0.8333 vs duplicated 0.75', abs(ua - 5 / 6) < 1e-9 and abs(ub - 0.75) < 1e-9, f'{ua:.6f}, {ub:.6f}')

    # 4. binned AP equals exact AP when no two scores share a bin
    sc = np.sort(rng.choice(np.arange(1, 4000), 600, replace=False)) / 500.0
    lab = rng.random(600) < 0.1
    T, P = C.dense([C.hist(C.to_bins(sc), lab)])
    check('binned AP == exact AP without ties', abs(C.ap_from_hist(T[0], P[0]) - exact_ap(sc, lab)) < 1e-12)

    # 5. never-scored pixels rank last
    f = np.array([2., -np.inf, 1.]); m = np.array([False, True, False])
    T, P = C.dense([C.hist(C.to_bins(f), m)])
    check('unseen defect pixel ranked last', abs(C.ap_from_hist(T[0], P[0]) - 1 / 3) < 1e-12)

    # 6. AUROC, sign-flip, bootstrap
    check('AUROC 3 of 4 defective-clean pairs = 0.75', abs(C.auroc([.9, .2, .3, .4], [1, 0, 1, 0]) - 0.75) < 1e-12)
    check('AUROC example', abs(C.auroc([5, 1, 4, 2], [1, 0, 1, 1]) - 1.0) < 1e-12 and abs(C.auroc([1, 2, 3, 4], [1, 0, 1, 0]) - 0.25) < 1e-12)
    check('sign-flip p, 7 of 7 positive = 2/128', abs(C.signflip_p([.01, .02, .03, .01, .02, .01, .04]) - 2 / 128) < 1e-12)
    hists = [C.hist(C.to_bins(rng.gamma(2, 1, 1000)), rng.random(1000) < .05) for _ in range(20)]
    T, P = C.dense(hists)
    W = C.bootstrap_counts(20, reps=200)
    d = C.boot_ap_diff(T, P, T, P, W)
    check('bootstrap AP(a) - AP(a) is exactly 0', np.allclose(d[np.isfinite(d)], 0))
    check('bootstrap weights sum to n per draw', np.all(W.sum(1) == 20))
    Ws = C.bootstrap_counts(10, reps=50, strata=[0] * 4 + [1] * 6)
    check('stratified bootstrap keeps stratum sizes', np.all(Ws[:, :4].sum(1) == 4) and np.all(Ws[:, 4:].sum(1) == 6))
    Wc = C.cluster_counts(['a', 'a', 'b', 'c', 'c', 'c'], reps=50)
    check('cluster bootstrap gives one weight per cluster', np.all(Wc[:, 0] == Wc[:, 1]) and np.all(Wc[:, 3] == Wc[:, 5])
          and np.all(Wc[:, [0, 2, 3]].sum(1) == 3))
    rb = C.roll_bootstrap([.01] * 7, reps=100)
    check('roll bootstrap of constant diffs is constant', np.allclose(rb, .01))

    # 7. VisA decoder: category IDs > 0; values >= 128 rejected
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / 'm.png'
        Image.fromarray(np.array([[0, 3], [8, 0]], np.uint8)).save(p)
        check('VisA decoder: IDs > 0 are defects', C.decode_visa(p).sum() == 2)
        Image.fromarray(np.array([[0, 255], [0, 0]], np.uint8)).save(p)
        try:
            C.decode_visa(p); ok = False
        except AssertionError:
            ok = True
        check('VisA decoder rejects 0/255 masks', ok)

    # 8. practice A on a synthetic AITEX strip creates the expected gap
    m = np.zeros((256, 4096), bool); m[100:110, 200:230] = True; m[50:60, 300:310] = True
    lay = C.aitex_original_layout(m)
    cov = C.coverage(256, 4096, lay)
    check('practice A moves a window and can leave unseen pixels', lay[0] != (0, 0) or lay[1] != (0, 256))
    check('practice A keeps 16 windows of 256', len(lay) == 16 and cov.max() >= 1)


def aitex_tests(data):
    man = json.loads((HERE / 'manifests' / 'aitex_manifest.json').read_text())
    fg = unseen = 0
    for n, r in man['images'].items():
        if r['role'] != 'evaluation_defective':
            continue
        m = C.decode_aitex([data / x['path'] for x in r['masks']])
        cov = C.coverage(256, 4096, C.aitex_original_layout(m))
        fg += int(m.sum()); unseen += int((m & (cov == 0)).sum())
    check('AITEX real masks: 485,027 defect pixels', fg == 485027, str(fg))
    check('AITEX real masks: 150 unseen under practice A', unseen == 150, str(unseen))


def torch_tests():
    import torch
    import p5_models as M
    # restoration size and alignment, CNN and DINO paths, on sizes not divisible by 8 or 14
    for h, w in [(257, 263), (256, 256), (1024, 1792), (256, 4096), (640, 800)]:
        fh, fw = -(-h // 8), -(-w // 8)
        r = M.restore_cnn(torch.zeros(1, fh, fw), h, w)
        check(f'CNN map restored to {h}x{w}', tuple(r.shape[-2:]) == (h, w))
        hp, wp = -(-h // 14), -(-w // 14)
        s = torch.zeros(1, hp, wp)
        i, j = hp // 2, wp // 3
        s[0, i, j] = 1.0
        r = M.restore_dino(s, h, w)
        py, px = np.unravel_index(int(r[0].argmax()), r[0].shape)
        cy, cx = i * 14 + 7, j * 14 + 7
        check(f'DINO map restored to {h}x{w}, peak on its token', tuple(r.shape[-2:]) == (h, w)
              and abs(py - cy) <= 14 and abs(px - cx) <= 14, f'peak {py},{px} vs token centre {cy},{cx}')
        x = torch.zeros(1, 3, h, w)
        check(f'pad14 {h}x{w} -> multiples of 14', all(v % 14 == 0 for v in M.pad14(x).shape[-2:]))
    v = M.DINO.valid(None, 19, 19, 256, 256)
    check('DINO bank uses only fully real tokens (18x18 of 19x19 for 256)', int(v.sum()) == 324)
    # end to end: a bright square on a textured image is found by both families
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(1)
    base = lambda: Image.fromarray((rng.normal(120, 12, (256, 256, 3))).clip(0, 255).astype(np.uint8))
    tiles = [base() for _ in range(40)]
    test = np.asarray(base()).copy().astype(np.uint8)
    test = np.asarray(Image.fromarray(test).resize((263, 257)))
    test = test.copy(); test[150:180, 60:90] = 250
    img = Image.fromarray(test)
    for Fam in (M.WRN, M.DINO):
        fam = Fam(device)
        X, info = M.sample_candidates(fam, tiles, total=4000, log=lambda *a: None)
        bank = M.greedy_coreset(X, 500, device, log=lambda *a: None)
        sc = M.PatchCore(bank, device) if Fam is M.WRN else M.DinoKNN(bank, device)
        m = M.score_maps(fam, [sc], [img])[sc.name][0]
        py, px = np.unravel_index(int(m.argmax()), m.shape)
        check(f'{fam.name}: 257x263 map size and peak on the square', m.shape == (257, 263)
              and 140 <= py <= 190 and 50 <= px <= 100, f'peak {py},{px}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--aitex')
    ap.add_argument('--torch', action='store_true')
    a = ap.parse_args()
    numpy_tests()
    if a.aitex:
        aitex_tests(Path(a.aitex))
    if a.torch:
        torch_tests()
    bad = [n for n, ok in RESULTS if not ok]
    print(f'\n{len(RESULTS) - len(bad)} passed, {len(bad)} failed')
    sys.exit(1 if bad else 0)
