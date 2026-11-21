"""Protocol 5 core (no PyTorch): mask decoders, layouts, fusion, binning, AP, resampling.

Everything here is pure NumPy so it can be unit-tested anywhere. Detector code lives in
p5_models.py. Definitions follow PROTOCOL_5.md as amended by PROTOCOL_5_AMENDMENT_1.md.
"""
import hashlib
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

SIDE = 256
BPL = 5000          # histogram bins per unit of log1p(score); bin 0 = never scored
SEED = 13017


# ---------- A1: mask decoders (one per dataset, output: boolean defect mask) ----------

def _gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L'))


def decode_rawfabrid(path):
    return _gray(path) >= 128                      # 0/255 masks


def decode_aitex(paths):
    arrs = [_gray(p) >= 128 for p in paths]        # original external_aitex.py rule, OR over _mask1/_mask2
    assert arrs and all(a.shape == arrs[0].shape for a in arrs)
    return np.logical_or.reduce(arrs)


def decode_visa(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    assert a.ndim == 2, 'VisA mask must be single-channel category IDs'
    assert a.max() < 128, 'VisA mask has a value >= 128: not a category-ID mask'
    return a > 0                                   # category IDs 1..8 are defects


# ---------- layouts ----------

def grid_positions(length, side=SIDE):
    """Edge-aligned 1-D grid: steps of `side`; the last window is aligned to the edge."""
    assert length >= side, f'image side {length} smaller than window {side}'
    pos = list(range(0, length - side + 1, side))
    if pos[-1] + side < length:
        pos.append(length - side)
    return pos


def grid_layout(h, w, side=SIDE):
    """R2 / practice B layout: (y, x) top-left corners covering every pixel."""
    return [(y, x) for y in grid_positions(h, side) for x in grid_positions(w, side)]


def aitex_original_layout(mask):
    """Practice A on AITEX: the original guided_layout rule (floor division), 256 x 4096 strips."""
    assert mask.shape == (256, 4096)
    coords = []
    for x in range(0, 4096, SIDE):
        cols = np.flatnonzero(mask[:, x:x + SIDE].any(axis=0))
        left = int(np.clip(x + (cols[0] + cols[-1]) // 2 - 128, 0, 4096 - SIDE)) if len(cols) else x
        coords.append((0, left))
    return coords


def coverage(h, w, coords, side=SIDE):
    """Pixel multiplicity map: how many windows contain each pixel."""
    m = np.zeros((h, w), np.int32)
    for y, x in coords:
        m[y:y + side, x:x + side] += 1
    return m


# ---------- fusion ----------

def fuse_max(h, w, tile_maps, coords, side=SIDE):
    """One score per unique native pixel: maximum over windows; never-scored pixels = -inf."""
    field = np.full((h, w), -np.inf, np.float32)
    for m, (y, x) in zip(tile_maps, coords):
        assert m.shape == (side, side)
        np.maximum(field[y:y + side, x:x + side], m, out=field[y:y + side, x:x + side])
    return field


# ---------- binning and histograms ----------

def to_bins(v):
    """Scores -> integer bins; -inf (never scored) -> 0, the lowest rank."""
    out = np.zeros(v.shape, np.int64)
    ok = np.isfinite(v)
    out[ok] = (np.log1p(np.maximum(v[ok], 0.0)) * BPL).astype(np.int64) + 1
    return out


def hist(bins, fg):
    """Sparse histogram of one image (or one tile): (bin ids, totals, defect counts)."""
    bins = np.asarray(bins).ravel()
    fg = np.asarray(fg, bool).ravel()
    tot = np.bincount(bins)
    pos = np.bincount(bins[fg], minlength=tot.size)
    nz = np.flatnonzero(tot)
    return nz.astype(np.int32), tot[nz].astype(np.int64), pos[nz].astype(np.int64)


def dense(hists, nbins=None):
    """List of sparse histograms -> two dense matrices (n, nbins)."""
    nbins = nbins or (max(int(h[0].max()) for h in hists if len(h[0])) + 1)
    T = np.zeros((len(hists), nbins), np.float64)
    P = np.zeros((len(hists), nbins), np.float64)
    for i, (b, t, p) in enumerate(hists):
        T[i, b] = t
        P[i, b] = p
    return T, P


def ap_from_hist(tot, pos):
    """Average precision; pixels in one bin are tied and ranked together. Works on 1-D or 2-D (rows)."""
    tot = np.atleast_2d(tot)[:, ::-1]
    pos = np.atleast_2d(pos)[:, ::-1]
    ct = np.cumsum(tot, 1)
    cp = np.cumsum(pos, 1)
    npos = cp[:, -1]
    with np.errstate(invalid='ignore', divide='ignore'):
        prec = np.where(ct > 0, cp / np.maximum(ct, 1), 0.0)
        ap = (pos * prec).sum(1) / npos
    ap[npos == 0] = np.nan                         # undefined, never silently zero
    return ap if ap.size > 1 else float(ap[0])


def auroc(scores, labels):
    """Mann-Whitney AUROC with ties counted half."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------- A5: resampling ----------

def bootstrap_counts(n, reps=2000, seed=SEED, strata=None):
    """Shared resampling weights (reps, n). strata: array of group labels resampled separately."""
    rng = np.random.default_rng(seed)
    C = np.zeros((reps, n), np.float64)
    groups = [np.arange(n)] if strata is None else [np.flatnonzero(np.asarray(strata) == g)
                                                     for g in sorted(set(strata))]
    for g in groups:
        draws = rng.integers(0, len(g), size=(reps, len(g)))
        for r in range(reps):
            np.add.at(C[r], g[draws[r]], 1)
    return C


def cluster_counts(groups, reps=2000, seed=SEED):
    """Cluster bootstrap weights (reps, n_images): resample whole clusters; each image gets its cluster's count."""
    rng = np.random.default_rng(seed)
    labels = sorted(set(groups))
    idx = np.array([labels.index(g) for g in groups])
    draws = rng.integers(0, len(labels), size=(reps, len(labels)))
    cc = np.zeros((reps, len(labels)))
    for r in range(reps):
        np.add.at(cc[r], draws[r], 1)
    return cc[:, idx]


def boot_ap_diff(Ta, Pa, Tb, Pb, C, chunk=250):
    """Bootstrap distribution of AP(a) - AP(b) under shared weights C (reps, n_images)."""
    out = []
    for s in range(0, len(C), chunk):
        c = C[s:s + chunk]
        out.append(ap_from_hist(c @ Ta, c @ Pa) - ap_from_hist(c @ Tb, c @ Pb))
    return np.concatenate(out)


def macro_boot(cats, C_by_cat, chunk=250):
    """VisA: per-category AP difference, then macro mean, with the same draws for every comparison.
    cats: {name: (Ta, Pa, Tb, Pb)}; C_by_cat: {name: (reps, n_cat)}."""
    per = {k: boot_ap_diff(*v, C_by_cat[k], chunk) for k, v in cats.items()}
    return np.mean(np.stack(list(per.values())), 0), per


def percentile_ci(x, level=0.95):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def signflip_p(diffs):
    """Exact two-sided sign-flip p-value for a mean of paired differences."""
    d = np.asarray(diffs, float)
    obs = abs(d.mean())
    flips = np.array(list(product([1, -1], repeat=len(d))))
    return float((np.abs((flips * d).mean(1)) >= obs - 1e-15).mean())


def roll_bootstrap(diffs, reps=2000, seed=SEED):
    """Paired bootstrap over rolls, equal roll weights: resample roll differences, take the mean."""
    d = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    return d[rng.integers(0, len(d), size=(reps, len(d)))].mean(1)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
