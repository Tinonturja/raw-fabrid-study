"""Protocol 5 result figures, drawn only from the saved analysis files, in the shared manuscript style.
Fig 3: R2 - R1 forest plot per dataset and detector.  Fig 4: per-unit R2 - R1 (primary and DINOv2).
Fig 5: fragmentation of defect regions by layout.  Fig 6: detector AP under R1 and R2 per dataset (ranking).
Usage: python3 fig_p5_results.py   (writes manuscript_v2/figures/fig3..fig6 .png and .pdf)"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'manuscript_v2' / 'figures'
R = HERE / 'results_p5'
S.apply()


def load():
    j = lambda p: json.loads((R / p).read_text())
    return {'RAW-FABRID': ('roll', j('rawfabrid/full/analysis.json')),
            'TSFabrics': ('roll', j('tsfabrics/full/analysis_from_kaggle_log.json')),
            'VisA': ('macro', j('visa/full/analysis.json')),
            'AITEX': ('pooled', j('aitex/full/analysis.json')),
            'MVTec AD 2 Fabric': ('pooled', j('mvtec_fabric/full/analysis.json'))}


def contrast(kind, d, key):
    v = (d['roll_level'] if kind == 'roll' else d['dataset_level'])[key]
    return v['estimate'], v['ci95']


def mean_ap(d, fam, lay):
    return float(np.mean([u['ap'][f'{fam}|{lay}'] for u in d['units'].values()]))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png')
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)


def fig3(data):
    names = list(data)
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 3.0), gridspec_kw={'width_ratios': [1.4, 1], 'wspace': 0.16},
                             sharey=True)
    for ax, sub, xlim, title, letter in ((axes[0], ['patchcore', 'gauss'], (-0.05, 0.02), 'WRN-50 feature detectors', 'a'),
                                         (axes[1], ['dino'], (-0.06, 0.34), 'DINOv2-token kNN', 'b')):
        ax.axvspan(-0.01, 0.01, color=S.BAND, zorder=0, lw=0)
        ax.axvline(0, color=S.INK2, lw=0.7, zorder=1)
        for i, ds in enumerate(names):
            kind, d = data[ds]
            sec = ds.startswith('MVTec')
            for k, fam in enumerate(sub):
                y = i + (k - (len(sub) - 1) / 2) * 0.24
                est, (lo, hi) = contrast(kind, d, f'{fam}: R2 - R1')
                if not sec:
                    ax.plot([lo, hi], [y, y], color=S.DET[fam], lw=1.3, solid_capstyle='butt', zorder=2)
                ax.plot(est, y, 'o', ms=4.8, mfc='white' if sec else S.DET[fam], mec=S.DET[fam], mew=1.1, zorder=3)
        ax.axhline(3.55, color=S.SAND, lw=0.6, ls=(0, (2, 2)))
        ax.set_xlim(*xlim)
        ax.set_title(title)
        ax.grid(axis='x')
        ax.tick_params(axis='y', length=0)
        S.panel(ax, letter, x=-0.14 if letter == 'b' else -0.30)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([n if not n.startswith('MVTec') else 'MVTec AD 2 Fabric\n(secondary)' for n in names])
    axes[0].invert_yaxis()
    fig.supxlabel('Tiled minus native full-frame prediction, whole-frame pixel AP  (R2 − R1)', fontsize=8, y=-0.04)
    handles = [Line2D([], [], color=S.DET[f], marker='o', ms=4.8, lw=1.3, label=S.DET_NAME[f]) for f in S.DET]
    handles += [Line2D([], [], color=S.MUTED, marker='o', ms=4.8, mfc='white', lw=0, label='no interval (secondary)'),
                plt.Rectangle((0, 0), 1, 1, color=S.BAND, label='|effect| < 0.01')]
    fig.legend(handles=handles, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.08), handlelength=1.6, columnspacing=1.2)
    save(fig, 'fig3_layout_contrast')


def fig4(data):
    units = ['RAW-FABRID', 'TSFabrics', 'VisA']
    nunit = {'RAW-FABRID': 'rolls', 'TSFabrics': 'clusters', 'VisA': 'categories'}
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.7), gridspec_kw={'wspace': 0.32})
    rng = np.random.default_rng(13017)
    for ax, fam, title, letter in ((axes[0], 'patchcore', 'PatchCore-style (primary)', 'a'),
                                   (axes[1], 'dino', 'DINOv2-token kNN', 'b')):
        if fam == 'patchcore':
            ax.axhspan(-0.01, 0.01, color=S.BAND, lw=0, zorder=0)
        ax.axhline(0, color=S.INK2, lw=0.7, zorder=1)
        ticks = []
        for i, ds in enumerate(units):
            kind, d = data[ds]
            v = np.array([u['ap'][f'{fam}|R2'] - u['ap'][f'{fam}|R1'] for u in d['units'].values()])
            x = i + rng.uniform(-0.17, 0.17, len(v))
            ax.scatter(x, v, s=16, color=S.DET[fam], alpha=0.85, edgecolor='white', linewidth=0.5, zorder=3)
            est, (lo, hi) = contrast(kind, d, f'{fam}: R2 - R1')
            ax.plot([i + 0.3, i + 0.3], [lo, hi], color=S.INK, lw=1.0, zorder=4)
            ax.plot([i + 0.3], [est], marker='D', ms=4, color=S.INK, zorder=5)
            ticks.append(f'{ds}\n{(v < 0).sum()} of {len(v)} below 0')
        ax.set_xticks(range(len(units)))
        ax.set_xticklabels(ticks)
        ax.tick_params(axis='x', length=0)
        ax.set_xlim(-0.5, 2.6)
        ax.set_title(title)
        ax.set_ylabel('R2 − R1, pixel AP')
        ax.grid(axis='y')
        S.panel(ax, letter, x=-0.2)
    save(fig, 'fig4_per_unit')


def fig5():
    rows = [('AITEX', 'aitex', 'R2', 'blind grid'), ('AITEX', 'aitex', 'A', 'guided crops'),
            ('RAW-FABRID', 'rawfabrid', 'R2', 'blind grid'), ('RAW-FABRID', 'rawfabrid', 'A', 'released crops'),
            ('VisA', 'visa', 'R2', 'blind grid')]
    fig, ax = plt.subplots(figsize=(S.DOUBLE, 2.2))
    labels = []
    cats = [('Whole in one window', S.COBALT, 'white'), ('Split across windows', S.COPPER, 'white'),
            ('Never seen', S.INK, 'white'), ('Larger than a tile', S.SAND, S.INK)]
    for i, (ds, f, lay, nm) in enumerate(rows):
        s = json.loads((R / 'fragmentation' / f'{f}.json').read_text())['all'][lay]
        n = s['components']
        vals = [s['fit_components'] - s['fit_cut'] - s.get('fit_unseen', 0), s['fit_cut'], s.get('fit_unseen', 0), s['oversize']]
        left = 0
        for (lab, c, tc), v in zip(cats, vals):
            w = 100 * v / n
            ax.barh(i, w, left=left, color=c, height=0.66, label=lab if i == 0 else None, edgecolor='white', lw=0.8)
            if w >= 5:
                ax.text(left + w / 2, i, f'{w:.0f}%', ha='center', va='center', fontsize=7, color=tc)
            left += w
        labels.append(f'{ds}, {nm}   n = {n:,}')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of 8-connected defect regions (%)')
    ax.spines['left'].set_visible(False)
    ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.42, 1.2), handlelength=1.2)
    save(fig, 'fig5_fragmentation')


def fig6(data):
    names = [n for n in data if not n.startswith('MVTec')]
    fig, axes = plt.subplots(1, len(names), figsize=(S.DOUBLE, 2.4), gridspec_kw={'wspace': 0.42})
    for ax, ds, letter in zip(axes, names, 'abcd'):
        kind, d = data[ds]
        for fam in ('patchcore', 'gauss', 'dino'):
            a1, a2 = mean_ap(d, fam, 'R1'), mean_ap(d, fam, 'R2')
            ax.plot([0, 1], [a1, a2], '-', color=S.DET[fam], lw=1.4, zorder=2)
            ax.plot([0, 1], [a1, a2], 'o', color=S.DET[fam], ms=4.5, mec='white', mew=0.6, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Native', 'Tiled'])
        ax.tick_params(axis='x', length=0)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(ds)
        ax.grid(axis='y')
        S.panel(ax, letter, x=-0.42)
    axes[0].set_ylabel('Mean pixel AP')
    handles = [Line2D([], [], color=S.DET[f], marker='o', ms=4.5, lw=1.4, label=S.DET_NAME[f]) for f in S.DET]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1))
    save(fig, 'fig6_rankings')


if __name__ == '__main__':
    data = load()
    fig3(data); fig4(data); fig5(); fig6(data)
    print('written to', OUT)
