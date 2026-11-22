"""Protocol 5 result figures, drawn only from the saved analysis files.
Fig 3: R2 - R1 forest plot per dataset and detector.  Fig 4: per-unit R2 - R1 (primary and DINOv2, separate panels).
Fig 5: fragmentation of defect regions by layout.  Fig 6: detector AP under R1 and R2 per dataset (ranking).
Usage: python3 fig_p5_results.py   (writes manuscript_v2/figures/fig3..fig6 .png and .pdf)"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'manuscript_v2' / 'figures'
R = HERE / 'results_p5'
COL = {'patchcore': '#2a78d6', 'gauss': '#eb6834', 'dino': '#1baf7a'}
NAME = {'patchcore': 'PatchCore-style (primary)', 'gauss': 'Shared Gaussian', 'dino': 'DINOv2-token kNN'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def load():
    j = lambda p: json.loads((R / p).read_text())
    return {'RAW-FABRID': ('roll', j('rawfabrid/full/analysis.json')),
            'TSFabrics': ('roll', j('tsfabrics/full/analysis_from_kaggle_log.json')),
            'VisA': ('macro', j('visa/full/analysis.json')),
            'AITEX': ('pooled', j('aitex/full/analysis.json')),
            'MVTec AD 2 Fabric\n(secondary)': ('pooled', j('mvtec_fabric/full/analysis.json'))}


def contrast(kind, d, key):
    blk = d['roll_level'] if kind == 'roll' else d['dataset_level']
    v = blk[key]
    return v['estimate'], v['ci95']


def mean_ap(kind, d, fam, lay):
    vals = [u['ap'][f'{fam}|{lay}'] for u in d['units'].values()]
    return float(np.mean(vals))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png')
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)


def fig3(data):
    fams = ['patchcore', 'gauss', 'dino']
    names = list(data)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={'width_ratios': [1.35, 1]}, sharey=True)
    for ax, sub, xlim, title in ((axes[0], ['patchcore', 'gauss'], (-0.05, 0.02), 'WRN-50 detectors'),
                                 (axes[1], ['dino'], (-0.05, 0.35), 'DINOv2-token kNN')):
        ax.axvspan(-0.01, 0.01, color=GRID, zorder=0, lw=0)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        for i, ds in enumerate(names):
            kind, d = data[ds]
            for k, fam in enumerate(sub):
                y = i + (k - (len(sub) - 1) / 2) * 0.22
                est, (lo, hi) = contrast(kind, d, f'{fam}: R2 - R1')
                secondary = 'secondary' in ds
                if not secondary:
                    ax.plot([lo, hi], [y, y], color=COL[fam], lw=1.6, solid_capstyle='round', zorder=2)
                ax.plot(est, y, 'o', ms=6, mfc='white' if secondary else COL[fam], mec=COL[fam], mew=1.4, zorder=3,
                        label=NAME[fam] if i == 0 else None)
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=9, color=INK, loc='left')
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    fig.supxlabel('Tiled minus native full frame, whole-frame pixel AP (R2 − R1)', fontsize=8.5, y=-0.03)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    save(fig, 'fig3_layout_contrast')


def fig4(data):
    units = [ds for ds in ('RAW-FABRID', 'TSFabrics', 'VisA')]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, fam, title in ((axes[0], 'patchcore', 'PatchCore-style (primary)'), (axes[1], 'dino', 'DINOv2-token kNN')):
        ax.axhline(0, color=INK2, lw=0.8)
        if fam == 'patchcore':
            ax.axhspan(-0.01, 0.01, color=GRID, lw=0, zorder=0)
        ticks = []
        for i, ds in enumerate(units):
            kind, d = data[ds]
            v = [u['ap'][f'{fam}|R2'] - u['ap'][f'{fam}|R1'] for u in d['units'].values()]
            x = i + np.linspace(-0.25, 0.25, len(v))
            ax.plot(x, v, 'o', ms=4.5, mfc=COL[fam], mec='white', mew=0.6, zorder=3)
            est, _ = contrast(kind, d, f'{fam}: R2 - R1')
            ax.plot([i - 0.32, i + 0.32], [est, est], color=INK, lw=1.4, zorder=4)
            ticks.append(f'{ds}\n{sum(t < 0 for t in v)}/{len(v)} below 0')
        ax.set_xticks(range(len(units)))
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_title(title, fontsize=9, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('R2 − R1, pixel AP per unit')
    axes[1].set_ylabel('R2 − R1, pixel AP per unit')
    fig.text(0.5, -0.04, 'Dots: rolls, condition clusters or categories. Black bar: equal-weight (macro) mean. Grey band: |effect| < 0.01. The two panels use different scales.',
             ha='center', fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, 'fig4_per_unit')


def fig5(data):
    names = [n for n in data if 'secondary' not in n]
    fig, axes = plt.subplots(1, len(names), figsize=(7.2, 2.6), sharey=False)
    for ax, ds in zip(axes, names):
        kind, d = data[ds]
        for fam in ('patchcore', 'gauss', 'dino'):
            a1, a2 = mean_ap(kind, d, fam, 'R1'), mean_ap(kind, d, fam, 'R2')
            ax.plot([0, 1], [a1, a2], '-o', color=COL[fam], ms=4.5, lw=1.6, label=NAME[fam])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Native\n(R1)', 'Tiled\n(R2)'])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(ds, fontsize=9, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Pixel AP (mean over units)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, 'fig6_rankings')


def fig6():
    rows = [('AITEX', 'aitex', 'R2', 'blind grid'), ('AITEX', 'aitex', 'A', 'guided crops'),
            ('RAW-FABRID', 'rawfabrid', 'R2', 'blind grid'), ('RAW-FABRID', 'rawfabrid', 'A', 'released crops'),
            ('VisA', 'visa', 'R2', 'blind grid')]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    labels = []
    for i, (ds, f, lay, nm) in enumerate(rows):
        s = json.loads((R / 'fragmentation' / f'{f}.json').read_text())['all'][lay]
        n = s['components']
        parts = [('Whole in one window', s['fit_components'] - s['fit_cut'] - s.get('fit_unseen', 0), '#2a78d6'),
                 ('Split across windows', s['fit_cut'], '#eb6834'),
                 ('Never seen', s.get('fit_unseen', 0), '#0b0b0b'),
                 ('Larger than a tile', s['oversize'], '#b8b6af')]
        left = 0
        for lab, v, c in parts:
            w = 100 * v / n
            ax.barh(i, w, left=left, color=c, height=0.62, label=lab if i == 0 else None, edgecolor='white', lw=1)
            if w >= 6:
                ax.text(left + w / 2, i, f'{w:.0f}%', ha='center', va='center', fontsize=7.5,
                        color='white' if c != '#b8b6af' else INK)
            left += w
        labels.append(f'{ds}, {nm}  (n = {n:,})')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of 8-connected defect regions (%)')
    ax.legend(loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.4, -0.5), fontsize=8)
    save(fig, 'fig5_fragmentation')


if __name__ == '__main__':
    data = load()
    fig3(data); fig4(data); fig5(data); fig6()
    print('written to', OUT)
