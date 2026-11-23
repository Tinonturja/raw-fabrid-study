"""Methods figure: the Protocol 5 evaluation workflow (monochrome flow diagram). No data are read; every statement in
it restates the frozen protocol and implementation log. Usage: python3 fig_pipeline.py"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
INK, BAND, EDGE, GREY = '#111111', '#EFEFEF', '#BDBDBD', '#444444'
fig = plt.figure(figsize=(10.49, 7.79))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 74.3); ax.axis('off')

ax.text(1.4, 71.6, 'Protocol 5  |  Coverage, inference and evaluation', fontsize=13, weight='bold', va='center')
ax.text(1.4, 68.6, 'One fitted bank per detector and unit; paired layouts evaluated on native pixels', fontsize=8,
        va='center')

COL = [(6.2, 28.6), (36.2, 28.6), (68.0, 28.6)]          # x, width of the three box columns
ROWS = {'PROVENANCE': 61.0, 'INPUTS': 47.4, 'SCORING': 33.8, 'EVALUATION': 18.4}
BAND_H, BOX_H = 11.0, 8.8


def band(name):
    y = ROWS[name]
    ax.add_patch(Rectangle((1.4, y - BAND_H / 2), 97.0, BAND_H, fc=BAND, ec=EDGE, lw=0.6))
    ax.text(3.2, y, name, rotation=90, fontsize=6.8, weight='bold', ha='center', va='center')


def box(x, w, row, title, lines, h=BOX_H):
    y = ROWS[row]
    ax.add_patch(Rectangle((x, y - h / 2), w, h, fc='white', ec=INK, lw=0.8))
    ax.text(x + 1.2, y + h / 2 - 1.6, title, fontsize=8.2, weight='bold', va='center')
    for i, line in enumerate(lines):
        ax.text(x + 1.2, y + h / 2 - 3.9 - 1.75 * i, line, fontsize=7.2, va='center')
    return x, y, w, h


def arrow(x0, y0, x1, y1, dashed=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=0.8, color=INK,
                                 linestyle=(0, (3, 2)) if dashed else '-', shrinkA=0, shrinkB=0))


for b in ROWS:
    band(b)

# provenance
box(6.2, 43.0, 'PROVENANCE', 'Frozen protocol and cohort', ['Local SHA-256 fingerprints; one seed (13017).'])
box(51.0, 45.6, 'PROVENANCE', 'Verify source files',
    ['Four local datasets: hashes checked.', 'TSFabrics: no per-file hash verification in this run.'])
# inputs
box(*COL[0], 'INPUTS', 'Frames and masks', ['Native-resolution images; masks', 'retained for evaluation and audit.'])
box(*COL[1], 'INPUTS', 'Label-free layouts', ['R1: native   R2: edge-aligned grid', 'C4: quarter-scale frame'])
box(*COL[2], 'INPUTS', 'Practice A', ['Released RAW-FABRID crops;', 'constructed guided rules elsewhere.',
                                      'Label-informed retrospective path.'])
# scoring
box(*COL[0], 'SCORING', 'Normal reference tiles', ['Fit memory bank or shared Gaussian.',
                                                   'R1 and R2 reuse the fitted model.'])
box(*COL[1], 'SCORING', 'Frozen feature extractors', ['WRN-50: kNN / shared Gaussian', 'DINOv2: token kNN'])
box(*COL[2], 'SCORING', 'Restore native score maps', ['Max-fuse overlapping tile scores.',
                                                       'Assign unseen pixels the lowest score.'])
# evaluation
H = 9.2
box(*COL[0], 'EVALUATION', 'Label-only audit', ['Defect-pixel coverage', 'Connected-region fragmentation',
                                                'Patch defect fraction'], h=H)
box(*COL[1], 'EVALUATION', 'Whole-frame pixel AP', ['RAW-FABRID / TSFabrics: units', 'VisA: category macro-average',
                                                    'AITEX: pooled strip pixels'], h=H)
box(*COL[2], 'EVALUATION', 'Paired R2 − R1 inference', ['RAW / TS: resample units',
                                                             'VisA: images within each category',
                                                             'AITEX: resample strips'], h=H)

yP, yI, yS, yE = ROWS['PROVENANCE'], ROWS['INPUTS'], ROWS['SCORING'], ROWS['EVALUATION']
# provenance -> inputs
arrow(27.0, yP - BOX_H / 2, 27.0, yI + BOX_H / 2)
arrow(73.8, yP - BOX_H / 2, 73.8, yI + BOX_H / 2)
ax.plot([20.5, 20.5, 82.0], [yI + BOX_H / 2 + 1.3, yI + BOX_H / 2 + 2.0, yI + BOX_H / 2 + 2.0], color=INK, lw=0.8,
        ls=(0, (3, 2)))
arrow(82.0, yI + BOX_H / 2 + 2.0, 82.0, yI + BOX_H / 2, dashed=True)
# inputs row
arrow(COL[0][0] + COL[0][1], yI, COL[1][0], yI)
# layouts and Practice A -> scoring
arrow(50.5, yI - BOX_H / 2, 50.5, yS + BOX_H / 2)
ax.plot([82.0, 82.0, 50.5], [yI - BOX_H / 2, yI - BOX_H / 2 - 1.0, yI - BOX_H / 2 - 1.0], color=INK, lw=0.8,
        ls=(0, (3, 2)))
# scoring row
arrow(COL[0][0] + COL[0][1], yS, COL[1][0], yS)
arrow(COL[1][0] + COL[1][1], yS, COL[2][0], yS)
# restore -> AP, AP -> paired
ax.plot([82.0, 82.0, 50.5], [yS - BOX_H / 2, yS - BOX_H / 2 - 1.8, yS - BOX_H / 2 - 1.8], color=INK, lw=0.8)
arrow(50.5, yS - BOX_H / 2 - 1.8, 50.5, yE + H / 2)
arrow(COL[1][0] + COL[1][1], yE, COL[2][0], yE)
# frames and masks -> label-only audit
ax.plot([6.2, 5.0, 5.0], [yI, yI, yE], color=INK, lw=0.8, ls=(0, (3, 2)))
arrow(5.0, yE, 6.2, yE, dashed=True)

ax.text(6.2, 8.2, 'Decision rule: same direction on ≥3 of 4 main datasets; |point estimate| ≥0.01 and each 95% '
        'interval excludes zero.', fontsize=7.2, va='center')
ax.text(6.2, 4.6, 'Scope: Protocol 5. Coverage controls (P1–P4) and exploratory bank experiments (E1/E2) are '
        'separate analyses.', fontsize=7.2, va='center')

out = Path(__file__).resolve().parents[1] / 'figures'
fig.savefig(out / 'fig05_pipeline.pdf', metadata={'Creator': None, 'Producer': None})
fig.savefig(out / 'fig05_pipeline.png', dpi=300)
print('fig05_pipeline written')
