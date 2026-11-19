"""Plot released-minus-fixed localization differences for each held-out roll."""
import json

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from plot_study_design import BLUE, ORANGE, INK, ROOT, plt, save


def main():
    data = json.loads((ROOT/'results/unseen_roll/summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.985, bottom=.22, top=.78, wspace=.14)
    rolls = list(data['roll_populations'])
    y = np.arange(len(rolls)+1)[::-1]
    for j, name in enumerate(['resnet18','swin_t']):
        rows = data['per_roll']['primary'][name]
        diffs = [rows[r]['difference'] for r in rolls] + [data['summary']['primary'][name]['difference']]
        ax = axes[j]
        ax.axvline(0, c='#979d9c', ls='--', lw=.9)
        ax.axhline(.5, c='#aab5bb', lw=.7)
        for k, d in enumerate(diffs):
            a, b = d['observed_ap'], d['whole_ap']
            ax.plot([a,b],[y[k],y[k]], c='#9aa9ae', lw=1.1, zorder=2)
            size = 6.6 if k == len(rolls) else 5.1
            ax.plot(a,y[k], marker='o', ms=size, color=BLUE)
            ax.plot(b,y[k], marker='s', ms=size*.90, color=ORANGE)
        ax.set_title(['A   ResNet-18 / PatchCore','B   Swin-T / Gaussian'][j], loc='left', weight='bold', size=10)
        ax.set_yticks(y,[r.replace('Rollo','') for r in rolls]+['Mean'])
        ax.tick_params(axis='y',length=0)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_ylim(-.6,len(rolls)+.6)
    fig.legend(handles=[Line2D([],[],ls='none',marker=m,ms=5.8,color=c,label=l)
                       for m,c,l in [('o',BLUE,'Observed pixels'),('s',ORANGE,'Whole frame')]],
               loc='upper center', ncol=2, frameon=False,bbox_to_anchor=(.54,1.02))
    fig.text(.54,.035,'Released minus fixed pixel average precision',ha='center',color=INK)
    save(fig,'unseen_roll')

if __name__ == '__main__':
    main()
