"""Draw the paired RAW-FABRID comparisons from the saved frame and pixel results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
res = ROOT/'results'
global_runs = json.loads((res/'crop_publication_scope.json').read_text())['runs']
seeds = (13017,13018,13019)
colors = ('#186098','#b55020','#497346')
fig, axes = plt.subplots(1,2,figsize=(9.4,4.05),layout='constrained')
for idx,seed in enumerate(seeds):
    name = 'localization_study.json' if idx == 0 else f'localization_study_{seed}.json'
    spatial = json.loads((res/name).read_text())['metrics']
    glob = global_runs[str(seed)]['metrics']
    for group,metrics,key in [(0,glob,'image_auroc_303'),(1,spatial,'auroc')]:
        x=[group*2.6,group*2.6+1]
        axes[0].plot(x,[metrics[p][key] for p in ('released','fixed')],color=colors[idx],marker='o',ms=5)
    for group,key in enumerate(('pixel_ap_observed','pixel_ap_all')):
        x=[group*2.6,group*2.6+1]
        axes[1].plot(x,[spatial[p][key] for p in ('released','fixed')],color=colors[idx],marker='o',ms=5)
for ax,groups,title,y in zip(axes,[('Global','Spatial'),('Observed area','Whole frame')],
                             ['Frame detection','Spatial detector localization'],['Image AUROC','Pixel average precision']):
    ax.set_title(title,loc='left',fontweight='bold')
    ax.set_ylabel(y)
    ax.set_xticks([0,1,2.6,3.6],['Released','Fixed','Released','Fixed'])
    ax.set_xlim(-.38,3.98)
    for x,group in zip([.5,3.1],groups):
        ax.text(x,-.16,group,transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.22)
fig.legend([Line2D([0],[0],color=c,marker='o') for c in colors],[str(s) for s in seeds],
           loc='outside lower center',ncols=3,frameon=False,title='Split seed')
out=ROOT/'figures'/'crop_policy_comparison.png'
out.parent.mkdir(exist_ok=True)
fig.savefig(out,dpi=170)
