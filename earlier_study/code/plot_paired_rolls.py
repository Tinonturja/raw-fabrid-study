#!/usr/bin/env python3
"""Plot exploratory paired roll results using the fixed-grid features."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/'results'/'features_fixed_grid_paired.json',ROOT/'results'/'texture_features_fixed_grid_paired.json']
titles=['Pretrained ResNet-18 features','Fixed texture features']
fig,axes=plt.subplots(1,2,figsize=(12,5.5),sharey=True)
fig.subplots_adjust(left=.07,right=.99,top=.79,bottom=.27,wspace=.08)
for ax,p,title in zip(axes,paths,titles):
    d=json.loads(p.read_text()); seeds=sorted(d['seeds']); groups=sorted(d['seeds'][seeds[0]])
    x=np.arange(len(groups)); a=np.array([[d['seeds'][s][g]['within_pool_auroc'] for g in groups] for s in seeds]); b=np.array([[d['seeds'][s][g]['heldout_roll_auroc'] for g in groups] for s in seeds])
    ax.bar(x-.18,a.mean(axis=0),.36,label='Includes same-roll reference',color='#287f8e')
    ax.bar(x+.18,b.mean(axis=0),.36,label='Excludes same-roll reference',color='#d88b48')
    for i in range(len(groups)):
        ax.scatter(np.repeat(x[i]-.18,len(seeds)),a[:,i],color='#123f49',s=11,zorder=3)
        ax.scatter(np.repeat(x[i]+.18,len(seeds)),b[:,i],color='#703c15',s=11,zorder=3)
    labels=[f"{g.replace('Rollo','')}\n{d['seeds'][seeds[0]][g]['positive']} / {d['seeds'][seeds[0]][g]['negative']}" for g in groups]
    ax.set_xticks(x,labels);ax.set_ylim(0,1.04);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_xlabel('Roll ID\nDefect / clean test images')
axes[0].set_ylabel('Image-level AUROC')
handles,labels=axes[1].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,fontsize=8,frameon=False)
fig.suptitle('Exploratory RAW-FABRID fixed-grid evaluation',fontsize=14,fontweight='bold',y=.98)
fig.text(.5,.045,'Bars show means across three seeds; dots show individual runs. Seeds reuse the same dataset.\nThe two training pools differ in composition, so these comparisons are descriptive.',ha='center',fontsize=9)
out=ROOT/'reports'/'fixed_grid_roll_comparison.png';fig.savefig(out,dpi=180,bbox_inches='tight');print(out)
