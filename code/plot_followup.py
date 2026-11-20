"""Create figures for the held-out-roll research follow-up."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/figures'
RESULTS=ROOT/'results/region_followup'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
INK='#1e2930'
BLUE='#1a5c7b'
ORANGE='#a45a1e'


def save(fig,name):
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)


def methodology():
    fig,ax=plt.subplots(figsize=(7.3,4.5))
    ax.set(xlim=(0,100),ylim=(0,100))
    ax.axis('off')
    def box(x,y,w,h,title,text):
        ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=.9,edgecolor=INK))
        ax.text(x+w/2,y+h-3,title,fontsize=8,weight='bold',ha='center',va='top')
        ax.text(x+w/2,y+h-9,text,fontsize=7.8,ha='center',va='top',linespacing=1.2)
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.8,color=INK))
    box(1,69,29,23,'Held-out test roll','Frames and source masks\nReleased and fixed windows')
    box(35,69,29,23,'Six reference rolls','1,000 clean tiles\nNo held-out roll samples')
    box(69,69,30,23,'Frozen feature models','ResNet-18 spatial features\nSwin-T attention features')
    box(1,34,29,23,'Tile extraction','28 windows per policy\n128 × 128 model input')
    box(35,34,29,23,'Normal reference fit','ResNet coreset\nSwin position-wise Gaussians')
    box(69,34,30,23,'Spatial scoring','Nearest distance or\nGaussian distance per cell')
    box(1,2,29,20,'Native reconstruction','Expand 16 × 16 scores\nMax fusion on overlaps')
    box(35,2,29,20,'Pixel and region metrics','Observed and whole AP\nRegion overlap below 5% FPR')
    box(69,2,30,20,'Roll-level comparison','Paired policy differences\nEqual-roll means')
    for a,b in [((15.5,69),(15.5,57)),((49.5,69),(49.5,57)),((84,69),(84,57)),
                ((64,45.5),(69,45.5)),
                ((30,12),(35,12)),((64,12),(69,12))]:
        arrow(a,b)
    ax.plot([15.5,15.5,66,66],[34,30,30,41],color=INK,lw=.8)
    arrow((66,41),(69,41))
    ax.plot([84,84,15.5],[34,26,26],color=INK,lw=.8)
    arrow((15.5,26),(15.5,22))
    save(fig,'heldout_methodology')


def overview():
    summary=json.loads((ROOT/'results/unseen_roll/summary.json').read_text())['summary']['primary']
    fig,axes=plt.subplots(1,3,figsize=(7.4,2.8))
    axes[0].bar(['Observed','Outside'],[100*(1-140701/1904926),100*140701/1904926],color=[BLUE,ORANGE])
    axes[0].set_ylim(0,105); axes[0].set_ylabel('Marked pixels (%)')
    axes[0].set_title('A  Mask coverage',loc='left',weight='bold',fontsize=10)
    for i,v in enumerate([100*(1-140701/1904926),100*140701/1904926]):
        axes[0].text(i,v+2,f'{v:.1f}',ha='center',fontsize=8)
    names=['ResNet-18','Swin-T']
    for ax, metric, title in [(axes[1],'frame_auroc','B  Frame AUROC'),(axes[2],'whole_ap','C  Whole-frame AP')]:
        values=[summary[m] for m in ('resnet18','swin_t')]
        y=np.arange(2)
        ax.barh(y-.17,[v['released'][metric] for v in values],height=.31,color=BLUE,label='Released')
        ax.barh(y+.17,[v['fixed'][metric] for v in values],height=.31,color=ORANGE,label='Fixed')
        ax.set_yticks(y,names);ax.invert_yaxis()
        ax.set_title(title,loc='left',weight='bold',fontsize=10)
        ax.tick_params(labelsize=8)
        if metric=='frame_auroc':ax.set_xlim(0,1)
        else:ax.set_xlim(0,.23)
    fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=8,bbox_to_anchor=(.67,-.02))
    fig.subplots_adjust(left=.075,right=.98,top=.83,bottom=.24,wspace=.55)
    save(fig,'results_overview')


def maps():
    filenames=sorted(RESULTS.glob('resnet18_*.npz'))
    assert len(filenames)==2
    fig,axes=plt.subplots(2,4,figsize=(9,3.4))
    for row,path in enumerate(filenames):
        swin=RESULTS/path.name.replace('resnet18_','swin_t_',1)
        a=np.load(path); b=np.load(swin)
        np.testing.assert_array_equal(a['image'],b['image'])
        np.testing.assert_array_equal(a['mask'],b['mask'])
        axes[row,0].imshow(a['image'],cmap='gray',vmin=0,vmax=255)
        axes[row,1].imshow(a['image'],cmap='gray',vmin=0,vmax=255)
        axes[row,1].contour(a['mask'],[.5],colors=['#f4c542'],linewidths=.65)
        axes[row,1].contour(a['released']>0,[.5],colors=['#00a6d4'],linewidths=.5)
        for col,data in [(2,a),(3,b)]:
            score=data['fixed'].astype(float)
            score=(score-score.min())/max(score.max()-score.min(),1)
            axes[row,col].imshow(score,cmap='magma',vmin=0,vmax=1)
        for ax in axes[row]:ax.set_axis_off()
        axes[row,0].text(-.06,.5,f'Example {row+1}',rotation=90,va='center',
                         ha='right',transform=axes[row,0].transAxes,fontsize=9)
    for ax,t in zip(axes[0],['Original frame','Mask and coverage','ResNet-18 map','Swin-T map']):
        ax.set_title(t,fontsize=9,weight='bold')
    fig.subplots_adjust(left=.06,right=.99,top=.91,bottom=.02,wspace=.06,hspace=.06)
    save(fig,'prediction_examples')


def regions():
    models=('resnet18','swin_t')
    grouped={}
    for model in models:
        paths=sorted(RESULTS.glob(model+'_Rollo*.json'))
        assert len(paths)==7
        grouped[model]=[json.loads(p.read_text()) for p in paths]
    summary={}
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.3),sharex=True,sharey=True)
    for ax,model,title in zip(axes,models,['A  ResNet-18','B  Swin-T']):
        rows=grouped[model]
        for i,r in enumerate(rows):
            released=r['policies']['released']['aupro_005']
            fixed=r['policies']['fixed']['aupro_005']
            ax.plot([fixed,released],[i,i],color='#8b999e',lw=1)
            ax.scatter([fixed,released],[i,i],c=[ORANGE,BLUE],s=22,zorder=3)
        a=np.array([r['policies']['released']['aupro_005'] for r in rows])
        b=np.array([r['policies']['fixed']['aupro_005'] for r in rows])
        ax.plot([b.mean(),a.mean()],[7,7],color='#8b999e')
        ax.scatter([b.mean(),a.mean()],[7,7],c=[ORANGE,BLUE],s=38,zorder=3)
        ax.axhline(6.5,color='#aaa',lw=.7)
        ax.set_yticks(range(8),[r['roll'].replace('Rollo','') for r in rows]+['Mean'])
        ax.invert_yaxis();ax.set_title(title,loc='left',fontsize=10,weight='bold')
        ax.set_xlim(0,1);ax.set_xlabel('Region overlap, FPR ≤ 0.05')
        summary[model]={'released_mean':float(a.mean()),'fixed_mean':float(b.mean()),
                        'mean_difference':float((a-b).mean()),
                        'rolls':{r['roll']:{'released':float(ra),'fixed':float(fa)}
                                 for r,ra,fa in zip(rows,a,b)}}
    fig.legend(handles=[plt.Line2D([],[],marker='o',color=c,ls='',label=l) for c,l in [(BLUE,'Released'),(ORANGE,'Fixed')]],
               loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.56,1.02))
    fig.subplots_adjust(left=.1,right=.98,top=.78,bottom=.18,wspace=.15)
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    save(fig,'region_overlap')



def external():
    record=json.loads((ROOT/'results/fabrid_external.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(7.4,2.7))
    models=['resnet18','swin_t']
    names=['ResNet-18','Swin-T']
    specs=[('frame_auroc','A  Frame AUROC',1),('whole_ap','B  Whole-frame AP',.1),
           ('aupro_005','C  Region overlap',.2)]
    for ax,(metric,title,maximum) in zip(axes,specs):
        y=np.arange(2)
        ax.barh(y-.17,[record['models'][m]['guided'][metric] for m in models],
                height=.3,color=BLUE,label='Mask-guided')
        ax.barh(y+.17,[record['models'][m]['fixed'][metric] for m in models],
                height=.3,color=ORANGE,label='Fixed')
        ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,maximum)
        ax.set_title(title,loc='left',weight='bold',fontsize=9)
        ax.tick_params(labelsize=7.5)
        if metric=='frame_auroc':
            ax.axvline(.5,color='#777',ls='--',lw=.8)
        for i,m in enumerate(models):
            for delta,policy in [(-.17,'guided'),(.17,'fixed')]:
                val=record['models'][m][policy][metric]
                ax.text(val+maximum*.015,i+delta,f'{val:.3f}',va='center',fontsize=6.7)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,
               frameon=False,fontsize=8,bbox_to_anchor=(.5,-.02))
    fig.subplots_adjust(left=.09,right=.97,top=.83,bottom=.26,wspace=.62)
    save(fig,'fabrid_external')


if __name__=='__main__':
    methodology()
    overview()
    if (ROOT/'results/fabrid_external.json').exists():
        external()
    if len(list(RESULTS.glob('*.npz')))==4:
        maps()
    if len(list(RESULTS.glob('*_Rollo*.json')))==14:
        regions()
