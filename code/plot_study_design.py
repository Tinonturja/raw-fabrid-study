"""Draw the study design and paired statistical figures at journal column-spanning size."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'manuscript'/'figures'
BLUE, ORANGE, INK, LINE = '#145884', '#a54b17', '#172b3b', '#657782'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelcolor':INK,
                     'text.color':INK,'axes.edgecolor':LINE,'xtick.color':INK,
                     'ytick.color':INK,'pdf.fonttype':42, 'axes.spines.top':False,
                     'axes.spines.right':False})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ('pdf','png'):
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {}
        fig.savefig(OUT/f'{name}.{ext}',dpi=360,facecolor='white',bbox_inches='tight',
                    pad_inches=.05,metadata=metadata)
    plt.close(fig)


def methodology():
    fig, ax = plt.subplots(figsize=(7.25,6.55))
    fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,100),ylim=(-5,100)); ax.axis('off')

    def box(x,top,w,h,title,body,color=INK,fill='#f3f6f7'):
        ax.add_patch(FancyBboxPatch((x,top-h),w,h,boxstyle='round,pad=0.14,rounding_size=1.2',
                                    facecolor=fill,edgecolor=color,linewidth=1))
        ax.text(x+2,top-2.05,title,va='top',color=color,fontsize=10.2,fontweight='bold')
        ax.text(x+2,top-5.6,body,va='top',fontsize=8.45,linespacing=1.23)

    def arrow(x,y,to):
        ax.annotate('',(x,to),(x,y),arrowprops={'arrowstyle':'-|>','lw':.95,
                                                'color':LINE,'shrinkA':0,'shrinkB':.9})

    def join(top,bottom,branch):
        mid=(top+bottom)/2
        if branch:
            ax.plot([50,50],[top,mid],color=LINE,lw=.95)
            for x in (25,75): arrow(x,mid,bottom)
        else:
            for x in (25,75):ax.plot([x,x],[top,mid],color=LINE,lw=.95)
            arrow(50,mid,bottom)
        ax.plot([25,75],[mid,mid],color=LINE,lw=.95)

    box(1,99,48,14,'A   RAW-FABRID','709 original frames · 7 rolls\n19,852 released crops verified pixel by pixel',BLUE,'#ecf4f8')
    box(51,99,48,14,'B   AITEX','103 defective and 121 clean eligible strips\n6 fabrics · constructed guided positions',ORANGE,'#faf1eb')
    join(85,80,False)
    box(1,80,98,12,'Run independently within each dataset','Split clean frames into reference, calibration and evaluation sets. Hold out defects for scoring.')
    join(68,63,True)
    box(1,63,48,16,'Clean reference tiles','Frozen ImageNet ResNet-18\n1,000 RAW tiles or 960 AITEX tiles\nFit each detector on clean examples.')
    box(51,63,48,16,'Windows on the same test frame','Guided · fixed · donor · dense overlap\nOnly guided and donor layouts derive from\ndefect-informed positions.')
    join(47,42,False)
    box(1,42,98,12,'Score each window','Global nearest-neighbor distance or spatial PatchCore variant. Fuse spatial scores at native coordinates.')
    join(30,25,True)
    box(1,25,48,12,'Frame detection','Frame AUROC · clean-calibrated alarms\nCount defects flagged and false alarms.')
    box(51,25,48,12,'Localization','Observed-area and whole-frame pixel AP\nCount connected defect regions hit.')
    join(13,9,False)
    box(1,9,98,13,'Sensitivity and follow-up','Both datasets: paired group resampling, group deletion and equal-budget rescanning.\nRAW-FABRID: leave one roll out, refit ResNet-18 and Swin-T on the other rolls, compare both pixel areas.')
    save(fig,'methodology')


def legend(fig,labels,colors,markers):
    handles=[Line2D([],[],color=c,marker=m,ls='none',ms=6,label=l) for l,c,m in zip(labels,colors,markers)]
    fig.legend(handles=handles,loc='upper center',frameon=False,ncol=2,bbox_to_anchor=(.5,1.02))


def intervals():
    data=json.loads((ROOT/'results/cluster_statistics.json').read_text())
    specs=[('A   Frame ranking',['global_auc','spatial_auc','interaction_auc'],
            ['Global detector','Spatial detector','Global − spatial¹'],'Difference in AUROC'),
           ('B   Pixel localization',['observed_ap','whole_ap','rescan_ap'],
            ['Observed area','Whole frame','Targeted − preset²'],'Difference in pixel AP')]
    fig,axes=plt.subplots(1,2,figsize=(7.5,2.9))
    fig.subplots_adjust(left=.02,right=.99,bottom=.22,top=.76,wspace=.30)
    for ax,(title,keys,labels,xlab) in zip(axes,specs):
        ax.axvline(0,ls='--',lw=.9,color=LINE)
        for idx,(key,label) in enumerate(zip(keys,labels)):
            y=2-idx
            ax.text(.015,y+.18,label,transform=ax.get_yaxis_transform(),size=8,
                    bbox=dict(boxstyle='square,pad=.05',fc='white',ec='none'))
            for dataset,off,marker,color in [('raw',.00,'o',BLUE),('aitex',-.16,'s',ORANGE)]:
                row=data[dataset]['effects'][key]
                lo,hi=row['interval_95']
                ax.plot([lo,hi],[y+off]*2,color=color,lw=1.5)
                ax.plot([lo,hi],[y+off]*2,'|',color=color,ms=5)
                ax.plot(row['estimate'],y+off,marker=marker,color=color,ms=4.7)
        ax.set_ylim(-.42,2.48);ax.set_yticks([]);ax.spines['left'].set_visible(False)
        ax.set_title(title,loc='left',weight='bold',fontsize=10.8,pad=6);ax.set_xlabel(xlab)
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.tick_params(labelsize=8)
    legend(fig,['RAW-FABRID','AITEX'],[BLUE,ORANGE],['o','s'])
    save(fig,'group_intervals')


def rescan():
    coverage=json.loads((ROOT/'results/rescan_coverage.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(7.5,2.55))
    fig.subplots_adjust(left=.09,right=.99,top=.73,bottom=.17,wspace=.23)
    for j,ax in enumerate(axes):
        for y,dataset in [(1,'raw'),(0,'aitex')]:
            r=json.loads((ROOT/f'results/budgeted_rescan_{dataset}.json').read_text())
            values=[]
            for policy in ('even','targeted'):
                m=r['metrics'][policy]
                values.append([coverage[dataset]['summary'][policy]['fraction_of_marked_pixels'],
                               m['regions_hit']/m['regions'],m['false_alarms']/len(r['evaluation_clean'])][j])
            ax.plot(values,[y,y],color=LINE,linewidth=1.2)
            for v,marker,color,voff in zip(values,['s','o'],[LINE,BLUE],[-.23,.16]):
                ax.plot(v,y,marker=marker,color=color,ms=5.5)
                ax.text(v,y+voff,f'{100*v:.1f}',ha='center',va='center',size=8,color=color)
        ax.set_title(['A   Marked pixels\n      revisited','B   Defect regions\n      hit','C   Clean frames\n      flagged'][j],fontsize=10,loc='left',weight='bold')
        ax.set_ylim(-.42,1.48);ax.set_xlim(-.035,[.72,.72,.26][j])
        ax.set_yticks([1,0],['RAW','AITEX'] if j==0 else []);ax.tick_params(axis='y',length=0)
        ax.xaxis.set_major_formatter(PercentFormatter(1,0));ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.spines['left'].set_visible(False);ax.tick_params(axis='x',labelsize=8)
    legend(fig,['Preset extras','Targeted extras'],[LINE,BLUE],['s','o'])
    save(fig,'rescan_outcomes')


if __name__=='__main__':
    methodology()
    intervals()
    rescan()
