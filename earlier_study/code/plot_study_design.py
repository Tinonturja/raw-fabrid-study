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
OUT = ROOT/'figures'
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
    """Draw a monochrome swimlane flowchart of the research sequence."""
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7.3, 6.1))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis('off')
    ink = '#17212a'
    ax.add_patch(Rectangle((.5, .5), 99, 99, fill=False, edgecolor=ink, linewidth=1))
    ax.text(2, 97.4, 'Fabric anomaly inspection  |  Research methodology', fontsize=10,
            fontweight='bold', va='center', color=ink)
    ax.plot([.5, 99.5], [94.5, 94.5], color=ink, lw=.8)

    lanes = [(77, 94.5, 'Source data'), (59, 77, 'Preparation'),
             (41, 59, 'Scoring'), (23, 41, 'Evaluation'),
             (.5, 23, 'Follow-up')]
    for bottom, top, label in lanes:
        ax.plot([.5, 99.5], [bottom, bottom], color=ink, lw=.6)
        ax.text(5.5, (bottom+top)/2, label, rotation=90, va='center',
                ha='center', fontsize=8, fontweight='bold', color=ink)
    ax.plot([10.5, 10.5], [.5, 94.5], color=ink, lw=.8)

    def box(x, y, width, height, title, body):
        ax.add_patch(Rectangle((x, y), width, height, facecolor='white',
                               edgecolor=ink, linewidth=.8))
        ax.text(x+width/2, y+height-2, title, ha='center', va='top',
                fontsize=8.4, fontweight='bold', color=ink)
        ax.text(x+width/2, y+height-5.3, body, ha='center', va='top',
                fontsize=7.1, linespacing=1.18, color=ink)

    def arrow(x, y1, y2):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='-|>', lw=.75, color=ink,
                                    shrinkA=0, shrinkB=0, mutation_scale=8))

    box(13, 81, 39, 10, 'RAW-FABRID',
        '709 original frames · seven rolls\nPublisher-provided crop coordinates')
    box(58, 81, 39, 10, 'AITEX',
        'Six eligible fabrics\nConstructed mask-guided coordinates')
    box(13, 63, 39, 10, 'Prepare RAW-FABRID',
        'Clean reference, calibration and test frames\nReleased, fixed, donor and dense windows')
    box(58, 63, 39, 10, 'Prepare AITEX',
        'Independent clean reference and test sets\nGuided, fixed, donor and dense windows')
    box(13, 45, 84, 10, 'Score paired windows',
        'Frozen ResNet-18 global features and spatial PatchCore variant\nFuse spatial scores at native image coordinates')
    box(13, 27, 39, 10, 'Frame detection',
        'Frame AUROC and calibrated alarms\nDefective frames and clean false alarms')
    box(58, 27, 39, 10, 'Defect localization',
        'Pixel AP on observed and whole areas\nMarked-region coverage and detection')
    box(13, 5, 39, 14, 'Both datasets',
        'Paired group resampling and deletion\nEqual-budget targeted and preset rescans')
    box(58, 5, 39, 14, 'RAW-FABRID follow-up',
        'Hold out each roll in turn\nRefit spatial ResNet-18 and Swin-T')
    for x in (32.5, 77.5):
        arrow(x, 81, 73)
        arrow(x, 63, 55)
        arrow(x, 45, 37)
        arrow(x, 27, 19)
    save(fig, 'methodology')


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
