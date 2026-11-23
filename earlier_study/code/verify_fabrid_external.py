"""Check FabriD test scores, counts, and low-FPR overlap from saved evidence."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from verify_region_followup import area

ROOT=Path(__file__).resolve().parents[1]
result=json.loads((ROOT/'results/fabrid_external.json').read_text())
frames=result['frames']
assert len(frames)>0 and len({r['source_id'] for r in frames.values()})==len(frames)
labels=[r['defect'] for r in frames.values()]
foreground=sum(r['foreground'] for r in frames.values())
for model,policies in result['models'].items():
    for policy,v in policies.items():
        h=v['histogram']
        bins=np.array(h['bins'])
        total=np.array(h['total'])
        fg=np.array(h['foreground'])
        assert np.all(total>=fg) and fg.sum()==foreground
        counts=np.r_[fg,total-fg]
        truth=np.r_[np.ones(len(bins)),np.zeros(len(bins))]
        scores=np.tile(bins,2)
        full=average_precision_score(truth,scores,sample_weight=counts)
        keep=bins>0
        observed=average_precision_score(np.r_[truth[:len(bins)][keep],truth[len(bins):][keep]],
                                         np.tile(bins[keep],2),
                                         sample_weight=np.r_[fg[keep],(total-fg)[keep]])
        assert abs(full-v['whole_ap'])<1e-12
        assert abs(observed-v['observed_ap'])<1e-12
        assert abs(roc_auc_score(labels,[r['scores'][model][policy] for r in frames.values()])-v['frame_auroc'])<1e-12
        width=int(bins[-1])+1
        bg=np.zeros(width);mass=np.zeros(width)
        bg[bins]=total-fg
        mass[bins]=h['region_mass']
        assert abs(mass.sum()-result['regions'])<1e-6
        assert abs(area(bg,mass,result['regions'])-v['aupro_005'])<1e-10
        unseen=sum(r['unseen_foreground'][policy] for r in frames.values())
        assert unseen==v['unseen_foreground']
print('Verified frame AUROC, weighted AP, region integration and coverage for both models and policies.')
