"""Independently verify saved low-FPR region curves and primary score bins."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'results/region_followup'


def area(bg, mass, regions, limit=.05):
    x=0.
    y=0.
    integral=0.
    for b,m in zip(bg[::-1],mass[::-1]):
        next_x=x+b/sum(bg)
        next_y=y+m/regions
        if next_x>limit:
            y_at_limit=y+(next_y-y)*(limit-x)/(next_x-x)
            integral+=(limit-x)*(y+y_at_limit)/2
            return integral/limit
        integral+=(next_x-x)*(y+next_y)/2
        x,y=next_x,next_y
    return integral/limit


def main():
    paths=sorted(folder.glob('*_Rollo*.json'))
    assert len(paths)==14
    for path in paths:
        d=json.loads(path.read_text())
        orig=json.loads((ROOT/'results/unseen_roll'/path.name).read_text())
        for policy,v in d['policies'].items():
            background=np.asarray(v['background'])
            mass=np.asarray(v['region_mass'])
            value=area(background,mass,d['regions'])
            assert abs(value-v['aupro_005'])<1e-10,(path,policy,value,v['aupro_005'])
            assert abs(mass.sum()-d['regions'])<1e-6
            original=orig['histograms']['primary_pixels'][policy]
            assert int(background.sum())==sum(original['total'])-sum(original['foreground'])
    for background,mass,expect in [([100,0],[0,1],1),([0,100],[1,0],0),([0,100],[0,1],.025)]:
        assert abs(area(background,mass,1)-expect)<1e-12
    print('Verified 14 held-out fits, two policies, background counts, and independent AU-PRO integration.')


if __name__=='__main__':
    main()
