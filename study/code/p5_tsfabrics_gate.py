"""Protocol 5, Amendment 1 (A7): TSFabrics data gate. Run on Kaggle with the author's TSFabrics dataset attached.
No detector, no scoring: only file structure, labels, masks and the cluster count that decides the panel.

Usage (Kaggle cell):  !python p5_tsfabrics_gate.py            (writes /kaggle/working/tsfabrics_gate.json)
Gate (frozen): at least 5 eligible independent positive test clusters, else TSFabrics is descriptive only and
MVTec AD 2 Fabric replaces it in the primary panel. Positive cluster = contains defect frames with usable masks.
"""
import csv, glob, hashlib, io, json, os, re, sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

# Frozen pilot table (tsfabrics_alignment_pilot-main/data/frozen/tsfabrics_condition_clusters.csv)
CLUSTERS = """scenario,texture,cluster,condition_group,pure_normal,defect_frames
T1_S148_I108_1,T1,C1,G1,False,747
T1_S148_I108_2,T1,C1,G1,False,167
T1_S164_I192_1,T1,C2,G2,False,10
T1_S164_I193_1,T1,C2,G2,False,10
T1_S164_I199_1,T1,C2,G2,True,0
T1_S164_I36_1,T1,C3,G3,True,0
T1_S179_I35_1,T1,C3,G3,True,0
T1_S174_I108_1,T1,C4,G1,False,153
T1_S174_I111_1,T1,C5,G1,False,6
T1_S177_I108_1,T1,C5,G1,True,0
T1_S478_I118_1,T1,C6,G4,False,248
T1_S478_I118_2,T1,C6,G4,False,117
T1_S555_I117_1,T1,C7,G4,False,1762
T2_S262_I45_1,T2,C8,G5,True,0
T2_S262_I52_1,T2,C8,G5,True,0
T2_S303_I44_1,T2,C8,G5,True,0
T2_S287_I184_1,T2,C9,G6,True,0
T2_S287_I185_1,T2,C9,G6,False,76
T2_S466_I39_1,T2,C10,G7,True,0
T3_S165_I175_1,T3,C11,G8,True,0
T3_S165_I178_1,T3,C11,G8,True,0
T3_S165_I178_2,T3,C11,G8,True,0"""


def find_root():
    for base, dirs, _ in os.walk('/kaggle/input'):
        if 'CSV' in dirs and 'IMAGE' in dirs:
            return base
    sys.exit('TSFabrics root with CSV/ and IMAGE/ not found under /kaggle/input')


def main():
    root = find_root()
    table = list(csv.DictReader(io.StringIO(CLUSTERS)))
    report = {'root': root, 'top_level': sorted(os.listdir(root)), 'scenarios': {}, 'notes': []}
    # 1. structure: any folder that looks like masks
    mask_dirs = [d for d in os.listdir(root) if re.search(r'mask|label|annot|seg|gt', d, re.I)]
    report['candidate_mask_dirs'] = mask_dirs
    for r in table:
        s = r['scenario']
        rec = {'cluster': r['cluster'], 'texture': r['texture'], 'condition_group': r['condition_group']}
        csvp = os.path.join(root, 'CSV', s + '.csv')
        if not os.path.exists(csvp):
            rec['error'] = 'missing CSV'
            report['scenarios'][s] = rec
            continue
        with open(csvp, newline='') as f:
            rows = list(csv.DictReader(f))
        labels = Counter(int(x['Label']) for x in rows)
        rec['frames'] = len(rows)
        rec['label_counts'] = dict(sorted(labels.items()))
        imgs = sorted(glob.glob(os.path.join(root, 'IMAGE', s, '**', '*.*'), recursive=True))
        rec['images_on_disk'] = len(imgs)
        if imgs:
            with Image.open(imgs[0]) as im:
                rec['image_size_wh'] = list(im.size)
                rec['image_mode'] = im.mode
        # 2. masks: ANNOTATION/<scenario>/<type>/<frame>.png; 'cutline' masks mark normal production marks, never defects
        types = {}
        for d in mask_dirs:
            sdir = os.path.join(root, d, s)
            if not os.path.isdir(sdir):
                continue
            for t in sorted(os.listdir(sdir)):
                tdir = os.path.join(sdir, t)
                if not os.path.isdir(tdir):
                    continue
                files = sorted(f for f in glob.glob(os.path.join(tdir, '**', '*.*'), recursive=True) if os.path.isfile(f))
                info = {'files': len(files)}
                if files:
                    vals, sizes, nonempty = Counter(), Counter(), 0
                    smp = files[:: max(1, len(files) // 30)][:30]
                    for m in smp:
                        arr = np.asarray(Image.open(m))
                        vals.update(np.unique(arr).tolist())
                        sizes[str(arr.shape)] += 1
                        nonempty += int((arr > 0).any())
                    info.update(values_sample=sorted(vals)[:20], shapes_sample=dict(sizes), nonempty_in_sample=nonempty,
                                sampled=len(smp), example=os.path.relpath(files[0], root))
                types[t] = info
        rec['mask_types'] = types
        rec['defect_mask_files'] = sum(v['files'] for t, v in types.items() if t.lower() != 'cutline')
        rec['defect_masks_nonempty_in_sample'] = sum(v.get('nonempty_in_sample', 0) for t, v in types.items()
                                                     if t.lower() != 'cutline')
        report['scenarios'][s] = rec
    # 3. cluster gate
    pos, neg = defaultdict(list), defaultdict(list)
    for s, rec in report['scenarios'].items():
        if 'error' in rec:
            continue
        lab = rec.get('label_counts', {})
        has_defect = any(k >= 3 for k in lab)          # pilot convention: labels 3-9 are defects, 2 = cutline
        (pos if has_defect else neg)[rec['cluster']].append(s)
    usable = {c for c, ss in pos.items() if any(report['scenarios'][s].get('defect_masks_nonempty_in_sample', 0) > 0 for s in ss)}
    report['positive_clusters_by_labels'] = sorted(pos)
    report['positive_clusters_with_usable_masks'] = sorted(usable)
    report['normal_only_clusters'] = sorted(set(neg) - set(pos))
    report['gate'] = {'rule': '>= 5 eligible independent positive test clusters',
                      'eligible_positive_clusters': len(usable),
                      'passes': len(usable) >= 5,
                      'caveat': 'eligibility also requires a texture-matched normal bank from other clusters (leave-cluster-out); '
                                'check the per-texture counts below before freezing the panel'}
    report['positive_clusters_by_texture'] = {
        t: sorted({r['cluster'] for r in report['scenarios'].values() if r['texture'] == t} & usable)
        for t in ('T1', 'T2', 'T3')}
    out = '/kaggle/working/tsfabrics_gate.json' if os.path.isdir('/kaggle/working') else 'tsfabrics_gate.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=1)
    for s_, r_ in report['scenarios'].items():
        print(s_, r_.get('cluster'), 'frames', r_.get('frames'), 'labels', r_.get('label_counts'), 'size', r_.get('image_size_wh'),
              'masks', {t: (v['files'], v.get('nonempty_in_sample'), v.get('values_sample')) for t, v in r_.get('mask_types', {}).items()})
    print(json.dumps({k: report[k] for k in ('top_level', 'candidate_mask_dirs', 'positive_clusters_by_labels',
                                             'positive_clusters_with_usable_masks', 'normal_only_clusters', 'gate',
                                             'positive_clusters_by_texture')}, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
