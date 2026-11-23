"""Label-only cut audit of AITEX on the frozen Protocol 5 cohort (manifests/aitex_manifest.json), with the original
decoder (>=128, masks merged) and the original floor-division guided rule. Replaces the rounding-variant values.
Usage: python3 p5_cut_audit_aitex.py <aitex dir> <out.json>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_core as C
from cut_audit import audit_frame, summarise

data, out = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((Path(__file__).resolve().parents[1] / 'manifests' / 'aitex_manifest.json').read_text())
A, B = [], []
for n, r in sorted(man['images'].items()):
    if r['role'] != 'evaluation_defective':
        continue
    m = C.decode_aitex([data / x['path'] for x in r['masks']])
    A.append(audit_frame(m, C.aitex_original_layout(m)))
    B.append(audit_frame(m, C.grid_layout(256, 4096)))
res = {'cohort': 'Protocol 5 AITEX manifest, 103 defective strips', 'decoder': '>=128, masks merged',
       'A_original_floor_rule': summarise(A), 'B_blind_16_grid': summarise(B)}
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
