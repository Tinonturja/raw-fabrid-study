"""Print every numeric field that changed between the pre-fix and post-fix analysis files."""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results_p5'
pairs = [(R / '_before_audit_fix' / f'{d}_analysis.json', R / d / 'full' / 'analysis.json')
         for d in ('aitex', 'rawfabrid', 'mvtec_fabric', 'visa', 'e1_dino_size')]

def walk(a, b, path, out):
    if isinstance(a, dict):
        for k in a:
            if k in b: walk(a[k], b[k], path + [str(k)], out)
            else: out.append(('/'.join(path + [str(k)]), 'missing in new', ''))
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, path + [str(i)], out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > float(sys.argv[1] if len(sys.argv) > 1 else 5e-5): out.append(('/'.join(path), a, b))

for old, new in pairs:
    print('=====', new.parent.parent.name)
    out = []
    walk(json.loads(old.read_text()), json.loads(new.read_text()), [], out)
    print(len(out), 'changed fields')
    for p, a, b in out[:400]:
        print(f'  {p}: {a} -> {b}')
