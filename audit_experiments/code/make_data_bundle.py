"""Build the versioned numerical data bundle that accompanies a GitHub release.

Packs the per-image score histograms (.npz), per-image records (.json) and analysis files of Protocols 1 to 5 and
the follow-up analyses E1 and E2 into four zip files, with a SHA-256 list of every packed file and of the zips.
Excluded: memory banks and model caches (*.pt; rebuilt by the code, their hashes are in the run logs), smoke-test
outputs, temporary files, and anything holding source images (results_p5/qualitative), which stay with their
dataset custodians under their licences.

Usage (from audit_experiments/): python code/make_data_bundle.py --version v1.0.0 --out <dir>
"""
import argparse, hashlib, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BUNDLES = {
    'numerical_p1_p4': ['results', 'results_cut', 'results_p2', 'results_p3', 'results_p4'],
    'numerical_p5_core': ['results_p5/aitex', 'results_p5/aitex_seeds', 'results_p5/rawfabrid', 'results_p5/tsfabrics',
                          'results_p5/mvtec_fabric', 'results_p5/fragmentation', 'results_p5/_before_audit_fix',
                          'results_p5/_before_frag_fix'],
    'numerical_p5_visa': ['results_p5/visa'],
    'numerical_followups_e1_e2': ['results_p5/e1_dino_size', 'results_p5/e2_bank_layout'],
}
SKIP_PARTS = {'smoke', '__pycache__', 'banks', 'feature_cache'}


def keep(p):
    rel = p.relative_to(HERE)
    return (p.is_file() and p.suffix != '.pt' and '.tmp' not in p.name and not p.name.startswith('.')
            and not (SKIP_PARTS & set(rel.parts)))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines, zips = [], []
    for name, roots in BUNDLES.items():
        files = sorted(f for r in roots if (HERE / r).exists() for f in (HERE / r).rglob('*') if keep(f))
        zp = out / f'raw-fabrid-study_{a.version}_{name}.zip'
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in files:
                rel = f.relative_to(HERE).as_posix()
                info = zipfile.ZipInfo(rel, date_time=(2026, 11, 23, 0, 0, 0))   # fixed time: same content, same zip
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, f.read_bytes())
                lines.append(f'{sha(f)}  {zp.name}:{rel}')
        zips.append(f'{sha(zp)}  {zp.name}')
        print(f'{zp.name}: {len(files)} files, {zp.stat().st_size / 2**20:.1f} MiB')
    (out / f'raw-fabrid-study_{a.version}_SHA256SUMS.txt').write_text(
        '# zip files\n' + '\n'.join(zips) + '\n# packed files (zip:path)\n' + '\n'.join(lines) + '\n')
    print('wrote', out / f'raw-fabrid-study_{a.version}_SHA256SUMS.txt')


if __name__ == '__main__':
    main()
