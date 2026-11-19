"""Run all manuscript analyses from the official RAW-FABRID v1 archive."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (13017, 13018, 13019)


def run(program, *args):
    command = [sys.executable, str(ROOT / 'code' / program), *map(str, args)]
    print('Running', program, *args, flush=True)
    env = dict(os.environ, MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download', action='store_true', help='Fetch the official archive when absent.')
    ap.add_argument('--device', choices=('auto','mps','cpu'), default='auto',
                    help='Backend for the original global feature extraction.')
    args = ap.parse_args()
    import torch
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'reports').mkdir(exist_ok=True)
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    run('prepare_data.py', *(['--download'] if args.download else []))
    run('audit_zip.py', 'data/RAW_FABRID.zip', 'reports/archive_audit.json')
    for script, report in [('audit_metadata.py', 'metadata_split_audit_preliminary.json'),
                           ('audit_coco.py', 'coco_audit_preliminary.json')]:
        run(script, 'data/remote_metadata', 'reports/' + report)
    for script in ('audit_original_images.py','audit_mask_samples.py','audit_patch_masks.py',
                   'audit_good_crop_alignment.py','check_release_coverage.py'):
        run(script)
    run('extract_features.py', '--device', device, '--batch-size', 16)
    run('extract_texture_features.py')
    run('extract_fixed_grid_features.py')
    for directory, stem, rebuilt_grid in [('features','baseline',False), ('texture_features','texture_baseline',False),
                                           ('features_fixed_grid','features_fixed_grid',True),
                                           ('texture_features_fixed_grid','texture_features_fixed_grid',True)]:
        for seed in SEEDS:
            run('evaluate_baseline.py', '--features', 'results/'+directory, '--seed', seed,
                '--output', f'results/{stem}_seed_{seed}.json', *(['--skip-official'] if rebuilt_grid else []))
        if rebuilt_grid:
            run('evaluate_paired_rolls.py', '--features', 'results/'+directory, '--baseline-prefix', stem+'_seed_',
                '--output', f'results/{directory}_paired.json')
    run('check_crop_publication_scope.py')
    for seed in SEEDS:
        run('localization_study.py', '--seed', seed)
    run('plot_paired_rolls.py')
    run('plot_missed_region.py')
    run('plot_manuscript.py')
    run('verify_results.py')


if __name__ == '__main__':
    main()
