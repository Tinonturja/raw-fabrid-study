"""Measure how much marked fabric the two saved second-pass layouts revisit."""
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

import external_aitex as aitex
from audit_zip import PUBLISHER_SHA256, sha256
from localization_study import ROOT, read_image, read_json


def measure(study, get_mask):
    evidence = ROOT/f'results/budgeted_rescan_{study}.json'
    result = read_json(evidence)
    frames = {}
    for p in result['defective']:
        mask = get_mask(p)
        counts = {'foreground': int(mask.sum())}
        for policy, locations in [('targeted', result['targeted_extra_yx'][p]),
                                  ('even', result['even_extra_yx'])]:
            union = np.zeros_like(mask, dtype=bool)
            touched = 0
            for y, x in locations:
                union[y:y+256, x:x+256] = True
                touched += int(mask[y:y+256, x:x+256].any())
            counts[policy] = {'marked_pixels_revisited': int(mask[union].sum()),
                              'windows_touching_foreground': touched}
        frames[p] = counts
    total = sum(r['foreground'] for r in frames.values())
    original = read_json(ROOT/'results'/('localization_study.json' if study == 'raw' else 'aitex_external.json'))
    assert total == sum(original['histograms']['fixed']['foreground'])
    summary = {}
    for policy in ('targeted', 'even'):
        masses = [r[policy]['marked_pixels_revisited'] for r in frames.values()]
        summary[policy] = {
            'frames_with_marked_pixel_revisited': sum(count > 0 for count in masses),
            'revisited_pixels': sum(masses), 'fraction_of_marked_pixels': sum(masses)/total,
            'mean_within_frame_fraction': float(np.mean([r[policy]['marked_pixels_revisited']/r['foreground']
                                                         for r in frames.values()])),
            'windows_touching_foreground': sum(r[policy]['windows_touching_foreground'] for r in frames.values())}
    return {'rescan_scores_sha256': sha256(evidence), 'positive_frames': len(frames),
            'total_foreground': total, 'extra_windows_on_positive_frames': len(frames)*len(result['even_extra_yx']),
            'summary': summary, 'frames': frames}


def main():
    archive_path = ROOT/'data/RAW_FABRID.zip'
    if sha256(archive_path) != PUBLISHER_SHA256:
        raise ValueError('Unexpected RAW-FABRID source archive')
    with zipfile.ZipFile(archive_path) as archive:
        raw = measure('raw', lambda p: np.asarray(read_image(archive, 'masks/'+p, 'L')) >= 128)
    with tempfile.TemporaryDirectory(prefix='rescan-mask-') as temp:
        samples, _ = aitex.audit_sources(aitex.prepare_archives(False, Path(temp)))
        external = measure('aitex', lambda p: aitex.source_mask(samples[p]))
    report = {'analysis': 'Exploratory coverage diagnostic chosen after the two rescan outcomes',
              'raw': raw, 'aitex': external}
    target = ROOT/'results/rescan_coverage.json'
    with tempfile.NamedTemporaryFile(dir=target.parent, mode='w', delete=False) as out:
        json.dump(report, out, indent=1, allow_nan=False)
        out.write('\n')
        pending = Path(out.name)
    pending.replace(target)
    for name in ('raw', 'aitex'):
        print(name, report[name]['summary'])


if __name__ == '__main__':
    main()
