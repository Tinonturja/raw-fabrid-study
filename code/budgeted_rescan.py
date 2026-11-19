"""Test label-independent, equal-budget rescanning from the original fabric images."""
import argparse
import csv
import hashlib
import io
import json
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

import check_crop_publication_scope as raw
import external_aitex as aitex
from audit_zip import PUBLISHER_SHA256, sha256
from localization_study import (BATCH, ROOT, embeddings, fit_bank, pixel_ap, read_image,
                                read_json, select_bank, summarize_fields, summarize_policy,
                                tile_scores)

SEED = 13017
POLICIES = ('fixed', 'targeted', 'even', 'overlap')


def select_extras(fixed, base_fields, candidates, budget):
    """Rank candidate footprints using only the first-pass score lattice."""
    height, width = (max(point[i] for point in fixed) // 16 + 16 for i in (0, 1))
    grid = np.empty((height, width), dtype=base_fields.dtype)
    for (y, x), field in zip(fixed, base_fields, strict=True):
        grid[y//16:y//16+16, x//16:x//16+16] = field

    def priority(point):
        y, x = (value // 16 for value in point)
        footprint = grid[y:y+16, x:x+16].ravel()
        return -float(np.partition(footprint, -4)[-4:].mean(dtype=np.float64)), point

    return sorted(candidates, key=priority)[:budget]


def evenly_spaced(candidates, budget):
    return [candidates[(2*i+1)*len(candidates)//(2*budget)] for i in range(budget)]


def empty_histograms():
    return {k: [np.zeros(1, np.int64), np.zeros(1, np.int64)] for k in POLICIES}


def pool_histograms(histograms):
    pairs = list(histograms)
    size = max(len(pair[0]) for pair in pairs)
    total = np.zeros((2, size), dtype=np.int64)
    for counts, fg in pairs:
        total[:, :len(counts)] += np.array([counts, fg])
    return total


def compare(study, original, groups, frames, bank, selected, model, fixed, dense, device):
    candidates = sorted(set(dense) - set(fixed))
    budget = len(fixed) // 4
    even = evenly_spaced(candidates, budget)
    cal, clean = (set(original[k]) for k in ('calibration_clean', 'evaluation_clean'))
    positive = set(original['defective_frames' if study == 'raw' else 'defective'])
    records, selections = {}, {}
    by_group = defaultdict(empty_histograms)
    for index, (name, image, mask) in enumerate(frames, 1):
        base = tile_scores(image, fixed, model, bank, device)
        chosen = select_extras(fixed, base, candidates, budget)
        # Score these K extras first, just as a scanner running only the targeted
        # policy would. Score the remaining windows separately for the controls.
        other = sorted(set(candidates) - set(chosen))
        extra = tile_scores(image, chosen, model, bank, device)
        rest = tile_scores(image, other, model, bank, device)
        coords = fixed + chosen + other
        layouts = dict(fixed=fixed, targeted=fixed+chosen, even=fixed+even, overlap=dense)
        records[name] = summarize_fields(coords, np.concatenate((base, extra, rest)), mask,
                                         layouts, name not in cal, by_group[groups[name]])
        selections[name] = chosen
        if index % 50 == 0:
            print(f'{study}: {index} frames scored', flush=True)
    assert set(records) == cal | clean | positive
    metrics, deletions = {}, {}
    for policy in POLICIES:
        hist = pool_histograms(group[policy] for group in by_group.values())
        metrics[policy] = summarize_policy(policy, records, hist, positive, cal, clean, groups)
        assert hist[0, 0] == 0 and hist[0].sum() == (len(positive)+len(clean))*len(fixed)*256**2
    for held in sorted(by_group):
        deletions[held] = {}
        for policy in ('targeted', 'even'):
            pooled = pool_histograms(hist[policy] for group, hist in by_group.items() if group != held)
            deletions[held][policy] = pixel_ap(*pooled)
        deletions[held]['difference'] = deletions[held]['targeted'] - deletions[held]['even']
    baseline = original['metrics' if study == 'raw' else 'spatial_metrics']
    old_frames = original['frames' if study == 'raw' else 'spatial_frames']
    control_checks = {}
    for policy in ('fixed', 'overlap'):
        control_checks[policy] = {
            'max_abs_frame_score_change': max(abs(records[p][policy]['frame_score'] -
                                                old_frames[p][policy]['frame_score']) for p in records),
            'auroc_change': metrics[policy]['auroc'] - baseline[policy]['auroc'],
            'whole_frame_ap_change': metrics[policy]['pixel_ap_all'] - baseline[policy]['pixel_ap_all'],
            'region_hits_change': metrics[policy]['regions_hit'] - baseline[policy]['regions_hit'],
            'false_alarms_change': metrics[policy]['false_alarms'] - baseline[policy]['false_alarms']}
    return {
        'dataset': study, 'seed': SEED, 'device': str(bank.device), 'torch': str(torch.__version__),
        'protocol_sha256': sha256(ROOT/'reports/rescan_protocol.md'),
        'bank_sha256': hashlib.sha256(bank.cpu().numpy().tobytes()).hexdigest(),
        'original_coreset_indices_match': selected == original['selected_spatial_indices'],
        'windows': dict(fixed=len(fixed), targeted=len(fixed)+budget, even=len(fixed)+budget,
                        overlap=len(dense)),
        'original_control_checks': control_checks, 'even_extra_yx': even,
        'targeted_extra_yx': selections, 'calibration_clean': sorted(cal),
        'evaluation_clean': sorted(clean), 'defective': sorted(positive), 'group': groups,
        'metrics': metrics, 'omit_one_group_ap': deletions, 'frames': records,
        'group_histograms': {g: {k: [a.tolist(), b.tolist()] for k, (a, b) in h.items()}
                             for g, h in by_group.items()}}


def run_raw(model, device):
    original = read_json(raw.RESULTS/'localization_study.json')
    archive_path = raw.DATA/'RAW_FABRID.zip'
    if sha256(archive_path) != PUBLISHER_SHA256:
        raise ValueError('The RAW-FABRID archive fingerprint differs from the publisher record.')
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open('RAW_FABRID/RAW_FABRID_HighRes_Metadata.csv') as f:
            meta = {r['filename']: r['roll_id'] for r in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig'))}
        names = sorted(original['frames'])
        groups = {p: meta[p] for p in names}
        positive = set(original['defective_frames'])
        refs = [{'patch_name': p} for p in original['reference_tiles']]
        bank, selected = fit_bank(archive, refs, model, device, SEED)

        def frames():
            for p in names:
                image = read_image(archive, 'images/'+p, 'RGB')
                mask = (np.asarray(read_image(archive, 'masks/'+p, 'L')) >= 128 if p in positive
                        else np.zeros((image.height, image.width), dtype=bool))
                yield p, image, mask

        return compare('raw', original, groups, frames(), bank, selected, model, raw.FIXED, raw.OVERLAP, device)


def run_aitex(model, device):
    original = read_json(raw.RESULTS/'aitex_external.json')
    with tempfile.TemporaryDirectory(prefix='rescan-aitex-') as temp:
        samples, _ = aitex.audit_sources(aitex.prepare_archives(False, Path(temp)))
        if {p: r['hash'] for p, r in samples.items()} != original['source_hashes']:
            raise ValueError('AITEX source hashes differ from the original run.')
        refs = original['reference_tiles']
        vectors = []
        for i in range(0, len(refs), BATCH):
            batch = []
            for p, x in refs[i:i+BATCH]:
                with Image.open(samples[p]['path']) as im:
                    batch.append(im.convert('L').crop((x, 0, x+256, 256)).convert('RGB'))
            vectors.append(embeddings(batch, model, device).reshape(-1, 256).cpu())
        bank, selected = select_bank(torch.cat(vectors), device, SEED)
        del vectors
        names = sorted(original['spatial_frames'])
        groups = {p: samples[p]['fabric'] for p in names}
        frames = ((p, Image.fromarray(aitex.gray(samples[p]['path'])).convert('RGB'),
                   aitex.source_mask(samples[p])) for p in names)
        return compare('aitex', original, groups, frames, bank, selected, model, aitex.FIXED, aitex.OVERLAP, device)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('raw', 'aitex'))
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'),
                        default='mps' if torch.backends.mps.is_available() else
                        'cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(f'Running {args.dataset} on {args.device}', flush=True)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(args.device)
    result = run_raw(model, args.device) if args.dataset == 'raw' else run_aitex(model, args.device)
    target = raw.RESULTS/f'budgeted_rescan_{args.dataset}.json'
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, mode='w') as out:
            pending = Path(out.name)
            json.dump(result, out, allow_nan=False, separators=(',', ':'))
            out.write('\n')
        pending.replace(target)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    for policy, m in result['metrics'].items():
        print(policy, m['pixel_ap_all'], m['auroc'], m['regions_hit'], m['false_alarms'])


if __name__ == '__main__':
    main()
