"""Fit and evaluate two frozen spatial detectors in the seven saved roll folds."""
import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights, Swin_T_Weights, resnet18, swin_t

from audit_zip import PUBLISHER_SHA256, sha256
from localization_study import (BATCH, DATA, FIXED, RESULTS, ROOT, TRANSFORM, fit_bank,
                                read_image, read_json, summarize_fields, tile_scores)

SEED = 13017
CHANNELS = np.random.default_rng(SEED).choice(288, 32, replace=False)
POLICIES = ('released', 'fixed')
PROTOCOL = ROOT / 'reports/unseen_roll_protocol.md'
OUT = RESULTS / 'unseen_roll'


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_json(path, obj):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        json.dump(obj, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')
    os.replace(handle.name, path)


def transformer_features(images, model, channels, device):
    x = torch.stack([TRANSFORM(im) for im in images]).to(device)
    fine = model.features[1](model.features[0](x))
    coarse = model.features[3](model.features[2](fine))
    fine = F.avg_pool2d(fine.permute(0, 3, 1, 2), 2, 2).permute(0, 2, 3, 1)
    joined = torch.cat((fine, coarse), dim=-1).index_select(-1, channels)
    return joined.reshape(len(images), 256, 32)


def fit_gaussian(samples):
    by_position = np.ascontiguousarray(samples.transpose(1, 0, 2), dtype=np.float64)
    mean = by_position.mean(axis=1)
    by_position -= mean[:, None, :]
    covariance = by_position.transpose(0, 2, 1) @ by_position / (samples.shape[0]-1)
    covariance += .01 * np.eye(samples.shape[-1])
    return mean.astype('float32'), np.linalg.inv(np.linalg.cholesky(covariance)).astype('float32')


def gaussian_scores(features, mean, inverse_chol):
    white = torch.einsum('bnc,ndc->bnd', features-mean, inverse_chol)
    return white.square().mean(dim=-1).sqrt()


def transformer_tiles(image, coords, model, channels, params, device):
    fields = []
    for start in range(0, len(coords), BATCH):
        crops = [image.crop((x, y, x+256, y+256)) for y, x in coords[start:start+BATCH]]
        features = transformer_features(crops, model, channels, device)
        fields.append(gaussian_scores(features, *params).reshape(-1, 16, 16).cpu().numpy())
    return np.concatenate(fields)


def csv_rows(archive, name):
    with archive.open('RAW_FABRID/RAW_FABRID_' + name + '_Metadata.csv') as stream:
        return list(csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig')))


def pack_hist(pair):
    total, foreground = pair
    occupied = np.flatnonzero(total)
    return {'bins': occupied.tolist(), 'total': total[occupied].tolist(),
            'foreground': foreground[occupied].tolist()}


def run_fold(archive, roll, manifest, refs, parents, meta, coords, primary_pixels,
             primary_frames, model_name, model, weights_hash, device, out_dir):
    start = time.perf_counter()
    if model_name == 'resnet18':
        bank, indices = fit_bank(archive, refs, model, device, SEED)
        fit = {'selected_indices': indices, 'bank_sha256': array_digest(bank.cpu().numpy())}
        def score(image, positions):
            return tile_scores(image, positions, model, bank, device)
    else:
        channels = torch.tensor(CHANNELS, device=device)
        chunks = []
        for i in range(0, len(refs), BATCH):
            tiles = [read_image(archive, 'MVTec/train/good/'+r['patch_name'], 'RGB')
                     for r in refs[i:i+BATCH]]
            chunks.append(transformer_features(tiles, model, channels, device).cpu().numpy())
        mean, inv = fit_gaussian(np.concatenate(chunks))
        params = tuple(torch.from_numpy(a).to(device) for a in (mean, inv))
        fit = {'channels': CHANNELS.tolist(), 'mean_sha256': array_digest(mean),
               'inverse_cholesky_sha256': array_digest(inv)}
        def score(image, positions):
            return transformer_tiles(image, positions, model, channels, params, device)
    fit_seconds = time.perf_counter()-start
    hist = {k: {p: [np.zeros(1, dtype=np.int64) for _ in range(2)] for p in POLICIES}
            for k in ('primary_pixels', 'other_clean')}
    records = {}
    for parent in sorted(parents):
        positive = meta[parent]['has_defect'].lower() == 'true'
        image = read_image(archive, 'images/'+parent, 'RGB')
        mask = (np.asarray(read_image(archive, 'masks/'+parent, 'L')) >= 128 if positive
                else np.zeros((1024, 1792), dtype=bool))
        assert image.size == (1792, 1024) and mask.shape == (1024, 1792)
        assert bool(mask.any()) == positive
        policies = {'released': coords[parent], 'fixed': FIXED}
        assert len(coords[parent]) == 28 and all(0 <= y <= 768 and 0 <= x <= 1536 for y,x in coords[parent])
        if not positive:
            assert sorted(coords[parent]) == sorted(FIXED)
        positions = list(dict.fromkeys(coords[parent] + FIXED))
        cohort = 'primary_pixels' if parent in primary_pixels else 'other_clean'
        summaries = summarize_fields(positions, score(image, positions), mask, policies, True, hist[cohort])
        records[parent] = {'defect': positive, 'primary_pixels': parent in primary_pixels,
                           'primary_frames': parent in primary_frames, 'foreground': int(mask.sum()),
                           'unseen_foreground': summaries['released']['unseen_foreground'],
                           'scores': {p: summaries[p]['frame_score'] for p in POLICIES}}
    out = {'roll': roll, 'model': model_name, 'seed': SEED, 'weights_sha256': weights_hash,
           'archive_sha256': PUBLISHER_SHA256, 'protocol_sha256': sha256(PROTOCOL),
           'manifest_sha256': sha256(manifest), 'train_parents': sorted({r['parent_raw'] for r in refs}),
           'ordered_tiles_sha256': hashlib.sha256(('\n'.join(r['patch_name'] for r in refs)+'\n').encode()).hexdigest(),
           'references': len(refs), 'device': device, 'torch': str(torch.__version__),
           'torchvision': str(torchvision.__version__), 'fit': fit, 'frames': records,
           'histograms': {k: {p: pack_hist(pair) for p, pair in v.items()} for k, v in hist.items()},
           'fit_seconds': fit_seconds, 'total_seconds': time.perf_counter()-start}
    atomic_json(out_dir / f'{model_name}_{roll}.json', out)
    print(f'{model_name} {roll}: {len(parents)} originals scored in {out["total_seconds"]:.1f}s', flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roll', help='Rebuild just this held-out roll')
    parser.add_argument('--model', choices=('resnet18', 'swin_t'))
    parser.add_argument('--output-dir', type=Path, default=OUT, help='Destination for independent reruns')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    path = DATA / 'RAW_FABRID.zip'
    if sha256(path) != PUBLISHER_SHA256:
        raise ValueError('RAW-FABRID publisher digest mismatch')
    manifest = RESULTS / 'baseline_seed_13017.json'
    folds = read_json(manifest)['folds']
    orig = read_json(RESULTS / 'localization_study.json')
    primary_pixels = set(orig['defective_frames']) | set(orig['evaluation_clean'])
    primary_frames = primary_pixels | set(orig['calibration_clean'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        meta = {r['filename']: r for r in csv_rows(archive, 'HighRes')}
        patches = csv_rows(archive, 'Patches')
        assert len(meta) == 709 and len(primary_pixels) == 255 and len(primary_frames) == 303
        assert set(orig['defective_frames']) == {p for p,r in meta.items() if r['has_defect'].lower() == 'true'}
        coords = defaultdict(list)
        for row in json.loads(archive.read('RAW_FABRID/RAW_FABRID_Patches_COCO.json'))['images']:
            coords[row['parent']].append((row['r_offset'],row['c_offset']))
        rolls = sorted({r['roll_id'] for r in meta.values()})
        if args.roll and args.roll not in rolls:
            parser.error('Unknown roll: '+args.roll)
        for name, factory, weights in [('resnet18', resnet18, ResNet18_Weights.IMAGENET1K_V1),
                                       ('swin_t', swin_t, Swin_T_Weights.IMAGENET1K_V1)]:
            if args.model and args.model != name:
                continue
            model = factory(weights=weights).eval().to(device)
            checkpoint = Path(torch.hub.get_dir()) / 'checkpoints' / weights.url.split('/')[-1]
            digest = sha256(checkpoint)
            if not digest.startswith(checkpoint.stem.split('-')[-1]):
                raise ValueError('Unexpected checkpoint digest')
            for roll in rolls:
                if args.roll and args.roll != roll:
                    continue
                fold = folds['held_out_'+roll]
                ids = fold['train_patch_indices']
                refs = [patches[i] for i in ids]
                parents = set(fold['test_parents'])
                assert len(ids) == len(set(ids)) == 1000
                assert parents == {p for p, r in meta.items() if r['roll_id'] == roll}
                assert all(r['type'] == 'train_good' and meta[r['parent_raw']]['roll_id'] != roll
                           and meta[r['parent_raw']]['has_defect'].lower() == 'false' for r in refs)
                run_fold(archive, roll, manifest, refs, parents, meta, coords, primary_pixels,
                         primary_frames, name, model, digest, device, args.output_dir)

if __name__ == '__main__':
    main()
