"""Run the fixed spatial-detector sensitivity analysis on RAW-FABRID v1."""
import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from check_crop_publication_scope import DATA, DONORS, FIXED, OVERLAP, POLICIES, RESULTS

BATCH = 16
QUANTIZATION = 1000
ROOT = Path(__file__).resolve().parents[1]
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


def read_json(path):
    return json.loads(path.read_text())


def read_image(archive, member, mode):
    with Image.open(io.BytesIO(archive.read('RAW_FABRID/' + member))) as im:
        return im.convert(mode)


def embeddings(images, backbone, device):
    batch = torch.stack([TRANSFORM(im) for im in images]).to(device)
    x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(batch))))
    layer2 = backbone.layer2(backbone.layer1(x))
    layer3 = backbone.layer3(layer2)
    fine = F.avg_pool2d(layer2, 3, 1, 1)
    # Averaging adjacent channel pairs after the 3x3 neighborhood mean gives
    # the 128-value adaptive average used for the 256-channel deep layer.
    coarse = F.avg_pool2d(layer3, 3, 1, 1)
    coarse = coarse.reshape(len(images), 128, 2, *coarse.shape[-2:]).mean(2)
    coarse = F.interpolate(coarse, size=fine.shape[-2:], mode='bilinear', align_corners=False)
    return torch.cat([fine, coarse], dim=1).permute(0, 2, 3, 1).contiguous()


def select_bank(vectors, device, seed):
    generator = torch.Generator().manual_seed(seed)
    projection = torch.empty(256, 64).uniform_(-1 / 16, 1 / 16, generator=generator)
    projected = (vectors @ projection).to(device)
    norms = projected.square().sum(1)
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(vectors), 10, replace=False).tolist()

    def distances_to(indices):
        anchors = projected[indices]
        squared = norms[:, None] + norms[indices][None, :] - 2 * (projected @ anchors.T)
        return squared.clamp_min_(0).sqrt_()

    closest = distances_to(starts).mean(1)
    selected = []
    for _ in range(256):
        idx = int(closest.argmax().item())
        selected.append(idx)
        closest = torch.minimum(closest, distances_to([idx])[:, 0])
        closest[idx] = -1.0
    assert len(set(selected)) == 256
    return vectors[selected].to(device), selected


def fit_bank(archive, refs, backbone, device, seed):
    chunks = []
    for start in range(0, len(refs), BATCH):
        images = [read_image(archive, 'MVTec/train/good/' + r['patch_name'], 'RGB')
                  for r in refs[start:start+BATCH]]
        chunks.append(embeddings(images, backbone, device).reshape(-1, 256).cpu())
    vectors = torch.cat(chunks)
    del chunks
    return select_bank(vectors, device, seed)


def tile_scores(image, coordinates, backbone, bank, device):
    fields = []
    norms = bank.square().sum(1)
    for start in range(0, len(coordinates), BATCH):
        crops = [image.crop((x, y, x+256, y+256)) for y, x in coordinates[start:start+BATCH]]
        features = embeddings(crops, backbone, device)
        flat = features.reshape(-1, 256)
        distances = flat.square().sum(1, keepdim=True) + norms - 2 * (flat @ bank.T)
        nearest = distances.amin(1).clamp_min_(0).sqrt_()
        fields.append(nearest.reshape(features.shape[:-1]).cpu().numpy())
    return np.concatenate(fields)


def pixel_histogram(counts, fg_counts, bins, totals):
    for row, weights in enumerate((counts, fg_counts)):
        sums = np.bincount(bins, weights=weights).astype(np.int64)
        if totals[row].size < sums.size:
            totals[row] = np.pad(totals[row], (0, sums.size - totals[row].size))
        totals[row][:len(sums)] += sums


def score_parent(image, mask, policies, backbone, bank, device, accumulate, histograms):
    coords = list(dict.fromkeys(pos for positions in policies.values() for pos in positions))
    fields = tile_scores(image, coords, backbone, bank, device)
    values, ranks = np.unique(fields, return_inverse=True)
    if len(values) >= 65535 or not np.isfinite(values).all():
        raise ValueError('Unexpected score scale or number of ranks')
    ranks = (ranks.reshape(fields.shape) + 1).astype(np.uint16)
    levels = np.r_[-1.0, values]
    bins = np.r_[0, np.floor(values * QUANTIZATION).astype(np.int64) + 1]
    positions = {pos: i for i, pos in enumerate(coords)}
    labels, n_regions = label(mask, structure=np.ones((3, 3), dtype=bool))
    region_ids = labels[mask]
    summaries = {}
    seen_layouts = {}
    for name, locations in policies.items():
        key = tuple(sorted(locations))
        if key in seen_layouts:
            summary, total_counts, fg_counts = seen_layouts[key]
        else:
            field = np.zeros(mask.shape, dtype=np.uint16)
            for y, x in locations:
                window = ranks[positions[y, x]].repeat(16, axis=0).repeat(16, axis=1)
                out = field[y:y+256, x:x+256]
                np.maximum(out, window, out=out)
            fg_ranks = field[mask]
            total_counts = np.bincount(field.ravel(), minlength=len(levels))
            fg_counts = np.bincount(fg_ranks, minlength=len(levels))
            region_max = np.zeros(n_regions+1, dtype=np.uint16)
            np.maximum.at(region_max, region_ids, fg_ranks)
            summary = {'frame_score': float(levels[field.max()]),
                       'region_maxima': levels[region_max[1:]].tolist(),
                       'unseen_pixels': int(total_counts[0]), 'unseen_foreground': int(fg_counts[0])}
            seen_layouts[key] = summary, total_counts, fg_counts
        summaries[name] = summary
        if accumulate:
            pixel_histogram(total_counts, fg_counts, bins, histograms[name])
    return summaries


def pixel_ap(total, positive):
    tp = np.cumsum(positive[::-1], dtype=np.float64)
    n = np.cumsum(total[::-1], dtype=np.float64)
    precision = tp / np.maximum(n, 1)
    return float(np.dot(positive[::-1], precision) / positive.sum())


def summarize_policy(policy, records, hist, positive, cal, clean_test, rolls):
    score = {p: v[policy]['frame_score'] for p, v in records.items()}
    cutoff = float(np.quantile([score[p] for p in cal], .95, method='higher'))
    region_scores = [s for p in positive for s in records[p][policy]['region_maxima']]
    total, fg = hist
    result = {'auroc': float(roc_auc_score([p in positive for p in records], list(score.values()))),
              'cutoff': cutoff, 'defects_flagged': sum(score[p] > cutoff for p in positive),
              'false_alarms': sum(score[p] > cutoff for p in clean_test),
              'regions_hit': sum(s > cutoff for s in region_scores), 'regions': len(region_scores),
              'unseen_foreground': int(fg[0]), 'unseen_pixels': int(total[0]),
              'pixel_ap_all': pixel_ap(total, fg), 'pixel_ap_observed': pixel_ap(total[1:], fg[1:]),
              'per_roll': {}}
    for roll in sorted(set(rolls.values())):
        group = [p for p in score if rolls[p] == roll]
        result['per_roll'][roll] = {'frames': len(group), 'defects': sum(p in positive for p in group),
                                    'auroc': float(roc_auc_score([p in positive for p in group],
                                                                [score[p] for p in group]))}
    # Merge adjacent distance bins, keeping the uncovered-pixel bin separate.
    pairs = np.r_[0, 1 + np.arange(len(total)-1)//2]
    coarse_total = np.bincount(pairs, weights=total)
    coarse_fg = np.bincount(pairs, weights=fg)
    result['pixel_ap_all_width_0.002'] = pixel_ap(coarse_total, coarse_fg)
    result['pixel_ap_observed_width_0.002'] = pixel_ap(coarse_total[1:], coarse_fg[1:])
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', choices=(13017, 13018, 13019), type=int, default=13017)
    seed = parser.parse_args().seed
    start_time = time.perf_counter()
    torch.set_num_threads(4)
    device = ('mps' if torch.backends.mps.is_available() else
              'cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    old = read_json(RESULTS / 'crop_publication_scope.json')
    run = old['runs'][str(seed)]
    baseline = read_json(RESULTS / f'baseline_seed_{seed}.json')['folds']['clean_image_holdout']
    with (DATA / 'remote_metadata/RAW_FABRID_Patches_Metadata.csv').open(encoding='utf-8-sig', newline='') as f:
        patch_rows = list(csv.DictReader(f))
    with (DATA / 'remote_metadata/RAW_FABRID_HighRes_Metadata.csv').open(encoding='utf-8-sig', newline='') as f:
        meta = {row['filename']: row for row in csv.DictReader(f)}
    coords = defaultdict(list)
    for im in read_json(DATA / 'remote_metadata/RAW_FABRID_Patches_COCO.json')['images']:
        coords[im['parent']].append((im['r_offset'], im['c_offset']))
    parents = sorted(run['scores'])
    positive = {p for p in parents if meta[p]['has_defect'].lower() == 'true'}
    cal, clean_test = set(run['calibration_clean']), set(run['evaluation_clean'])
    assert cal.isdisjoint(clean_test) and positive.isdisjoint(cal | clean_test)
    assert len(positive) == 202 and len(cal) == 48 and len(clean_test) == 53
    assert set(parents) == positive | cal | clean_test
    refs = [patch_rows[i] for i in baseline['train_patch_indices']]
    assert len(refs) == 1000 and all(r['type'] == 'train_good' and r['parent_raw'] not in run['scores'] for r in refs)
    hist = {k: [np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)] for k in POLICIES}
    records = {}
    with zipfile.ZipFile(DATA / 'RAW_FABRID.zip') as archive:
        bank, selected = fit_bank(archive, refs, model, device, seed)
        fit_seconds = time.perf_counter() - start_time
        print(f'Fitted 256-vector coreset in {fit_seconds:.1f} seconds', flush=True)
        for i, p in enumerate(parents, 1):
            image = read_image(archive, 'images/'+p, 'RGB')
            mask = (np.asarray(read_image(archive, 'masks/'+p, 'L')) >= 128 if p in positive
                    else np.zeros((1024, 1792), dtype=bool))
            policies = {'released': coords[p], 'fixed': FIXED, 'overlap': OVERLAP}
            for k in DONORS:
                policies[k] = coords[old['donor_layouts'][p][k]] if p in positive else FIXED
            records[p] = score_parent(image, mask, policies, model, bank, device, p not in cal, hist)
            assert records[p]['released']['unseen_foreground'] == old['geometry'].get(p, {'released': {'unseen_mask_pixels': 0}})['released']['unseen_mask_pixels']
            if i % 40 == 0:
                print(f'Scored {i}/{len(parents)} frames in {time.perf_counter()-start_time:.1f} seconds', flush=True)
    metrics = {}
    for name in POLICIES:
        total, fg = hist[name]
        assert total.sum() == (len(positive)+len(clean_test))*1792*1024 and fg.sum() == 1904926
        metrics[name] = summarize_policy(name, records, hist[name], positive, cal, clean_test,
                                         {p: meta[p]['roll_id'] for p in parents})
    out = {'analysis': 'Exploratory PatchCore variant', 'seed': seed, 'device': device,
           'protocol_sha256': hashlib.sha256((ROOT/'reports/localization_protocol.md').read_bytes()).hexdigest(),
           'bank_sha256': hashlib.sha256(bank.cpu().numpy().tobytes()).hexdigest(),
           'reference_tiles': [r['patch_name'] for r in refs], 'selected_spatial_indices': selected,
           'calibration_clean': sorted(cal), 'evaluation_clean': sorted(clean_test),
           'defective_frames': sorted(positive), 'fit_seconds': fit_seconds,
           'total_seconds': time.perf_counter() - start_time, 'metrics': metrics, 'frames': records,
           'histograms': {k: {'all': v[0].tolist(), 'foreground': v[1].tolist()} for k,v in hist.items()}}
    target = RESULTS / ('localization_study.json' if seed == 13017 else f'localization_study_{seed}.json')
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    temp.replace(target)
    for name, m in metrics.items():
        print(name, {k: m[k] for k in ('auroc', 'pixel_ap_all', 'pixel_ap_observed', 'regions_hit', 'regions', 'false_alarms')}, flush=True)


if __name__ == '__main__':
    main()
