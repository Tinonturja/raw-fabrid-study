"""Recover group-level guided pixel counts from the frozen primary experiments."""
import json
import zipfile

import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

from budgeted_rescan import run_aitex, run_raw, pool_histograms
from localization_study import ROOT, score_parent


def evaluate(study, original, groups, frames, bank, selected, model, fixed, dense, device):
    if selected != original['selected_spatial_indices']:
        raise ValueError('Refitted bank differs from the recorded primary run.')
    if study == 'raw':
        with zipfile.ZipFile(ROOT/'data/RAW_FABRID.zip') as z:
            rows = json.loads(z.read('RAW_FABRID/RAW_FABRID_Patches_COCO.json'))['images']
        coords = {}
        for row in rows:
            coords.setdefault(row['parent'], []).append((row['r_offset'], row['c_offset']))
    else:
        coords = original['guided_coordinates']
    cal = set(original['calibration_clean'])
    hist = {g: {'released': [np.zeros(1, np.int64), np.zeros(1, np.int64)]}
            for g in sorted(set(groups.values()))}
    for name, image, mask in frames:
        if name in cal:
            continue
        layout = [tuple(pt) for pt in coords.get(name, fixed)]
        score_parent(image, mask, {'released':layout}, model, bank, device, True, hist[groups[name]])
    pooled = pool_histograms(item['released'] for item in hist.values())
    expected = original['histograms']['released']
    for values, key in zip(pooled, ('all', 'foreground')):
        actual = np.trim_zeros(values, 'b')
        reference = np.trim_zeros(expected[key], 'b')
        if not np.array_equal(actual, reference):
            raise ValueError(f'{study}: reconstructed {key} histogram differs from saved counts')
    print(study, 'all native pixel bins agree with the original run', flush=True)
    return {g: [h.tolist() for h in item['released']] for g, item in hist.items()}


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    data = {key: fn(model, device, evaluate) for key,fn in [('raw',run_raw),('aitex',run_aitex)]}
    output = ROOT/'results/guided_group_histograms.json'
    tmp = output.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, separators=(',', ':'))+'\n')
    tmp.replace(output)


if __name__ == '__main__':
    main()
