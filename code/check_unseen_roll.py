"""Check Gaussian scoring, native pixel fusion, and optional image-based fold repeats."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from unseen_roll import DATA, FIXED, OUT, fit_gaussian, gaussian_scores, read_image, summarize_fields


def check_gaussian():
    rng = np.random.default_rng(21)
    mixing = rng.normal(size=(8,8))
    samples = (rng.normal(size=(63,4,8)) @ mixing).astype('float32')
    observations = (rng.normal(size=(4,4,8)) @ mixing).astype('float32')
    mean, inv = fit_gaussian(samples)
    expected = []
    for row in observations:
        distances = []
        for i, v in enumerate(row):
            delta = v - samples[:, i].mean(axis=0)
            covariance = np.cov(samples[:,i].T) + .01*np.eye(8)
            distances.append(np.sqrt(delta @ np.linalg.solve(covariance,delta) / 8))
        expected.append(distances)
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        actual = gaussian_scores(*(torch.tensor(a,device=device) for a in (observations,mean,inv)))
        np.testing.assert_allclose(actual.cpu().numpy(),expected,rtol=1e-4)
        print(f'{device}: matched direct Mahalanobis solves')


def check_native_map():
    with zipfile.ZipFile(DATA/'RAW_FABRID.zip') as archive:
        source = json.loads(archive.read('RAW_FABRID/RAW_FABRID_Patches_COCO.json'))
        coords = defaultdict(list)
        for im in source['images']:
            coords[im['parent']].append((im['r_offset'],im['c_offset']))
        example = next(p for p in sorted(coords) if set(coords[p]) != set(FIXED))
        mask = np.asarray(read_image(archive,'masks/'+example,'L')) >= 128
    layouts = {'released':coords[example],'fixed':FIXED}
    all_coords = list(dict.fromkeys(coords[example]+FIXED))
    # Deliberate score ties and off-grid overlaps exercise max fusion and AP binning.
    rng = np.random.default_rng(13)
    fields = rng.choice(np.array([0.,.0011,.0049,.0072],dtype='float32'),size=(len(all_coords),16,16))
    lookup = dict(zip(all_coords,fields))
    hists = {p:[np.zeros(1,dtype='int64') for _ in range(2)] for p in layouts}
    record = summarize_fields(all_coords,fields,mask,layouts,True,hists)
    for policy, locations in layouts.items():
        native = np.full(mask.shape,-1.,dtype='float32')
        for y,x in locations:
            patch = np.kron(lookup[y,x], np.ones((16,16),dtype='float32'))
            native[y:y+256,x:x+256] = np.maximum(native[y:y+256,x:x+256],patch)
        bins = np.where(native<0,0,1+np.floor(1000*native)).astype(int)
        np.testing.assert_array_equal(np.bincount(bins.ravel()),hists[policy][0])
        np.testing.assert_array_equal(np.bincount(bins[mask],minlength=bins.max()+1),hists[policy][1])
        assert record[policy]['frame_score'] == native.max()
        if policy=='released':
            assert (native<0).any()
    print('Both native maps, including overlap and unseen pixels, matched direct accumulation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat-dir',type=Path,help='Directory written by a separate image-based fold rerun')
    args = parser.parse_args()
    check_gaussian()
    check_native_map()
    if args.repeat_dir:
        repeated = sorted(args.repeat_dir.glob('*.json'))
        if not repeated:
            raise ValueError('No repeated folds found')
        for path in repeated:
            original = json.loads((OUT/path.name).read_text())
            rerun = json.loads(path.read_text())
            for record in (original,rerun):
                for k in ['fit_seconds','total_seconds']:
                    del record[k]
            if original != rerun:
                raise ValueError('Fold repeat differs: '+path.name)
            print(path.name+': all recorded scores, counts and fitted-parameter hashes matched.')

if __name__ == '__main__':
    main()
