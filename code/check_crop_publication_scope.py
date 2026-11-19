"""Compare fixed, released, overlapping and reassigned 256-pixel crop layouts.

Run after the three saved ResNet clean-image holdout baselines and fixed-grid runs.
All policies are replayed from the same original images with the same frozen
model. Source masks are read for the geometry diagnostics.
"""
import csv
import io
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEEDS = (13017, 13018, 13019)
SIDE = 256
FIXED = [(y, x) for y in range(0, 769, SIDE) for x in range(0, 1537, SIDE)]
OVERLAP = [(y, x) for y in range(0, 769, 128) for x in range(0, 1537, 128)]
DONORS = ("donor_1", "donor_2", "donor_3")
POLICIES = ("released", "fixed", "overlap", *DONORS)


def read_json(path):
    return json.loads(path.read_text())


def timestamp(parent):
    raw = re.search(r"_(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}\.\d+)\.png$", parent)
    return datetime.strptime(raw[1], "%d-%m-%Y_%H-%M-%S.%f")


def unit_vectors(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def crop_geometry(mask, coords_by_policy):
    area = int(mask.sum())
    if area == 0:
        raise ValueError("Empty paired mask")
    out = {"mask_area": area}
    for policy, coords in coords_by_policy.items():
        masses = [int(mask[y:y+SIDE, x:x+SIDE].sum()) for y, x in coords]
        item = {"largest_tile_fraction": max(masses) / area}
        if policy == "released":
            coverage = np.zeros(mask.shape, dtype=np.uint8)
            for y, x in coords:
                coverage[y:y+SIDE, x:x+SIDE] += 1
            item.update(unseen_pixels=int((coverage == 0).sum()),
                        unseen_mask_pixels=int(mask[coverage == 0].sum()),
                        repeated_mask_pixels=int(mask[coverage > 1].sum()))
        out[policy] = item
    return out


def donor_layouts(info):
    """Permute layouts between positive frames within roll, preserving exact counts."""
    groups = defaultdict(list)
    for parent, row in sorted(info.items()):
        if row["has_defect"].lower() == "true":
            groups[row["roll_id"]].append(parent)
    rng = np.random.default_rng(20261119)
    assignments = {}
    for roll in sorted(groups):
        group = groups[roll]
        rng.shuffle(group)
        for index, parent in enumerate(group):
            assignments[parent] = {k: group[(index + step) % len(group)]
                                   for step, k in enumerate(DONORS, 1)}
            assert parent not in assignments[parent].values()
        for k in DONORS:
            assert {assignments[p][k] for p in group} == set(group)
    return assignments


def coordinate_control(info, patch_rows):
    moved = defaultdict(bool)
    table = defaultdict(lambda: {"off_grid": 0, "on_grid": 0})
    for row in patch_rows:
        off_grid = bool(row["r_offset"] % SIDE or row["c_offset"] % SIDE)
        table[row["type"]]["off_grid" if off_grid else "on_grid"] += 1
        moved[row["parent"]] |= off_grid
    assert set(moved) == set(info)
    matrix = defaultdict(lambda: {"flagged": 0, "unflagged": 0})
    for p in info:
        label = "defective" if info[p]["has_defect"].lower() == "true" else "clean"
        matrix[label]["flagged" if moved[p] else "unflagged"] += 1
    return {"rule": "any coordinate not divisible by 256; no training or pixels",
            "patch_table": dict(table), "parent_table": dict(matrix),
            "parent_auroc": float(roc_auc_score([info[p]["has_defect"].lower() == "true" for p in info],
                                                [moved[p] for p in info]))}


def extract(parent_names, info, released, assignments, models, transform, device):
    predictions = {seed: {} for seed in SEEDS}
    geometry = {}
    with zipfile.ZipFile(DATA / "RAW_FABRID.zip") as archive, torch.inference_mode():
        for index, parent in enumerate(parent_names, 1):
            policies = {"released": released[parent], "fixed": FIXED, "overlap": OVERLAP}
            for k in DONORS:
                policies[k] = released[assignments[parent][k]] if parent in assignments else FIXED
            if any(len(coords) != (91 if k == "overlap" else 28) for k, coords in policies.items()):
                raise ValueError(parent)
            unique = list(dict.fromkeys(pos for coords in policies.values() for pos in coords))
            positions = {point: i for i, point in enumerate(unique)}
            with Image.open(io.BytesIO(archive.read("RAW_FABRID/images/" + parent))) as raw:
                image = raw.convert("RGB")
            if image.size != (1792, 1024):
                raise ValueError(parent)
            features = []
            for start in range(0, len(unique), 16):
                batch = [transform(image.crop((x, y, x+SIDE, y+SIDE))) for y, x in unique[start:start+16]]
                features.append(models["encoder"](torch.stack(batch).to(device)).cpu().numpy())
            vectors = unit_vectors(np.concatenate(features))
            for seed in SEEDS:
                model = models[seed]
                if parent not in model["old"]:
                    continue
                distances = model["nn"].kneighbors(vectors, return_distance=True)[0][:, 0]
                scores = {}
                for policy, coords in policies.items():
                    scores[policy] = float(max(distances[positions[pos]] for pos in coords))
                predictions[seed][parent] = scores
            if info[parent]["has_defect"].lower() == "true":
                with Image.open(io.BytesIO(archive.read("RAW_FABRID/masks/" + parent))) as raw:
                    mask = np.asarray(raw.convert("L")) >= 128
                geometry[parent] = crop_geometry(mask, policies)
            if index % 100 == 0:
                print(f"Scored {index}/{len(parent_names)} original images", flush=True)
    return predictions, geometry


def evaluate(scores, previous, fixed_old, info, seed, ref_parents):
    names = sorted(scores)
    positives = [p for p in names if previous[p]["label"] == 1]
    negatives = [p for p in names if previous[p]["label"] == 0]
    # Half of the clean holdouts within each roll calibrate thresholds. No
    # calibration image enters the reference bank or the false-alarm denominator.
    cal = []
    rng = np.random.default_rng(seed + 20261118)
    for roll in sorted({info[p]["roll_id"] for p in negatives}):
        group = [p for p in negatives if info[p]["roll_id"] == roll]
        rng.shuffle(group)
        cal.extend(group[:len(group)//2])
    cal_set = set(cal)
    clean_test = sorted(set(negatives) - cal_set)
    assert set(names).isdisjoint(ref_parents)
    assert all(scores[p][k] == scores[p]["fixed"] for p in negatives for k in ("released", *DONORS))
    assert all(scores[p]["overlap"] >= scores[p]["fixed"] for p in names)
    times = defaultdict(list)
    for parent in ref_parents:
        times[info[parent]["roll_id"]].append(timestamp(parent))
    nearest = {p: min(abs((timestamp(p)-t).total_seconds()) for t in times[info[p]["roll_id"]]) for p in names}
    metrics = {}
    for policy in POLICIES:
        cutoff = float(np.quantile([scores[p][policy] for p in cal], 0.95, method="higher"))
        metrics[policy] = {
            "image_auroc_303": float(roc_auc_score([previous[p]["label"] for p in names],
                                                    [scores[p][policy] for p in names])),
            "cutoff": cutoff,
            "defects_flagged": sum(scores[p][policy] > cutoff for p in positives),
            "false_alarms": sum(scores[p][policy] > cutoff for p in clean_test),
            "per_roll_auroc": {},
        }
        for roll in sorted({info[p]["roll_id"] for p in names}):
            group = [p for p in names if info[p]["roll_id"] == roll]
            metrics[policy]["per_roll_auroc"][roll] = float(roc_auc_score(
                [previous[p]["label"] for p in group], [scores[p][policy] for p in group]))
    replay = {"released": max(abs(scores[p]["released"] - previous[p]["score"]) for p in names),
              "fixed": max(abs(scores[p]["fixed"] - fixed_old[p]["score"]) for p in names)}
    return {"metrics": metrics, "positive_count": len(positives), "calibration_clean": sorted(cal),
            "evaluation_clean": clean_test, "max_absolute_replay_error": replay,
            "unique_reference_frames": len(ref_parents),
            "nearest_same_roll_reference_seconds": nearest, "scores": scores}


def main():
    meta = DATA / "remote_metadata"
    with (meta / "RAW_FABRID_HighRes_Metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        info = {r["filename"]: r for r in csv.DictReader(f)}
    rows = read_json(RESULTS / "features" / "feature_rows.json")
    reference = np.load(RESULTS / "features" / "features.npy", mmap_mode="r")
    released = defaultdict(list)
    patch_rows = read_json(meta / "RAW_FABRID_Patches_COCO.json")["images"]
    for row in patch_rows:
        released[row["parent"]].append((row["r_offset"], row["c_offset"]))
    models = {}
    for seed in SEEDS:
        orig = read_json(RESULTS / f"baseline_seed_{seed}.json")["folds"]["clean_image_holdout"]
        fixed = read_json(RESULTS / f"features_fixed_grid_seed_{seed}.json")["folds"]["clean_image_holdout"]
        indices = orig["train_patch_indices"]
        if indices != fixed["train_patch_indices"] or orig["test_parents"] != fixed["test_parents"]:
            raise ValueError("Unmatched comparison")
        assert all(rows[i]["type"] == "train_good" for i in indices)
        models[seed] = {
            "nn": NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="brute", n_jobs=1).fit(
                unit_vectors(reference[indices])),
            "old": {r["parent"]: r for r in orig["predictions"]},
            "fixed": {r["parent"]: r for r in fixed["predictions"]},
            "ref_parents": {rows[i]["parent_raw"] for i in indices},
        }
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    encoder.fc = torch.nn.Identity()
    models["encoder"] = encoder.eval().to(device)
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    names = sorted(set.union(*(set(models[s]["old"]) for s in SEEDS)))
    assignments = donor_layouts(info)
    predictions, geometry = extract(names, info, released, assignments, models, transform, device)
    runs = {s: evaluate(predictions[s], models[s]["old"], models[s]["fixed"], info, s,
                        models[s]["ref_parents"]) for s in SEEDS}
    out = {"analysis": "exploratory, RAW-FABRID v1", "device": device, "policies": {
        "released": "28 COCO windows replayed from originals",
        "fixed": "28 disjoint grid windows, stride 256",
        "overlap": "91 grid windows, stride 128",
        "donor_1/2/3": "28 released windows from another positive frame in the same roll; clean frames keep fixed grid"},
        "aggregation": "maximum nearest-reference distance, common 256-pixel crops",
        "mask_foreground": ">=128", "threshold": "95th percentile (higher) of held-out clean calibration scores; strict exceedance",
        "coordinate_control": coordinate_control(info, patch_rows),
        "donor_layouts": assignments, "source_images_processed": len(names), "geometry": geometry, "runs": runs}
    target = RESULTS / "crop_publication_scope.json"
    target.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    for seed, run in runs.items():
        print(seed, {k: round(v["image_auroc_303"], 5) for k,v in run["metrics"].items()},
              "replay errors", run["max_absolute_replay_error"])


if __name__ == "__main__":
    main()
