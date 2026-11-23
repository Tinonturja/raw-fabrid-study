#!/usr/bin/env python3
"""Audit source-image and roll grouping in RAW-FABRID's metadata CSVs."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata_dir", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    high = read_csv(args.metadata_dir / "RAW_FABRID_HighRes_Metadata.csv")
    patches = read_csv(args.metadata_dir / "RAW_FABRID_Patches_Metadata.csv")
    by_name = {r["filename"]: r for r in high}
    if len(by_name) != len(high):
        raise SystemExit("Duplicate high-resolution filenames")
    type_counts = Counter(r["type"] for r in patches)
    parents_by_type = defaultdict(set)
    roll_patch_counts = defaultdict(Counter)
    unmatched = []
    for p in patches:
        parent = p["parent_raw"]
        raw = by_name.get(parent)
        if raw is None:
            unmatched.append(parent)
            continue
        parents_by_type[p["type"]].add(parent)
        roll_patch_counts[raw["roll_id"]][p["type"]] += 1
    groups_by_roll = {}
    for roll in sorted({r["roll_id"] for r in high}):
        rows = [r for r in high if r["roll_id"] == roll]
        groups_by_roll[roll] = {
            "raw_images": len(rows),
            "clean_images": sum(r["has_defect"].strip().lower() in {"false", "0", "no"} for r in rows),
            "defect_images": sum(r["has_defect"].strip().lower() in {"true", "1", "yes"} for r in rows),
            "patches_by_split": dict(roll_patch_counts[roll]),
        }
    overlap = {}
    keys = sorted(parents_by_type)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlap[f"{a} & {b}"] = len(parents_by_type[a] & parents_by_type[b])
    clean_by_type = defaultdict(set)
    defect_by_type = defaultdict(set)
    for p in patches:
        raw = by_name.get(p["parent_raw"])
        if not raw:
            continue
        is_defect = raw["has_defect"].strip().lower() in {"true", "1", "yes"}
        (defect_by_type if is_defect else clean_by_type)[p["type"]].add(p["parent_raw"])
    report = {
        "highres_rows": len(high),
        "highres_by_has_defect": dict(Counter(r["has_defect"] for r in high)),
        "highres_unique_rolls": len(groups_by_roll),
        "highres_by_roll": groups_by_roll,
        "patch_rows": len(patches),
        "patches_by_type": dict(type_counts),
        "unique_parents_by_type": {k: len(v) for k, v in parents_by_type.items()},
        "clean_source_parents_by_type": {k: len(v) for k, v in clean_by_type.items()},
        "defect_source_parents_by_type": {k: len(v) for k, v in defect_by_type.items()},
        "parent_overlap_by_type": overlap,
        "unmatched_parent_rows": len(unmatched),
        "unmatched_parent_examples": unmatched[:10],
        "paper_count_discrepancy": {"paper_clean": 505, "metadata_clean": sum(r["has_defect"].strip().lower() in {"false", "0", "no"} for r in high),
                                    "paper_defect": 204, "metadata_defect": sum(r["has_defect"].strip().lower() in {"true", "1", "yes"} for r in high)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
