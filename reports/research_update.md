# RAW-FABRID: dataset audit and exploratory research update

**Date:** 18 November 2026

**Status:** Dataset and archive audit complete for the checks described below; model analysis is exploratory. Do not use these results as publication-ready findings.

## Decision

RAW-FABRID remains a promising candidate for a study about the performance of a fabric anomaly detector on previously unseen fabric rolls. The data also contain annotation and reporting inconsistencies that need to be documented and handled explicitly. This is not yet evidence for a new algorithm or a universal manufacturing result.

The initial hypothesis that training and test patches share the same original image is **not supported** by the release metadata. A more meaningful research question is whether a detector that works with some reference images from a roll also works on a held-out roll, and whether the benchmark includes adequate wholly clean images for evaluating false alarms.

## Evidence and data provenance

- Official dataset: [RAW-FABRID, Mendeley Data version 1](https://data.mendeley.com/datasets/db6g85xsyg/1), CC BY 4.0.
- Data descriptor: [Pérez-Llorens et al., 2026](https://doi.org/10.3390/data11050116).
- The first assembled archive had the expected byte count but failed the publisher's SHA-256 and contained 16 CRC-failing original image entries in one range. After independently replacing that range, the complete 4,133,046,386-byte archive matches the publisher SHA-256 `f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9`, and all ZIP member CRCs pass. All four initially range-extracted metadata files exactly match the verified archive.
- All 709 original images decode at 1792 × 1024. Each is stored as RGB with identical channels, so its content is grayscale. All high-resolution masks and the relevant patch masks and image crops were rechecked directly against the verified archive. Features for all 19,852 released patch images were calculated before archive repair; each patch member passed its individual CRC, and the repaired byte range contained original images, not patches.
- Replaying seed 13017 from the saved features reproduced every reported AUROC exactly in the current environment. The ResNet weights and analysis scripts are hashed in `research_sha256.txt`.
- Detailed machine-readable findings are in `metadata_split_audit_preliminary.json`, `coco_audit_preliminary.json`, `mask_sample_audit_preliminary.json`, `patch_mask_consistency_preliminary.json`, and `good_crop_alignment_preliminary.json` in this folder.

## What the split actually contains

The two released metadata files contain 709 source images: 507 flagged clean and 202 flagged defective. The patch metadata contain 14,196 train-good, 4,969 test-good and 687 test-defect patches.

| Source-image and roll check | Observed from release metadata |
|---|---:|
| Source images in training | 507 clean |
| Source images in test | 202 defective |
| Original images shared between training and test | 0 |
| Entirely clean original images in the official test set | 0 |
| Defective parents represented in both test-good and test-defect patches | 202 |
| Roll IDs appearing in training and test | 7 of 7 |

Test-good patches are clean-labeled regions cropped from images that also contain defects. This can support a **patch-level** test, but it does not directly measure image-level false alarms on wholly defect-free images. Within each test parent, patches are dependent observations. The released split uses the same rolls for training and testing and cannot, by itself, measure unseen-roll performance.

## Annotation and release discrepancies

1. The paper reports 505 clean and 204 defective source images. The CSV reports 507 clean and 202 defective, and the ZIP catalog lists 204 high-resolution masks. Two masks have no same-named source image: `Rollo2A_0000031_D_07-05-2024_10-39-06.546.png` and `Rollo2A_0000032_D_07-05-2024_10-39-07.598.png`.
2. The paper describes binary source masks with 0 and 255 values. Only 53 of 204 source masks are strictly binary; 151 contain additional grayscale values. Among the 202 masks paired with source images, intermediate values affect 1,739 of 1,906,272 nonzero mask pixels. In 471 of 687 defect patch masks, intermediate values appear as well. This is numerically small overall, but exact threshold rules must be specified.
3. Five patches labeled test-defect have no patch-level COCO polygon, even though their masks exist. Their CSV and mask counts are 1, 10, 2, 2, and 8 marked pixels. The reason the polygons were omitted has not been established.
4. Twenty-five test-good patch crops contain nonzero source-mask pixels. For all 25, the patch pixels match the image crop at the recorded coordinates; 24 include mask values at least 128. The largest marked region among these is 79 pixels out of 65,536. These inconsistencies may be small, but a rigorous evaluation should identify and analyze them.
5. The paper describes a `_mask` filename suffix for patch masks. The released patch mask uses the same filename as the image, so a loader using the documented suffix will not find it.
6. The COCO coordinates show 640 overlapping pairs of test-good and test-defect crops from the same parent image, involving 565 good patches and 331 defect patches. Overlap does not alone prove the good patch has an incorrect label. It does mean that patch results should not be treated as independent source-image results.
7. The dataset paper explains that some crops were recentered using defect locations. The coordinates show 409 crops off the fixed grid, 408 of them defect-labeled, spanning 197 defective parent images. Results on those released crops do not directly represent a setting where the location of a defect is unknown. For the image-level analysis below, a second set of features was generated from a fixed 4 × 7 grid on each original image, without consulting defect masks.

All 709 originals were checked for exact pixel-identical duplicates after converting their equal RGB channels to grayscale; none were found. This does not rule out near-duplicates or temporal dependence.

## Exploratory baselines

Two fixed baselines provide a low-cost feasibility check, not a state-of-the-art comparison. For the first, each grayscale patch was converted to three channels, resized from 256 to 128 pixels, and passed through an ImageNet-pretrained ResNet-18. For the second, a 64 × 64 grayscale version produced 38 fixed features describing intensities, gradients and Fourier texture. Each feature vector was normalized; the anomaly score was distance to the nearest of 1,000 randomly selected normal training-patch features. No dataset-specific fine-tuning used test labels. Source-image scores are the maximum of their patch scores. Runs use three seeds: 13017, 13018 and 13019.

Across the three seeds:

- With ResNet features, the official **patch-level** test AUROC was 0.808–0.816; the texture baseline scored 0.762–0.768. This test contains 687 positive and 4,969 good-labeled patches from defective parent images.
- In a held-out-clean-image test using the **released crops**, image-level AUROC was 0.863–0.866 for ResNet features and 0.689–0.727 for texture features. Each run tested all 202 defect-containing images against 101 held-out clean source images. The clean samples vary with the seed.
- On the **fixed grid** using the same training references and the same test images for each seed, image-level AUROC was 0.815–0.823 for ResNet and 0.606–0.645 for texture. The released defect-centered crop result was higher by 0.043–0.049 and 0.082–0.084, respectively, for these particular models and splits. Crop locations and image content differ, so this observation should not be generalized to all detectors.

These two numbers describe different sample units and populations. Their difference cannot be interpreted as the effect of changing the split. The defect prevalence in these test cohorts is artificial and average precision should not be interpreted as production precision.

### Paired within-roll versus held-out-roll comparison

For each roll and seed, both reference-bank conditions score the **same test images**, using the fixed grid. One normal reference bank can include clean training images from that roll; the other excludes the roll entirely. Both use 1,000 training-patch features. The test includes the same seed's held-out clean images and all defect-containing images from the target roll. Values below are the mean across three seeds; the clean test images change between seeds. The table uses ResNet features.

| Roll | Defective test images | Clean test images | AUROC with some same-roll reference | AUROC with roll excluded |
|---|---:|---:|---:|---:|
| Rollo1A | 21 | 13 | 0.828 | 0.774 |
| Rollo2A | 27 | 11 | 0.842 | 0.838 |
| Rollo3A | 7 | 3 | 0.905 | 0.968 |
| Rollo4A | 78 | 18 | 0.799 | 0.766 |
| Rollo5A | 15 | 4 | 0.939 | 0.939 |
| Rollo6A | 28 | 19 | 0.801 | 0.569 |
| Rollo7A | 26 | 33 | 0.731 | 0.602 |

The classical texture baseline showed a similar direction for rolls 6 and 7: mean AUROC changed from 0.556 to 0.425 for roll 6 and 0.616 to 0.514 for roll 7. These numbers are not a claim that one method is universally better.

The [fixed-grid comparison chart](fixed_grid_roll_comparison.png) shows each seed and the mean for both feature sets.

The pattern for rolls 6 and 7 is worth follow-up. It is not proof that holding out a roll caused the difference: removing same-roll references also changes the rest of the training pool, individual images may be temporally related, and the reference samples and clean holdouts are random. Three seeds are repeated analyses of the same dataset, not three independent datasets. The small clean sample counts for rolls 3 and 5 make those figures especially unstable. No performance result here is a production false-alarm rate or a claim about other factories.

Source-image bootstrap intervals are saved with the unpaired baseline outputs. They condition on the seven observed rolls and treat source images as independent; if neighboring images are related, those intervals can be overly narrow. They should not be interpreted as population-level uncertainty over rolls.

## Recommended next work

1. Freeze a protocol for source-level and roll-level evaluation. Specify how to handle orphan masks, nonbinary values and the 25 clean-labeled patches that overlap source mask annotations. Keep an untouched test partition if later experiments tune the method.
2. Add a current anomaly-localization baseline, compare matched reference budgets and the same target source images, and report per-roll counts and repeated-seed variability. Assess sensitivity to patch size and image-score aggregation.
3. Check temporal proximity and acquisition metadata before claiming independence. The dataset contains seven roll IDs; it does not establish seven independent factories or suppliers.
4. Verify the literature gap and compare against existing unseen-fabric work, including the [OLP study](https://doi.org/10.3390/s22134750). A new evaluation protocol or result must be differentiated from prior unseen-fabric generalization studies.

The current paired comparison is a feasibility signal. It is not yet a completed research paper, validated model, or finalized project claim.
