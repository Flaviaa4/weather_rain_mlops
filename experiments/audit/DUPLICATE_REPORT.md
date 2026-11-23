# Duplicate and leakage report (Section 1)

Command: `python -m src.audit --quarantine` (code in `src/audit.py`). Rerun without `--quarantine` for a read-only scan.

## Method

1. Every image gets a 64-bit dHash (8x9 grayscale, EXIF-corrected).
2. Candidate pairs: dHash Hamming distance 4 or less, OR grayscale correlation of 32x32 normalized thumbnails of 0.94 or more.
3. Each candidate gets two pixel checks: color thumbnail mean absolute difference (`thumb_mae`, 32x32 RGB, 0 to 1) and grayscale correlation (`gray_corr`).
4. A pair is a confirmed duplicate if:
   - hash stage: Hamming 4 or less AND (`thumb_mae` 0.07 or less OR `gray_corr` 0.98 or more), or
   - correlation stage: `gray_corr` 0.94 or more AND `thumb_mae` 0.10 or less,
   - and it is not in `reviewed_not_duplicates.json` (2 pairs rejected after visual review).
5. Confirmed pairs are merged into groups. One copy per group is kept, preferring train, then valid, then test. All other copies are moved to `data/quarantine/<split>/<class>/`.

Thresholds were set by viewing pair sheets in `sheets/` (hash-only, middle band, high band, correlation candidates).

## Why the hash alone was not enough

dHash with Hamming 4 or less flagged 619 pairs, 151 of them across different labels. Visual review showed every cross-label pair was a false match: rain splash scenes and sunsets share the structure "dark lower half, bright horizontal band". The color thumbnail distance separates them cleanly: all 151 cross-label pairs have `thumb_mae` above 0.12, while true duplicates sit below 0.07 (crops, re-encodes, color edits). The correlation scan found 6 more true duplicates the hash missed (Hamming 5 to 15, crops and tone changes).

## Results

| | before | after |
|---|---|---|
| images | 1099 | 997 |
| confirmed duplicate pairs | 114 | 0 |
| cross-split confirmed pairs | 55 | 0 |
| confirmed pairs with conflicting labels | 0 | 0 |
| duplicate groups | 90 | |
| files quarantined | 102 | |

Groups by split membership: train only 42, train+test 21, train+valid 15, valid+test 4, all three splits 3, test only 3, valid only 2. Group sizes: 79 groups of 2, 10 of 3, 1 of 4. The full list is in `quarantine_manifest.csv`.

Quarantined files per split and class:

| split | cloudy | rainy | shiny | sunrise | total |
|---|---|---|---|---|---|
| train | 18 | 7 | 12 | 13 | 50 |
| valid | 9 | 3 | 6 | 3 | 21 |
| test | 5 | 5 | 14 | 7 | 31 |

Clean split counts:

| split | cloudy | rainy | shiny | sunrise | total |
|---|---|---|---|---|---|
| train | 182 | 142 | 145 | 250 | 719 |
| valid | 48 | 20 | 33 | 43 | 144 |
| test | 29 | 29 | 40 | 36 | 134 |

## Files

- `before_audit_table.csv`, `after_audit_table.csv`: one row per image (split, label, size, mode, format, EXIF orientation, file size, aspect ratio, dHash).
- `before_duplicate_candidates.csv`: every candidate with Hamming, `thumb_mae`, `gray_corr`, `confirmed`, `reviewed_reject`.
- `before_confirmed_duplicates.csv`, `after_confirmed_duplicates.csv` (empty after cleaning).
- `quarantine_manifest.csv`: group id, path, action (keep or quarantine), keeper.
- `reviewed_not_duplicates.json`: pairs rejected after visual review, with reasons.

## Residual risk

Some scenes appear as several different shots of the same place (for example the mountain sunset series sunrise83, sunrise89, sunrise260, sunrise292). These are not duplicates, but they are close relatives and some cross splits. This is weaker leakage than a duplicate and is not removed. K-fold and the field test in Section 6 are the checks for it.
