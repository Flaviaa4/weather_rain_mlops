# Experiment log

Format per entry: id, date, section, what changes, hypothesis (direction, size, why), then result, right or wrong, conclusion.

## S1-E1: Is dHash Hamming 4 or less enough to find duplicates? (2026-11-22)

- Change: none, this is a data check.
- Hypothesis: dHash at Hamming 4 or less finds near-duplicates with few false matches, because Roboflow exports often contain re-encoded copies of the same source photo. Expected a few dozen pairs, mostly same label.
- Result: 619 pairs, 151 with different labels. Visual review: every cross-label pair was a false match (dark scene with a bright horizontal band, for example rain splashes vs sunsets). Color thumbnail distance separates them: true duplicates below 0.07, false matches above 0.12, with nothing between 0.02 and 0.03.
- Hypothesis: wrong on false matches. Simple sky and horizon structures collide in a 64-bit structural hash.
- Conclusion: confirm hash candidates with pixel checks (color thumbnail MAE and grayscale correlation) and record visually rejected pairs in `experiments/audit/reviewed_not_duplicates.json`.

## S1-E2: Does the hash miss duplicates? (2026-11-22)

- Change: add a hash-independent scan (grayscale thumbnail correlation of 0.93 or more over all pairs).
- Hypothesis: a few crops or tone-edited copies have a Hamming distance above 4 and are missed. Expected fewer than 10.
- Result: 6 more true duplicates (Hamming 5 to 15), 3 of them across splits (shine94 train = shine16 test, shine229 train = shine136 valid, shine190 train = shine213 valid). 1 false match passed the numbers (sunrise92 vs cloudy263) and was rejected after review.
- Hypothesis: right.
- Conclusion: the audit uses both stages. Final result: 114 confirmed pairs in 90 groups, 102 files quarantined, 0 confirmed duplicates left.

## Test set touches

| date | section | what | by |
|---|---|---|---|
| 2026-11-22 | 1 | Data audit only: file properties, duplicate detection, quarantine of 31 duplicate test copies. No model was evaluated. | Claude Code |
