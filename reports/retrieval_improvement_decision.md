# Duplicate removal: decision before the test run

Written 2026-11-26 13:39 UTC, before any duplicate-removal configuration was run on the test split.

- **Selected configuration:** `r512-hybrid-bge-k60-dedup-rerank`, the best on development and
  calibration together (70 answerable questions): Recall@5 0.836 and nDCG@10 0.779.
- **Compared with:** `r512-hybrid-bge-k60-rerank`, the best configuration without duplicate removal
  on test.
- **Primary metrics:** nDCG@10 and Recall@5, paired bootstrap 95% intervals. Called an improvement
  only if the interval excludes zero.
- **Run once.** No configuration is changed after seeing test results.

Selection results (development + calibration, paired, treatment minus baseline):

| Baseline | Treatment | Δ Recall@5 | Δ All evidence@5 | Δ nDCG@10 |
|---|---|---|---|---|
| r512-dense-bge | +dedup | +0.021 [+0.00, +0.06] | +0.029 [+0.00, +0.07] | +0.025 [+0.01, +0.04] |
| r256-hybrid-bge-k60 | +dedup | +0.014 [+0.00, +0.04] | +0.014 [+0.00, +0.04] | +0.023 [+0.01, +0.04] |
| r512-hybrid-bge-k60 | +dedup | +0.036 [+0.00, +0.09] | +0.043 [+0.00, +0.10] | +0.028 [+0.01, +0.05] |
| r256-hybrid-bge-k60-rerank | +dedup | +0.043 [+0.01, +0.09] | +0.057 [+0.01, +0.11] | +0.035 [+0.01, +0.07] |
| r512-hybrid-bge-k60-rerank | +dedup | +0.057 [+0.01, +0.11] | +0.086 [+0.03, +0.16] | +0.054 [+0.02, +0.10] |
