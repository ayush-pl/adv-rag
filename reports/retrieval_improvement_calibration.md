# Duplicate removal on the calibration split

- Split: **calibration**, 36 answerable questions scored (5 unanswerable skipped).
- Dataset `sha256-72eed53de9d7`, corpus `sha256-9e14d8a33aa1`, code `87ba91d02c` with uncommitted changes.
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU.
- Brackets are 95% bootstrap intervals. Latency is warm, per query, and excludes model loading.
- Recall counts evidence found, not chunks, so it is comparable across chunking.

## All configurations

| Configuration | Recall@5 | All evidence@5 | Recall@20 | MRR | nDCG@10 | MAP | p50 ms |
|---|---|---|---|---|---|---|---|
| r512-hybrid-bge-k60-dedup-rerank | 0.806 [0.67, 0.92] | 0.806 | 0.833 | 0.702 | 0.733 | 0.695 | 374 |
| r512-hybrid-bge-k60-rerank | 0.750 [0.61, 0.88] | 0.722 | 0.806 | 0.678 | 0.698 | 0.675 | 378 |
| r256-hybrid-bge-k60-dedup-rerank | 0.722 [0.58, 0.86] | 0.722 | 0.750 | 0.638 | 0.656 | 0.618 | 329 |
| r256-hybrid-bge-k60-rerank | 0.708 [0.56, 0.85] | 0.694 | 0.750 | 0.637 | 0.643 | 0.616 | 326 |
| r512-dense-bge-dedup | 0.681 [0.53, 0.82] | 0.667 | 0.806 | 0.556 | 0.606 | 0.549 | 15 |
| r512-hybrid-bge-k60-dedup | 0.667 [0.50, 0.81] | 0.667 | 0.833 | 0.553 | 0.586 | 0.549 | 140 |
| r512-dense-bge | 0.681 [0.53, 0.82] | 0.667 | 0.806 | 0.540 | 0.579 | 0.535 | 17 |
| r256-hybrid-bge-k60-dedup | 0.653 [0.50, 0.81] | 0.639 | 0.750 | 0.553 | 0.577 | 0.533 | 170 |
| r256-hybrid-bge-k60 | 0.625 [0.47, 0.78] | 0.611 | 0.750 | 0.542 | 0.559 | 0.522 | 182 |
| r512-hybrid-bge-k60 | 0.597 [0.44, 0.75] | 0.583 | 0.806 | 0.534 | 0.553 | 0.530 | 142 |

## Paired comparisons

Differences are treatment minus baseline on identical questions. A result is only called better or worse when its interval excludes zero.

| Exp | Baseline | Treatment | Δ Recall@5 | Δ MRR | Δ nDCG@10 | Verdict (nDCG@10) |
|---|---|---|---|---|---|---|
| D1 | r512-dense-bge | r512-dense-bge-dedup | +0.000 [+0.00, +0.00] | +0.015 [+0.00, +0.03] | +0.027 [+0.01, +0.05] | better |
| D1 | r256-hybrid-bge-k60 | r256-hybrid-bge-k60-dedup | +0.028 [+0.00, +0.08] | +0.010 [+0.00, +0.02] | +0.018 [+0.00, +0.04] | better |
| D1 | r512-hybrid-bge-k60 | r512-hybrid-bge-k60-dedup | +0.069 [+0.00, +0.17] | +0.019 [+0.01, +0.03] | +0.033 [+0.01, +0.06] | better |
| D1 | r256-hybrid-bge-k60-rerank | r256-hybrid-bge-k60-dedup-rerank | +0.014 [+0.00, +0.04] | +0.001 [-0.01, +0.02] | +0.013 [-0.00, +0.04] | inconclusive |
| D1 | r512-hybrid-bge-k60-rerank | r512-hybrid-bge-k60-dedup-rerank | +0.056 [+0.00, +0.12] | +0.024 [+0.00, +0.06] | +0.034 [+0.00, +0.08] | better |

## R7: Recall by cutoff

| Configuration | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| r512-hybrid-bge-k60-rerank | 0.583 | 0.736 | 0.750 | 0.778 | 0.806 |
| r512-hybrid-bge-k60-dedup-rerank | 0.583 | 0.764 | 0.806 | 0.833 | 0.833 |

## By question type, for r512-hybrid-bge-k60-dedup-rerank

Small groups, so these are exploratory only.

| Type | Questions | Recall@5 | nDCG@10 |
|---|---|---|---|
| config_code | 6 | 0.833 | 0.649 |
| multi_hop | 4 | 1.000 | 0.795 |
| paraphrase_ambiguous | 2 | 0.500 | 0.250 |
| single_hop | 16 | 0.812 | 0.781 |
| table | 4 | 0.750 | 0.825 |
| version_specific | 4 | 0.750 | 0.750 |
