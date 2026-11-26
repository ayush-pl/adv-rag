# Retrieval sweep on the calibration split

- Split: **calibration**, 36 answerable questions scored (5 unanswerable skipped).
- Dataset `sha256-72eed53de9d7`, corpus `sha256-9e14d8a33aa1`, code `87ba91d02c` with uncommitted changes.
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU.
- Brackets are 95% bootstrap intervals. Latency is warm, per query, and excludes model loading.
- Recall counts evidence found, not chunks, so it is comparable across chunking.

## All configurations

| Configuration | Recall@5 | All evidence@5 | Recall@20 | MRR | nDCG@10 | MAP | p50 ms |
|---|---|---|---|---|---|---|---|
| f512-hybrid-bge-k60-rerank | 0.833 [0.71, 0.94] | 0.806 | 0.889 | 0.700 | 0.742 | 0.694 | 347 |
| r512-hybrid-bge-k60-rerank | 0.750 [0.61, 0.88] | 0.722 | 0.806 | 0.678 | 0.698 | 0.675 | 378 |
| r256-hybrid-bge-k60-rerank | 0.708 [0.56, 0.85] | 0.694 | 0.750 | 0.637 | 0.643 | 0.616 | 326 |
| f512-hybrid-bge-k60 | 0.708 [0.56, 0.85] | 0.694 | 0.889 | 0.614 | 0.632 | 0.605 | 87 |
| f256-hybrid-bge-k60-rerank | 0.694 [0.54, 0.83] | 0.667 | 0.764 | 0.580 | 0.626 | 0.583 | 324 |
| r512-dense-bge | 0.681 [0.53, 0.82] | 0.667 | 0.806 | 0.540 | 0.579 | 0.535 | 17 |
| r512-hybrid-bge-k10 | 0.611 [0.46, 0.76] | 0.583 | 0.833 | 0.558 | 0.573 | 0.553 | 140 |
| r512-hybrid-bge-k30 | 0.597 [0.44, 0.75] | 0.583 | 0.806 | 0.536 | 0.559 | 0.535 | 139 |
| r256-hybrid-bge-k60 | 0.625 [0.47, 0.78] | 0.611 | 0.750 | 0.542 | 0.559 | 0.522 | 182 |
| r512-hybrid-bge-k60 | 0.597 [0.44, 0.75] | 0.583 | 0.806 | 0.534 | 0.553 | 0.530 | 142 |
| r512-hybrid-bge-k100 | 0.583 [0.43, 0.74] | 0.556 | 0.806 | 0.533 | 0.551 | 0.524 | 143 |
| f512-bm25 | 0.653 [0.50, 0.79] | 0.611 | 0.819 | 0.540 | 0.550 | 0.515 | 67 |
| f256-hybrid-bge-k60 | 0.583 [0.43, 0.74] | 0.556 | 0.764 | 0.531 | 0.547 | 0.508 | 154 |
| f512-dense-bge | 0.569 [0.42, 0.72] | 0.528 | 0.736 | 0.514 | 0.544 | 0.494 | 10 |
| r512-dense-minilm | 0.528 [0.38, 0.68] | 0.500 | 0.708 | 0.461 | 0.498 | 0.455 | 16 |
| r256-dense-minilm | 0.569 [0.42, 0.72] | 0.556 | 0.778 | 0.450 | 0.496 | 0.445 | 18 |
| r256-dense-bge | 0.542 [0.39, 0.69] | 0.500 | 0.778 | 0.477 | 0.496 | 0.452 | 18 |
| f256-bm25 | 0.528 [0.38, 0.68] | 0.472 | 0.597 | 0.489 | 0.491 | 0.462 | 124 |
| f256-dense-minilm | 0.569 [0.40, 0.72] | 0.556 | 0.681 | 0.428 | 0.481 | 0.428 | 15 |
| r512-bm25 | 0.500 [0.35, 0.65] | 0.472 | 0.681 | 0.484 | 0.480 | 0.459 | 115 |
| f256-dense-bge | 0.556 [0.39, 0.72] | 0.556 | 0.708 | 0.446 | 0.477 | 0.438 | 18 |
| r256-bm25 | 0.500 [0.35, 0.65] | 0.472 | 0.639 | 0.457 | 0.461 | 0.433 | 151 |
| f512-dense-minilm | 0.528 [0.36, 0.69] | 0.528 | 0.722 | 0.418 | 0.452 | 0.411 | 5 |

## Paired comparisons

Differences are treatment minus baseline on identical questions. A result is only called better or worse when its interval excludes zero.

| Exp | Baseline | Treatment | Δ Recall@5 | Δ MRR | Δ nDCG@10 | Verdict (nDCG@10) |
|---|---|---|---|---|---|---|
| R1 | r256-dense-minilm | r256-dense-bge | -0.028 [-0.22, +0.17] | +0.027 [-0.10, +0.15] | -0.000 [-0.12, +0.12] | inconclusive |
| R1 | r512-dense-minilm | r512-dense-bge | +0.153 [+0.00, +0.31] | +0.080 [-0.02, +0.18] | +0.081 [-0.01, +0.17] | inconclusive |
| R2 | r256-dense-bge | r256-bm25 | -0.042 [-0.19, +0.11] | -0.020 [-0.16, +0.12] | -0.035 [-0.17, +0.10] | inconclusive |
| R2 | r512-dense-bge | r512-bm25 | -0.181 [-0.35, -0.03] | -0.057 [-0.21, +0.09] | -0.099 [-0.24, +0.03] | inconclusive |
| R3 | r256-dense-bge | r256-hybrid-bge-k60 | +0.083 [-0.03, +0.21] | +0.066 [-0.03, +0.17] | +0.063 [-0.03, +0.16] | inconclusive |
| R3 | r512-dense-bge | r512-hybrid-bge-k60 | -0.083 [-0.22, +0.04] | -0.006 [-0.12, +0.11] | -0.026 [-0.14, +0.08] | inconclusive |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k10 | +0.014 [-0.07, +0.10] | +0.024 [-0.00, +0.05] | +0.020 [-0.03, +0.08] | inconclusive |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k30 | +0.000 [+0.00, +0.00] | +0.001 [-0.01, +0.01] | +0.006 [-0.02, +0.03] | inconclusive |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k100 | -0.014 [-0.04, +0.00] | -0.002 [-0.00, -0.00] | -0.002 [-0.00, +0.00] | inconclusive |
| R5 | r256-hybrid-bge-k60 | r256-hybrid-bge-k60-rerank | +0.083 [+0.00, +0.19] | +0.094 [-0.02, +0.22] | +0.084 [-0.01, +0.19] | inconclusive |
| R5 | r512-hybrid-bge-k60 | r512-hybrid-bge-k60-rerank | +0.153 [+0.06, +0.28] | +0.144 [+0.02, +0.28] | +0.145 [+0.03, +0.26] | better |
| R6 | f256-dense-bge | r256-dense-bge | -0.014 [-0.17, +0.14] | +0.030 [-0.11, +0.17] | +0.019 [-0.11, +0.15] | inconclusive |
| R6 | f512-dense-bge | r512-dense-bge | +0.111 [-0.01, +0.25] | +0.027 [-0.10, +0.16] | +0.035 [-0.08, +0.15] | inconclusive |
| R6 | f256-hybrid-bge-k60 | r256-hybrid-bge-k60 | +0.042 [-0.06, +0.15] | +0.012 [-0.06, +0.08] | +0.012 [-0.05, +0.07] | inconclusive |
| R6 | f512-hybrid-bge-k60 | r512-hybrid-bge-k60 | -0.111 [-0.25, +0.00] | -0.079 [-0.18, +0.02] | -0.079 [-0.17, +0.02] | inconclusive |

## R7: Recall by cutoff

| Configuration | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| r512-dense-bge | 0.417 | 0.611 | 0.681 | 0.736 | 0.806 |
| r512-bm25 | 0.403 | 0.472 | 0.500 | 0.556 | 0.681 |
| r512-hybrid-bge-k60 | 0.444 | 0.514 | 0.597 | 0.667 | 0.806 |
| r512-hybrid-bge-k60-rerank | 0.583 | 0.736 | 0.750 | 0.778 | 0.806 |

## By question type, for f512-hybrid-bge-k60-rerank

Small groups, so these are exploratory only.

| Type | Questions | Recall@5 | nDCG@10 |
|---|---|---|---|
| config_code | 6 | 0.833 | 0.655 |
| multi_hop | 4 | 0.750 | 0.735 |
| paraphrase_ambiguous | 2 | 1.000 | 0.750 |
| single_hop | 16 | 0.875 | 0.817 |
| table | 4 | 0.750 | 0.658 |
| version_specific | 4 | 0.750 | 0.658 |
