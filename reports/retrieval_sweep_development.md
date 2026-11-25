# Retrieval sweep on the development split

- Split: **development**, 35 answerable questions scored (4 unanswerable skipped).
- Dataset `sha256-5afd34ef1d17`, corpus `sha256-9e14d8a33aa1`, code `7862ce149d` with uncommitted changes.
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU.
- Brackets are 95% bootstrap intervals. Latency is warm, per query, and excludes model loading.
- Recall counts evidence found, not chunks, so it is comparable across chunking.

## All configurations

| Configuration | Recall@5 | All evidence@5 | Recall@20 | MRR | nDCG@10 | MAP | p50 ms |
|---|---|---|---|---|---|---|---|
| r256-hybrid-bge-k60-rerank | 0.814 [0.69, 0.93] | 0.771 | 0.886 | 0.736 | 0.755 | 0.717 | 319 |
| r512-hybrid-bge-k10 | 0.786 [0.66, 0.90] | 0.743 | 0.943 | 0.767 | 0.752 | 0.730 | 124 |
| r512-hybrid-bge-k30 | 0.814 [0.69, 0.93] | 0.771 | 0.943 | 0.741 | 0.741 | 0.702 | 126 |
| r512-hybrid-bge-k60-rerank | 0.800 [0.67, 0.91] | 0.743 | 0.900 | 0.748 | 0.741 | 0.718 | 366 |
| r512-hybrid-bge-k60 | 0.843 [0.73, 0.94] | 0.800 | 0.900 | 0.729 | 0.733 | 0.686 | 130 |
| r512-hybrid-bge-k100 | 0.814 [0.69, 0.93] | 0.771 | 0.900 | 0.726 | 0.721 | 0.680 | 127 |
| f256-hybrid-bge-k60-rerank | 0.800 [0.67, 0.91] | 0.771 | 0.900 | 0.683 | 0.704 | 0.660 | 314 |
| f256-hybrid-bge-k60 | 0.771 [0.63, 0.90] | 0.743 | 0.900 | 0.680 | 0.693 | 0.654 | 140 |
| f512-hybrid-bge-k60-rerank | 0.814 [0.69, 0.93] | 0.800 | 0.900 | 0.643 | 0.689 | 0.636 | 344 |
| r512-bm25 | 0.714 [0.57, 0.84] | 0.657 | 0.814 | 0.691 | 0.688 | 0.637 | 101 |
| r256-hybrid-bge-k60 | 0.757 [0.61, 0.89] | 0.714 | 0.886 | 0.693 | 0.682 | 0.648 | 172 |
| f512-hybrid-bge-k60 | 0.757 [0.61, 0.89] | 0.714 | 0.900 | 0.605 | 0.660 | 0.592 | 78 |
| f256-bm25 | 0.700 [0.54, 0.83] | 0.657 | 0.829 | 0.671 | 0.651 | 0.627 | 114 |
| r256-bm25 | 0.743 [0.60, 0.87] | 0.686 | 0.829 | 0.626 | 0.639 | 0.575 | 135 |
| r512-dense-bge | 0.671 [0.53, 0.81] | 0.629 | 0.800 | 0.636 | 0.637 | 0.603 | 16 |
| f256-dense-bge | 0.686 [0.53, 0.83] | 0.657 | 0.843 | 0.617 | 0.634 | 0.585 | 20 |
| f512-bm25 | 0.729 [0.59, 0.86] | 0.686 | 0.786 | 0.567 | 0.604 | 0.556 | 57 |
| r256-dense-bge | 0.643 [0.49, 0.79] | 0.600 | 0.743 | 0.577 | 0.590 | 0.547 | 20 |
| r256-dense-minilm | 0.543 [0.39, 0.70] | 0.486 | 0.771 | 0.545 | 0.545 | 0.508 | 15 |
| r512-dense-minilm | 0.643 [0.49, 0.79] | 0.600 | 0.786 | 0.542 | 0.544 | 0.505 | 14 |
| f512-dense-bge | 0.629 [0.47, 0.79] | 0.600 | 0.857 | 0.509 | 0.534 | 0.492 | 10 |
| f256-dense-minilm | 0.600 [0.44, 0.74] | 0.543 | 0.829 | 0.493 | 0.503 | 0.455 | 14 |
| f512-dense-minilm | 0.486 [0.33, 0.64] | 0.429 | 0.743 | 0.459 | 0.493 | 0.422 | 6 |

## Paired comparisons

Differences are treatment minus baseline on identical questions. A result is only called better or worse when its interval excludes zero.

| Exp | Baseline | Treatment | Δ Recall@5 | Δ MRR | Δ nDCG@10 | Verdict (nDCG@10) |
|---|---|---|---|---|---|---|
| R1 | r256-dense-minilm | r256-dense-bge | +0.100 [-0.01, +0.23] | +0.032 [-0.09, +0.16] | +0.045 [-0.05, +0.15] | inconclusive |
| R1 | r512-dense-minilm | r512-dense-bge | +0.029 [-0.09, +0.14] | +0.094 [-0.03, +0.22] | +0.093 [-0.01, +0.20] | inconclusive |
| R2 | r256-dense-bge | r256-bm25 | +0.100 [-0.06, +0.26] | +0.049 [-0.12, +0.22] | +0.049 [-0.11, +0.21] | inconclusive |
| R2 | r512-dense-bge | r512-bm25 | +0.043 [-0.11, +0.20] | +0.055 [-0.11, +0.21] | +0.051 [-0.10, +0.20] | inconclusive |
| R3 | r256-dense-bge | r256-hybrid-bge-k60 | +0.114 [-0.01, +0.26] | +0.116 [-0.01, +0.24] | +0.092 [-0.01, +0.20] | inconclusive |
| R3 | r512-dense-bge | r512-hybrid-bge-k60 | +0.171 [+0.06, +0.31] | +0.094 [-0.04, +0.22] | +0.096 [-0.01, +0.21] | inconclusive |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k10 | -0.057 [-0.14, +0.00] | +0.038 [-0.00, +0.09] | +0.019 [-0.03, +0.07] | inconclusive |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k30 | -0.029 [-0.09, +0.00] | +0.012 [-0.00, +0.03] | +0.008 [-0.00, +0.02] | inconclusive |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k100 | -0.029 [-0.09, +0.00] | -0.003 [-0.01, +0.00] | -0.012 [-0.03, -0.00] | worse |
| R5 | r256-hybrid-bge-k60 | r256-hybrid-bge-k60-rerank | +0.057 [-0.09, +0.20] | +0.043 [-0.09, +0.17] | +0.073 [-0.04, +0.19] | inconclusive |
| R5 | r512-hybrid-bge-k60 | r512-hybrid-bge-k60-rerank | -0.043 [-0.17, +0.09] | +0.019 [-0.11, +0.15] | +0.007 [-0.11, +0.12] | inconclusive |
| R6 | f256-dense-bge | r256-dense-bge | -0.043 [-0.19, +0.11] | -0.040 [-0.15, +0.08] | -0.045 [-0.15, +0.07] | inconclusive |
| R6 | f512-dense-bge | r512-dense-bge | +0.043 [-0.16, +0.24] | +0.126 [-0.02, +0.28] | +0.103 [-0.03, +0.25] | inconclusive |
| R6 | f256-hybrid-bge-k60 | r256-hybrid-bge-k60 | -0.014 [-0.13, +0.10] | +0.013 [-0.10, +0.12] | -0.011 [-0.11, +0.09] | inconclusive |
| R6 | f512-hybrid-bge-k60 | r512-hybrid-bge-k60 | +0.086 [+0.00, +0.20] | +0.124 [+0.02, +0.24] | +0.073 [-0.01, +0.16] | inconclusive |

## R7: Recall by cutoff

| Configuration | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| r512-dense-bge | 0.514 | 0.629 | 0.671 | 0.743 | 0.800 |
| r512-bm25 | 0.529 | 0.671 | 0.714 | 0.814 | 0.814 |
| r512-hybrid-bge-k60 | 0.557 | 0.743 | 0.843 | 0.871 | 0.900 |
| r512-hybrid-bge-k60-rerank | 0.614 | 0.786 | 0.800 | 0.814 | 0.900 |

## By question type, for r256-hybrid-bge-k60-rerank

Small groups, so these are exploratory only.

| Type | Questions | Recall@5 | nDCG@10 |
|---|---|---|---|
| config_code | 5 | 0.800 | 0.793 |
| multi_hop | 5 | 0.700 | 0.629 |
| paraphrase_ambiguous | 2 | 0.500 | 0.315 |
| single_hop | 17 | 0.824 | 0.768 |
| table | 4 | 1.000 | 1.000 |
| version_specific | 2 | 1.000 | 0.815 |
