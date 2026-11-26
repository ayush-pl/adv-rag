# Retrieval sweep on the development split

- Split: **development**, 34 answerable questions scored (4 unanswerable skipped).
- Dataset `sha256-d14d8e1dd615`, corpus `sha256-9e14d8a33aa1`, code `7862ce149d` with uncommitted changes.
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU.
- Brackets are 95% bootstrap intervals. Latency is warm, per query, and excludes model loading.
- Recall counts evidence found, not chunks, so it is comparable across chunking.

## All configurations

| Configuration | Recall@5 | All evidence@5 | Recall@20 | MRR | nDCG@10 | MAP | p50 ms |
|---|---|---|---|---|---|---|---|
| r256-hybrid-bge-k60-rerank | 0.824 [0.69, 0.94] | 0.794 | 0.897 | 0.748 | 0.768 | 0.733 | 319 |
| r512-hybrid-bge-k10 | 0.794 [0.66, 0.91] | 0.765 | 0.956 | 0.760 | 0.756 | 0.737 | 124 |
| r512-hybrid-bge-k60-rerank | 0.809 [0.68, 0.93] | 0.765 | 0.912 | 0.763 | 0.755 | 0.735 | 366 |
| r512-hybrid-bge-k30 | 0.824 [0.69, 0.94] | 0.794 | 0.956 | 0.733 | 0.745 | 0.708 | 126 |
| r512-hybrid-bge-k60 | 0.853 [0.74, 0.96] | 0.824 | 0.912 | 0.721 | 0.737 | 0.692 | 130 |
| r512-hybrid-bge-k100 | 0.824 [0.69, 0.94] | 0.794 | 0.912 | 0.718 | 0.725 | 0.685 | 127 |
| f256-hybrid-bge-k60-rerank | 0.809 [0.68, 0.93] | 0.794 | 0.912 | 0.688 | 0.713 | 0.672 | 314 |
| f512-hybrid-bge-k60-rerank | 0.824 [0.68, 0.94] | 0.824 | 0.912 | 0.654 | 0.701 | 0.651 | 344 |
| f256-hybrid-bge-k60 | 0.779 [0.63, 0.91] | 0.765 | 0.912 | 0.671 | 0.696 | 0.659 | 140 |
| r512-bm25 | 0.721 [0.57, 0.85] | 0.676 | 0.824 | 0.682 | 0.691 | 0.641 | 101 |
| r256-hybrid-bge-k60 | 0.765 [0.62, 0.90] | 0.735 | 0.897 | 0.684 | 0.684 | 0.652 | 172 |
| f512-hybrid-bge-k60 | 0.765 [0.62, 0.90] | 0.735 | 0.912 | 0.617 | 0.673 | 0.607 | 78 |
| f256-bm25 | 0.706 [0.54, 0.85] | 0.676 | 0.838 | 0.662 | 0.653 | 0.631 | 114 |
| r256-bm25 | 0.750 [0.60, 0.88] | 0.706 | 0.838 | 0.615 | 0.640 | 0.577 | 135 |
| r512-dense-bge | 0.676 [0.53, 0.82] | 0.647 | 0.809 | 0.625 | 0.638 | 0.606 | 16 |
| f256-dense-bge | 0.691 [0.53, 0.84] | 0.676 | 0.853 | 0.606 | 0.635 | 0.588 | 20 |
| f512-bm25 | 0.750 [0.60, 0.88] | 0.706 | 0.794 | 0.582 | 0.622 | 0.571 | 57 |
| r256-dense-bge | 0.647 [0.49, 0.79] | 0.618 | 0.750 | 0.565 | 0.589 | 0.548 | 20 |
| r256-dense-minilm | 0.544 [0.38, 0.71] | 0.500 | 0.779 | 0.532 | 0.543 | 0.508 | 15 |
| r512-dense-minilm | 0.647 [0.49, 0.79] | 0.618 | 0.794 | 0.528 | 0.542 | 0.505 | 14 |
| f512-dense-bge | 0.632 [0.47, 0.79] | 0.618 | 0.868 | 0.495 | 0.532 | 0.492 | 10 |
| f256-dense-minilm | 0.603 [0.44, 0.76] | 0.559 | 0.838 | 0.478 | 0.500 | 0.453 | 14 |
| f512-dense-minilm | 0.485 [0.32, 0.65] | 0.441 | 0.750 | 0.443 | 0.490 | 0.420 | 6 |

## Paired comparisons

Differences are treatment minus baseline on identical questions. A result is only called better or worse when its interval excludes zero.

| Exp | Baseline | Treatment | Δ Recall@5 | Δ MRR | Δ nDCG@10 | Verdict (nDCG@10) |
|---|---|---|---|---|---|---|
| R1 | r256-dense-minilm | r256-dense-bge | +0.103 [-0.01, +0.24] | +0.033 [-0.09, +0.16] | +0.046 [-0.06, +0.16] | inconclusive |
| R1 | r512-dense-minilm | r512-dense-bge | +0.029 [-0.09, +0.15] | +0.097 [-0.03, +0.23] | +0.095 [-0.01, +0.21] | inconclusive |
| R2 | r256-dense-bge | r256-bm25 | +0.103 [-0.06, +0.26] | +0.051 [-0.12, +0.23] | +0.051 [-0.10, +0.22] | inconclusive |
| R2 | r512-dense-bge | r512-bm25 | +0.044 [-0.12, +0.21] | +0.057 [-0.11, +0.22] | +0.053 [-0.10, +0.20] | inconclusive |
| R3 | r256-dense-bge | r256-hybrid-bge-k60 | +0.118 [-0.01, +0.26] | +0.119 [-0.01, +0.25] | +0.095 [-0.01, +0.21] | inconclusive |
| R3 | r512-dense-bge | r512-hybrid-bge-k60 | +0.176 [+0.04, +0.32] | +0.096 [-0.04, +0.23] | +0.099 [-0.01, +0.21] | inconclusive |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k10 | -0.059 [-0.15, +0.00] | +0.039 [-0.00, +0.09] | +0.019 [-0.03, +0.07] | inconclusive |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k30 | -0.029 [-0.09, +0.00] | +0.012 [-0.00, +0.03] | +0.008 [-0.00, +0.02] | inconclusive |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k100 | -0.029 [-0.09, +0.00] | -0.003 [-0.01, +0.00] | -0.012 [-0.03, -0.00] | worse |
| R5 | r256-hybrid-bge-k60 | r256-hybrid-bge-k60-rerank | +0.059 [-0.09, +0.21] | +0.064 [-0.06, +0.20] | +0.084 [-0.03, +0.21] | inconclusive |
| R5 | r512-hybrid-bge-k60 | r512-hybrid-bge-k60-rerank | -0.044 [-0.18, +0.09] | +0.042 [-0.08, +0.17] | +0.018 [-0.10, +0.13] | inconclusive |
| R6 | f256-dense-bge | r256-dense-bge | -0.044 [-0.21, +0.12] | -0.041 [-0.16, +0.08] | -0.046 [-0.16, +0.07] | inconclusive |
| R6 | f512-dense-bge | r512-dense-bge | +0.044 [-0.16, +0.24] | +0.130 [-0.01, +0.28] | +0.106 [-0.03, +0.25] | inconclusive |
| R6 | f256-hybrid-bge-k60 | r256-hybrid-bge-k60 | -0.015 [-0.13, +0.10] | +0.013 [-0.10, +0.12] | -0.011 [-0.12, +0.09] | inconclusive |
| R6 | f512-hybrid-bge-k60 | r512-hybrid-bge-k60 | +0.088 [+0.00, +0.21] | +0.104 [+0.00, +0.21] | +0.064 [-0.02, +0.15] | inconclusive |

## R7: Recall by cutoff

| Configuration | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| r512-dense-bge | 0.515 | 0.632 | 0.676 | 0.750 | 0.809 |
| r512-bm25 | 0.529 | 0.676 | 0.721 | 0.824 | 0.824 |
| r512-hybrid-bge-k60 | 0.559 | 0.750 | 0.853 | 0.882 | 0.912 |
| r512-hybrid-bge-k60-rerank | 0.632 | 0.809 | 0.809 | 0.824 | 0.912 |

## By question type, for r256-hybrid-bge-k60-rerank

Small groups, so these are exploratory only.

| Type | Questions | Recall@5 | nDCG@10 |
|---|---|---|---|
| config_code | 5 | 0.800 | 0.793 |
| multi_hop | 4 | 0.750 | 0.710 |
| paraphrase_ambiguous | 2 | 0.500 | 0.315 |
| single_hop | 17 | 0.824 | 0.768 |
| table | 4 | 1.000 | 1.000 |
| version_specific | 2 | 1.000 | 0.815 |
