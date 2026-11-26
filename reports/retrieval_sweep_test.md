# Retrieval sweep on the test split

- Split: **test**, 101 answerable questions scored (15 unanswerable skipped).
- Dataset `sha256-4648637fc865`, corpus `sha256-9e14d8a33aa1`, code `a204aae6fe`.
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU.
- Brackets are 95% bootstrap intervals. Latency is warm, per query, and excludes model loading.
- Recall counts evidence found, not chunks, so it is comparable across chunking.

## All configurations

| Configuration | Recall@5 | All evidence@5 | Recall@20 | MRR | nDCG@10 | MAP | p50 ms |
|---|---|---|---|---|---|---|---|
| r512-hybrid-bge-k60-rerank | 0.861 [0.80, 0.92] | 0.832 | 0.916 | 0.799 | 0.810 | 0.779 | 381 |
| r256-hybrid-bge-k60-rerank | 0.837 [0.76, 0.90] | 0.812 | 0.901 | 0.793 | 0.799 | 0.766 | 326 |
| r512-hybrid-bge-k10 | 0.832 [0.76, 0.90] | 0.812 | 0.955 | 0.752 | 0.771 | 0.726 | 135 |
| f256-hybrid-bge-k60-rerank | 0.802 [0.73, 0.87] | 0.752 | 0.856 | 0.745 | 0.750 | 0.705 | 323 |
| f512-hybrid-bge-k60-rerank | 0.757 [0.67, 0.83] | 0.713 | 0.842 | 0.755 | 0.750 | 0.713 | 355 |
| r512-hybrid-bge-k60 | 0.817 [0.74, 0.89] | 0.802 | 0.916 | 0.726 | 0.746 | 0.704 | 134 |
| r512-hybrid-bge-k30 | 0.827 [0.75, 0.90] | 0.812 | 0.946 | 0.728 | 0.745 | 0.704 | 134 |
| r512-hybrid-bge-k100 | 0.817 [0.74, 0.89] | 0.802 | 0.916 | 0.723 | 0.741 | 0.702 | 144 |
| r256-hybrid-bge-k60 | 0.797 [0.72, 0.87] | 0.772 | 0.901 | 0.684 | 0.715 | 0.660 | 175 |
| r512-bm25 | 0.748 [0.66, 0.83] | 0.723 | 0.876 | 0.640 | 0.673 | 0.620 | 110 |
| f256-hybrid-bge-k60 | 0.757 [0.68, 0.83] | 0.713 | 0.856 | 0.665 | 0.671 | 0.630 | 151 |
| f512-bm25 | 0.748 [0.66, 0.83] | 0.713 | 0.871 | 0.660 | 0.667 | 0.629 | 70 |
| r512-dense-bge | 0.738 [0.65, 0.82] | 0.703 | 0.866 | 0.631 | 0.654 | 0.606 | 15 |
| r256-dense-bge | 0.723 [0.64, 0.80] | 0.683 | 0.876 | 0.624 | 0.647 | 0.593 | 18 |
| f512-hybrid-bge-k60 | 0.733 [0.65, 0.81] | 0.693 | 0.842 | 0.625 | 0.645 | 0.593 | 78 |
| f256-bm25 | 0.688 [0.60, 0.77] | 0.644 | 0.847 | 0.615 | 0.635 | 0.582 | 132 |
| r256-bm25 | 0.683 [0.59, 0.77] | 0.644 | 0.817 | 0.612 | 0.633 | 0.587 | 162 |
| f256-dense-bge | 0.688 [0.60, 0.77] | 0.653 | 0.807 | 0.587 | 0.608 | 0.553 | 17 |
| r256-dense-minilm | 0.644 [0.55, 0.73] | 0.614 | 0.812 | 0.564 | 0.591 | 0.545 | 14 |
| r512-dense-minilm | 0.629 [0.53, 0.72] | 0.594 | 0.787 | 0.579 | 0.591 | 0.551 | 15 |
| f512-dense-bge | 0.594 [0.50, 0.68] | 0.564 | 0.733 | 0.561 | 0.565 | 0.531 | 7 |
| f256-dense-minilm | 0.619 [0.53, 0.71] | 0.574 | 0.743 | 0.502 | 0.535 | 0.474 | 14 |
| f512-dense-minilm | 0.515 [0.42, 0.61] | 0.475 | 0.683 | 0.438 | 0.457 | 0.404 | 7 |

## Paired comparisons

Differences are treatment minus baseline on identical questions. A result is only called better or worse when its interval excludes zero.

| Exp | Baseline | Treatment | Δ Recall@5 | Δ MRR | Δ nDCG@10 | Verdict (nDCG@10) |
|---|---|---|---|---|---|---|
| R1 | r256-dense-minilm | r256-dense-bge | +0.079 [-0.00, +0.16] | +0.061 [-0.01, +0.13] | +0.056 [-0.00, +0.11] | inconclusive |
| R1 | r512-dense-minilm | r512-dense-bge | +0.109 [+0.01, +0.20] | +0.052 [-0.02, +0.12] | +0.062 [-0.00, +0.13] | inconclusive |
| R2 | r256-dense-bge | r256-bm25 | -0.040 [-0.14, +0.06] | -0.012 [-0.10, +0.08] | -0.014 [-0.10, +0.07] | inconclusive |
| R2 | r512-dense-bge | r512-bm25 | +0.010 [-0.09, +0.11] | +0.009 [-0.08, +0.10] | +0.020 [-0.06, +0.10] | inconclusive |
| R3 | r256-dense-bge | r256-hybrid-bge-k60 | +0.074 [-0.01, +0.16] | +0.060 [-0.00, +0.12] | +0.068 [+0.02, +0.12] | better |
| R3 | r512-dense-bge | r512-hybrid-bge-k60 | +0.079 [+0.00, +0.16] | +0.095 [+0.03, +0.16] | +0.093 [+0.03, +0.16] | better |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k10 | +0.015 [-0.01, +0.05] | +0.026 [+0.01, +0.05] | +0.025 [+0.00, +0.05] | better |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k30 | +0.010 [+0.00, +0.03] | +0.002 [-0.00, +0.01] | -0.001 [-0.01, +0.01] | inconclusive |
| R4 | r512-hybrid-bge-k60 | r512-hybrid-bge-k100 | +0.000 [+0.00, +0.00] | -0.003 [-0.01, +0.00] | -0.005 [-0.01, +0.00] | inconclusive |
| R5 | r256-hybrid-bge-k60 | r256-hybrid-bge-k60-rerank | +0.040 [-0.01, +0.09] | +0.108 [+0.04, +0.18] | +0.084 [+0.03, +0.14] | better |
| R5 | r512-hybrid-bge-k60 | r512-hybrid-bge-k60-rerank | +0.045 [-0.01, +0.10] | +0.073 [+0.01, +0.14] | +0.064 [+0.01, +0.12] | better |
| R6 | f256-dense-bge | r256-dense-bge | +0.035 [-0.04, +0.11] | +0.037 [-0.04, +0.11] | +0.039 [-0.02, +0.10] | inconclusive |
| R6 | f512-dense-bge | r512-dense-bge | +0.144 [+0.04, +0.24] | +0.071 [-0.02, +0.16] | +0.088 [+0.01, +0.17] | better |
| R6 | f256-hybrid-bge-k60 | r256-hybrid-bge-k60 | +0.040 [-0.03, +0.11] | +0.020 [-0.04, +0.08] | +0.044 [-0.01, +0.10] | inconclusive |
| R6 | f512-hybrid-bge-k60 | r512-hybrid-bge-k60 | +0.084 [+0.00, +0.16] | +0.101 [+0.03, +0.18] | +0.102 [+0.03, +0.17] | better |

## R7: Recall by cutoff

| Configuration | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| r512-dense-bge | 0.490 | 0.673 | 0.738 | 0.802 | 0.866 |
| r512-bm25 | 0.505 | 0.673 | 0.748 | 0.837 | 0.876 |
| r512-hybrid-bge-k60 | 0.574 | 0.797 | 0.817 | 0.871 | 0.916 |
| r512-hybrid-bge-k60-rerank | 0.678 | 0.847 | 0.861 | 0.901 | 0.916 |

## By question type, for r512-hybrid-bge-k60-rerank

Small groups, so these are exploratory only.

| Type | Questions | Recall@5 | nDCG@10 |
|---|---|---|---|
| config_code | 16 | 0.875 | 0.798 |
| multi_hop | 13 | 0.769 | 0.792 |
| paraphrase_ambiguous | 6 | 0.833 | 0.833 |
| single_hop | 46 | 0.913 | 0.866 |
| table | 10 | 0.700 | 0.546 |
| version_specific | 10 | 0.900 | 0.850 |
