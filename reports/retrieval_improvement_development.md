# Duplicate removal on the development split

- Split: **development**, 34 answerable questions scored (4 unanswerable skipped).
- Dataset `sha256-72eed53de9d7`, corpus `sha256-9e14d8a33aa1`, code `7862ce149d` with uncommitted changes.
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU.
- Brackets are 95% bootstrap intervals. Latency is warm, per query, and excludes model loading.
- Recall counts evidence found, not chunks, so it is comparable across chunking.

## All configurations

| Configuration | Recall@5 | All evidence@5 | Recall@20 | MRR | nDCG@10 | MAP | p50 ms |
|---|---|---|---|---|---|---|---|
| r512-hybrid-bge-k60-dedup-rerank | 0.868 [0.75, 0.97] | 0.853 | 0.971 | 0.799 | 0.829 | 0.799 | 362 |
| r256-hybrid-bge-k60-dedup-rerank | 0.897 [0.79, 0.99] | 0.882 | 0.941 | 0.789 | 0.827 | 0.788 | 325 |
| r256-hybrid-bge-k60-rerank | 0.824 [0.69, 0.94] | 0.794 | 0.897 | 0.748 | 0.768 | 0.733 | 319 |
| r512-hybrid-bge-k60-dedup | 0.853 [0.74, 0.96] | 0.824 | 0.971 | 0.739 | 0.759 | 0.715 | 127 |
| r512-hybrid-bge-k60-rerank | 0.809 [0.68, 0.93] | 0.765 | 0.912 | 0.763 | 0.755 | 0.735 | 366 |
| r512-hybrid-bge-k60 | 0.853 [0.74, 0.96] | 0.824 | 0.912 | 0.721 | 0.737 | 0.692 | 130 |
| r256-hybrid-bge-k60-dedup | 0.765 [0.62, 0.90] | 0.735 | 0.941 | 0.699 | 0.712 | 0.671 | 153 |
| r256-hybrid-bge-k60 | 0.765 [0.62, 0.90] | 0.735 | 0.897 | 0.684 | 0.684 | 0.652 | 172 |
| r512-dense-bge-dedup | 0.721 [0.56, 0.85] | 0.706 | 0.882 | 0.642 | 0.660 | 0.624 | 19 |
| r512-dense-bge | 0.676 [0.53, 0.82] | 0.647 | 0.809 | 0.625 | 0.638 | 0.606 | 16 |

## Paired comparisons

Differences are treatment minus baseline on identical questions. A result is only called better or worse when its interval excludes zero.

| Exp | Baseline | Treatment | Δ Recall@5 | Δ MRR | Δ nDCG@10 | Verdict (nDCG@10) |
|---|---|---|---|---|---|---|
| D1 | r512-dense-bge | r512-dense-bge-dedup | +0.044 [+0.00, +0.12] | +0.017 [+0.00, +0.03] | +0.022 [+0.01, +0.04] | better |
| D1 | r256-hybrid-bge-k60 | r256-hybrid-bge-k60-dedup | +0.000 [+0.00, +0.00] | +0.015 [+0.00, +0.03] | +0.028 [+0.01, +0.06] | better |
| D1 | r512-hybrid-bge-k60 | r512-hybrid-bge-k60-dedup | +0.000 [+0.00, +0.00] | +0.017 [+0.00, +0.03] | +0.022 [+0.00, +0.05] | better |
| D1 | r256-hybrid-bge-k60-rerank | r256-hybrid-bge-k60-dedup-rerank | +0.074 [+0.00, +0.18] | +0.042 [+0.00, +0.11] | +0.058 [+0.01, +0.13] | better |
| D1 | r512-hybrid-bge-k60-rerank | r512-hybrid-bge-k60-dedup-rerank | +0.059 [+0.00, +0.13] | +0.037 [+0.00, +0.10] | +0.074 [+0.02, +0.15] | better |

## R7: Recall by cutoff

| Configuration | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| r512-hybrid-bge-k60-rerank | 0.632 | 0.809 | 0.809 | 0.824 | 0.912 |
| r512-hybrid-bge-k60-dedup-rerank | 0.662 | 0.868 | 0.868 | 0.941 | 0.971 |

## By question type, for r512-hybrid-bge-k60-dedup-rerank

Small groups, so these are exploratory only.

| Type | Questions | Recall@5 | nDCG@10 |
|---|---|---|---|
| config_code | 5 | 0.800 | 0.793 |
| multi_hop | 4 | 0.875 | 0.881 |
| paraphrase_ambiguous | 2 | 0.500 | 0.315 |
| single_hop | 17 | 0.882 | 0.849 |
| table | 4 | 1.000 | 1.000 |
| version_specific | 2 | 1.000 | 0.815 |
