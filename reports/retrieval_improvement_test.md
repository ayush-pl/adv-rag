# Duplicate removal on the test split

- Split: **test**, 101 answerable questions scored (15 unanswerable skipped).
- Dataset `sha256-4648637fc865`, corpus `sha256-9e14d8a33aa1`, code `a204aae6fe`.
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU.
- Brackets are 95% bootstrap intervals. Latency is warm, per query, and excludes model loading.
- Recall counts evidence found, not chunks, so it is comparable across chunking.

## All configurations

| Configuration | Recall@5 | All evidence@5 | Recall@20 | MRR | nDCG@10 | MAP | p50 ms |
|---|---|---|---|---|---|---|---|
| r512-hybrid-bge-k60-dedup-rerank | 0.861 [0.80, 0.92] | 0.832 | 0.926 | 0.809 | 0.820 | 0.789 | 374 |
| r256-hybrid-bge-k60-dedup-rerank | 0.856 [0.79, 0.92] | 0.832 | 0.906 | 0.802 | 0.812 | 0.778 | 318 |
| r512-hybrid-bge-k60-rerank | 0.861 [0.80, 0.92] | 0.832 | 0.916 | 0.799 | 0.810 | 0.779 | 381 |
| r256-hybrid-bge-k60-rerank | 0.837 [0.76, 0.90] | 0.812 | 0.901 | 0.793 | 0.799 | 0.766 | 326 |
| r512-hybrid-bge-k60-dedup | 0.827 [0.75, 0.90] | 0.812 | 0.926 | 0.734 | 0.757 | 0.712 | 133 |
| r512-hybrid-bge-k60 | 0.817 [0.74, 0.89] | 0.802 | 0.916 | 0.726 | 0.746 | 0.704 | 134 |
| r256-hybrid-bge-k60-dedup | 0.817 [0.74, 0.89] | 0.792 | 0.906 | 0.697 | 0.728 | 0.675 | 169 |
| r256-hybrid-bge-k60 | 0.797 [0.72, 0.87] | 0.772 | 0.901 | 0.684 | 0.715 | 0.660 | 175 |
| r512-dense-bge-dedup | 0.743 [0.66, 0.82] | 0.713 | 0.876 | 0.640 | 0.671 | 0.616 | 16 |
| r512-dense-bge | 0.738 [0.65, 0.82] | 0.703 | 0.866 | 0.631 | 0.654 | 0.606 | 15 |

## Paired comparisons

Differences are treatment minus baseline on identical questions. A result is only called better or worse when its interval excludes zero.

| Exp | Baseline | Treatment | Δ Recall@5 | Δ MRR | Δ nDCG@10 | Verdict (nDCG@10) |
|---|---|---|---|---|---|---|
| D1 | r512-dense-bge | r512-dense-bge-dedup | +0.005 [+0.00, +0.01] | +0.009 [+0.00, +0.02] | +0.017 [+0.01, +0.03] | better |
| D1 | r256-hybrid-bge-k60 | r256-hybrid-bge-k60-dedup | +0.020 [+0.00, +0.05] | +0.012 [+0.01, +0.02] | +0.013 [+0.01, +0.02] | better |
| D1 | r512-hybrid-bge-k60 | r512-hybrid-bge-k60-dedup | +0.010 [+0.00, +0.03] | +0.008 [+0.00, +0.01] | +0.011 [+0.00, +0.02] | better |
| D1 | r256-hybrid-bge-k60-rerank | r256-hybrid-bge-k60-dedup-rerank | +0.020 [+0.00, +0.05] | +0.009 [-0.00, +0.03] | +0.013 [+0.00, +0.03] | better |
| D1 | r512-hybrid-bge-k60-rerank | r512-hybrid-bge-k60-dedup-rerank | +0.000 [+0.00, +0.00] | +0.010 [+0.00, +0.02] | +0.010 [+0.00, +0.02] | better |

## R7: Recall by cutoff

| Configuration | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| r512-hybrid-bge-k60-rerank | 0.678 | 0.847 | 0.861 | 0.901 | 0.916 |
| r512-hybrid-bge-k60-dedup-rerank | 0.678 | 0.847 | 0.861 | 0.911 | 0.926 |

## By question type, for r512-hybrid-bge-k60-dedup-rerank

Small groups, so these are exploratory only.

| Type | Questions | Recall@5 | nDCG@10 |
|---|---|---|---|
| config_code | 16 | 0.875 | 0.806 |
| multi_hop | 13 | 0.769 | 0.793 |
| paraphrase_ambiguous | 6 | 0.833 | 0.833 |
| single_hop | 46 | 0.913 | 0.869 |
| table | 10 | 0.700 | 0.616 |
| version_specific | 10 | 0.900 | 0.850 |
