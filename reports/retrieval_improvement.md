# Duplicate removal

Using the benchmark to find and fix a retrieval failure, then confirming the fix on held-out data.

**Result.** Removing duplicate passages improved ranking on held-out test questions: nDCG@10 went
from 0.810 to 0.820 (+0.010, 95% interval [+0.003, +0.019]) and MRR from 0.799 to 0.809. Recall@5
did not change. It made no test question worse. On the questions it was chosen on, the gain was
five times larger, so most of that first estimate did not carry over.

## The failure

PostgreSQL 15 and 16 share most of their documentation, so 42% of the 512-token chunks have a
word-for-word twin. Looking at where the best configuration (hybrid retrieval with cross-encoder
reranking, `r512-hybrid-bge-k60-rerank`) missed evidence showed the twins taking result slots:

| Split | Top-5 slots holding a copy of a higher-ranked passage | Questions affected |
|---|---|---|
| Development | 18% | 18 of 34 |
| Calibration | 13% | 14 of 36 |
| Test | 11% | 32 of 101 |

In several multi-hop misses, the first passage and its twin filled two slots while the second
piece of evidence sat at rank 7 to 9.

## The fix

`DeduplicatingRetriever` ([dedup.py](../src/ragbench/retrieval/dedup.py)) drops a passage whose text
repeats one ranked above it, ignoring whitespace and case, and fills the slot with the next new
passage. It runs before reranking, so the reranker also sees 20 distinct candidates. When the
question names a version ("PostgreSQL 16"), it keeps that version's copy. It needs no model or
index changes and adds no measurable latency.

## Method

1. **Selection** on development and calibration together (70 answerable questions; development
   alone has 34). Five configurations were compared with and without duplicate removal
   ([retrieval_improvement.yaml](../configs/retrieval_improvement.yaml)).
2. **Decision recorded before the test run** ([retrieval_improvement_decision.md](retrieval_improvement_decision.md)):
   `r512-hybrid-bge-k60-dedup-rerank` against `r512-hybrid-bge-k60-rerank`, primary metrics nDCG@10
   and Recall@5, an improvement only if the paired 95% interval excludes zero.
3. **One test run.** Nothing was changed after it.
4. **Completing the test labels.** The test labels were pooled from configurations without
   duplicate removal, so 53 of the 60 passages it newly brought into the top 5 had never been
   judged, and would have counted as irrelevant. A human judged, blind to which system found them,
   every passage in one configuration's top 5 but not the other's: 54 passages over 34 questions.
   5 were relevant and became gold evidence. The comparison below uses the completed labels;
   passages both configurations returned count the same for both, so they cannot change it.

## Results

Selected configuration against the same configuration without duplicate removal. Paired, treatment
minus baseline, 95% bootstrap intervals.

| Split | Questions | Metric | Without | With | Δ [95% CI] | Better / worse |
|---|---|---|---|---|---|---|
| Dev + calibration (selection) | 70 | Recall@5 | 0.779 | 0.836 | +0.057 [+0.014, +0.107] | 6 / 0 |
| | | nDCG@10 | 0.726 | 0.779 | +0.054 [+0.021, +0.096] | 11 / 0 |
| | | MRR | 0.719 | 0.749 | +0.030 [+0.005, +0.067] | 9 / 0 |
| **Test (confirmation)** | 101 | Recall@5 | 0.861 | 0.861 | +0.000 | 0 / 0 |
| | | **nDCG@10** | 0.810 | **0.820** | **+0.010 [+0.003, +0.019]** | 8 / 0 |
| | | MRR | 0.799 | 0.809 | +0.010 [+0.003, +0.018] | 7 / 0 |

Every configuration, on test:

| Baseline | Δ Recall@5 | Δ nDCG@10 | Δ MRR |
|---|---|---|---|
| r512-dense-bge | +0.005 [+0.00, +0.01] | +0.017 [+0.01, +0.03] | +0.009 [+0.00, +0.02] |
| r256-hybrid-bge-k60 | +0.020 [+0.00, +0.05] | +0.013 [+0.01, +0.02] | +0.012 [+0.01, +0.02] |
| r512-hybrid-bge-k60 | +0.010 [+0.00, +0.03] | +0.011 [+0.00, +0.02] | +0.008 [+0.00, +0.01] |
| r256-hybrid-bge-k60-rerank | +0.020 [+0.00, +0.05] | +0.013 [+0.00, +0.03] | +0.009 [-0.00, +0.03] |
| r512-hybrid-bge-k60-rerank | +0.000 | +0.010 [+0.00, +0.02] | +0.010 [+0.00, +0.02] |

Full per-configuration tables: [development](retrieval_improvement_development.md),
[calibration](retrieval_improvement_calibration.md), [test](retrieval_improvement_test.md).

## Reading the result

- **Small but consistent.** Duplicate removal raised nDCG@10 for every configuration on every split
  and lowered no metric beyond rounding. For the selected configuration it made no question worse
  on any split.
- **It mostly reorders.** On test the evidence was already in the top 5; duplicate removal moved it
  up. That raises nDCG and MRR but not Recall@5. Generation reads only the top 5, so answers may
  improve less than nDCG suggests; this was not measured.
- **The selection estimate was too optimistic.** The test gain is a fifth of the selection gain.
  Test has fewer duplicates (11% of slots against 13 to 18%), the 70 selection questions give
  wide intervals, and choosing the best of five configurations favours one that did well by chance.
  The selection labels were also not completed for the new passages, unlike test's.
- **The benefit is specific to this corpus.** It comes from two near-identical PostgreSQL versions.
  A corpus without duplicated documentation would gain little.
