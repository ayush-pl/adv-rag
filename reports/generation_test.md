# Generation results on the test split

- Judge `qwen2.5:7b` with prompt `judge-v2`; generation prompt `grounded-v1`, temperature 0.0, top 5 passages.
- **Read with reports/judge_calibration.md**, which measures how well the judge agrees with human scores. Every score here is the judge's.
- On held-out human scores the judge was harsher than the human and almost never called a wrong answer correct, so **correct rates are conservative**. Comparisons between runs share that bias, so they are the more reliable reading.
- The judge's faithfulness scores did **not** agree with human scores: it tends to mark answers that miss the question as unsupported. Treat the faithfulness and low-faithfulness columns as indicative only.
- *Correct* means a correctness score of 4 or 5. Abstaining on, or failing to answer, an answerable question counts as incorrect. *Low faithfulness* means a score of 1 or 2, among answers given.
- 101 answerable and 15 unanswerable questions. Oracle runs give the gold passages and cover answerable questions only.
- Latency is steady-state generation time per answer, excluding model loading.

## All runs

| Model | Context | Correct [95% CI] | Correctness | Faithfulness | Low faithfulness | Relevance | False abstain | Refused unanswerable | Failed output | Cite precision | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.1:8b | r512-dense-bge | 50% [41%, 60%] | 3.30 | 3.86 | 24% | 4.00 | 12% | 93% | 1 | 73% | 1201 |
| llama3.1:8b | r512-hybrid-bge-k60 | 55% [46%, 65%] | 3.41 | 4.02 | 17% | 4.28 | 11% | 80% | 2 | 75% | 1234 |
| llama3.1:8b | r256-hybrid-bge-k60-rerank | 62% [52%, 71%] | 3.60 | 3.96 | 20% | 4.13 | 3% | 80% | 0 | 72% | 976 |
| llama3.1:8b | oracle r256-overlap50 | 74% [65%, 82%] | 4.04 | 4.41 | 6% | 4.56 | 3% | – | 0 | 100% | 646 |
| llama3.1:8b | oracle r512-overlap50 | 70% [60%, 79%] | 4.01 | 4.35 | 6% | 4.49 | 2% | – | 0 | 100% | 674 |
| gemma3:4b | r512-dense-bge | 52% [43%, 62%] | 3.16 | 3.80 | 17% | 3.92 | 13% | 80% | 2 | 70% | 1434 |
| gemma3:4b | r512-hybrid-bge-k60 | 50% [41%, 60%] | 3.18 | 3.69 | 24% | 4.00 | 7% | 73% | 7 | 61% | 1545 |
| gemma3:4b | r256-hybrid-bge-k60-rerank | 57% [48%, 67%] | 3.40 | 3.80 | 20% | 4.16 | 4% | 47% | 6 | 71% | 1400 |
| gemma3:4b | oracle r256-overlap50 | 77% [69%, 85%] | 4.00 | 4.23 | 4% | 4.61 | 1% | – | 0 | 100% | 1147 |
| gemma3:4b | oracle r512-overlap50 | 83% [76%, 90%] | 4.06 | 4.39 | 4% | 4.75 | 1% | – | 1 | 100% | 1195 |

## Paired comparisons of correct answers

Same model, same questions. McNemar's exact test on correct or not, with a paired bootstrap interval for the difference in correct rate.

| Model | Baseline | Treatment | Δ correct [95% CI] | Only baseline right | Only treatment right | p |
|---|---|---|---|---|---|---|
| llama3.1:8b | r512-dense-bge | r512-hybrid-bge-k60 | +5% [-5%, +14%] | 9 | 14 | 0.405 |
| llama3.1:8b | r512-hybrid-bge-k60 | r256-hybrid-bge-k60-rerank | +7% [-3%, +17%] | 11 | 18 | 0.265 |
| llama3.1:8b | r512-dense-bge | r256-hybrid-bge-k60-rerank | +12% [+1%, +23%] | 9 | 21 | 0.043 |
| gemma3:4b | r512-dense-bge | r512-hybrid-bge-k60 | -2% [-13%, +9%] | 17 | 15 | 0.860 |
| gemma3:4b | r512-hybrid-bge-k60 | r256-hybrid-bge-k60-rerank | +7% [-4%, +18%] | 13 | 20 | 0.296 |
| gemma3:4b | r512-dense-bge | r256-hybrid-bge-k60-rerank | +5% [-8%, +18%] | 19 | 24 | 0.542 |
| gemma3:4b vs llama3.1:8b | r512-dense-bge | same | +2% [-9%, +13%] | 15 | 17 | 0.860 |
| gemma3:4b vs llama3.1:8b | r512-hybrid-bge-k60 | same | -5% [-16%, +6%] | 19 | 14 | 0.487 |
| gemma3:4b vs llama3.1:8b | r256-hybrid-bge-k60-rerank | same | -5% [-16%, +6%] | 18 | 13 | 0.473 |

## Oracle-context diagnosis

Each retrieved run against the same model given the gold passages, at the same chunking. *Retrieval failures* were right with gold passages but wrong with retrieved ones. *Generation failures* were wrong even with gold passages.

| Model | Retrieval | Questions | Correct, retrieved | Correct, oracle | Both right | Retrieval failures | Generation failures | Right only when retrieved |
|---|---|---|---|---|---|---|---|---|
| llama3.1:8b | r512-dense-bge | 101 | 50% | 70% | 46 | 25 | 30 | 5 |
| llama3.1:8b | r512-hybrid-bge-k60 | 101 | 55% | 70% | 46 | 25 | 30 | 10 |
| llama3.1:8b | r256-hybrid-bge-k60-rerank | 101 | 62% | 74% | 54 | 21 | 26 | 9 |
| gemma3:4b | r512-dense-bge | 101 | 52% | 83% | 49 | 35 | 17 | 4 |
| gemma3:4b | r512-hybrid-bge-k60 | 101 | 50% | 83% | 48 | 36 | 17 | 3 |
| gemma3:4b | r256-hybrid-bge-k60-rerank | 101 | 57% | 77% | 49 | 29 | 23 | 9 |

## Reference answers to check

0 questions were answered badly (correctness 2 or lower) by every oracle run, despite having the gold passages. The reference answer or the question is the likely problem, so these need a human check before the results are final.

None.
