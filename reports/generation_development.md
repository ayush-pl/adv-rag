# Generation results on the development split

- Judge `qwen2.5:7b` with prompt `judge-v2`; generation prompt `grounded-v1`, temperature 0.0, top 5 passages.
- **Read with reports/judge_calibration.md**, which measures how well the judge agrees with human scores. Every score here is the judge's.
- On held-out human scores the judge was harsher than the human and almost never called a wrong answer correct, so **correct rates are conservative**. Comparisons between runs share that bias, so they are the more reliable reading.
- The judge's faithfulness scores did **not** agree with human scores: it tends to mark answers that miss the question as unsupported. Treat the faithfulness and low-faithfulness columns as indicative only.
- *Correct* means a correctness score of 4 or 5. Abstaining on, or failing to answer, an answerable question counts as incorrect. *Low faithfulness* means a score of 1 or 2, among answers given.
- 34 answerable and 4 unanswerable questions. Oracle runs give the gold passages and cover answerable questions only.
- Latency is steady-state generation time per answer, excluding model loading.

## All runs

| Model | Context | Correct [95% CI] | Correctness | Faithfulness | Low faithfulness | Relevance | False abstain | Refused unanswerable | Failed output | Cite precision | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.1:8b | r512-dense-bge | 44% [26%, 62%] | 3.00 | 3.38 | 31% | 3.79 | 15% | 75% | 0 | 65% | 1255 |
| llama3.1:8b | r512-hybrid-bge-k60 | 68% [53%, 82%] | 3.71 | 3.94 | 24% | 4.15 | 3% | 50% | 0 | 78% | 1310 |
| llama3.1:8b | r256-hybrid-bge-k60-rerank | 62% [47%, 76%] | 3.65 | 3.82 | 21% | 4.06 | 3% | 50% | 0 | 67% | 1053 |
| llama3.1:8b | oracle r256-overlap50 | 85% [74%, 97%] | 4.35 | 4.65 | 6% | 4.79 | 0% | – | 0 | 100% | 675 |
| llama3.1:8b | oracle r512-overlap50 | 82% [68%, 94%] | 4.41 | 4.59 | 3% | 4.82 | 0% | – | 0 | 100% | 712 |
| gemma3:4b | r512-dense-bge | 38% [24%, 56%] | 2.88 | 3.34 | 31% | 3.79 | 15% | 0% | 0 | 69% | 1567 |
| gemma3:4b | r512-hybrid-bge-k60 | 59% [41%, 74%] | 3.44 | 3.88 | 18% | 4.09 | 0% | 25% | 2 | 68% | 1478 |
| gemma3:4b | r256-hybrid-bge-k60-rerank | 59% [41%, 76%] | 3.59 | 3.71 | 24% | 4.18 | 0% | 50% | 0 | 70% | 1360 |
| gemma3:4b | oracle r256-overlap50 | 88% [76%, 97%] | 4.32 | 4.79 | 0% | 4.82 | 0% | – | 0 | 100% | 1104 |
| gemma3:4b | oracle r512-overlap50 | 88% [76%, 97%] | 4.29 | 4.74 | 0% | 4.82 | 0% | – | 0 | 100% | 1206 |

## Paired comparisons of correct answers

Same model, same questions. McNemar's exact test on correct or not, with a paired bootstrap interval for the difference in correct rate.

| Model | Baseline | Treatment | Δ correct [95% CI] | Only baseline right | Only treatment right | p |
|---|---|---|---|---|---|---|
| llama3.1:8b | r512-dense-bge | r512-hybrid-bge-k60 | +24% [+9%, +38%] | 0 | 8 | 0.008 |
| llama3.1:8b | r512-hybrid-bge-k60 | r256-hybrid-bge-k60-rerank | -6% [-24%, +9%] | 5 | 3 | 0.727 |
| llama3.1:8b | r512-dense-bge | r256-hybrid-bge-k60-rerank | +18% [+0%, +35%] | 3 | 9 | 0.146 |
| gemma3:4b | r512-dense-bge | r512-hybrid-bge-k60 | +21% [+0%, +41%] | 3 | 10 | 0.092 |
| gemma3:4b | r512-hybrid-bge-k60 | r256-hybrid-bge-k60-rerank | +0% [-21%, +21%] | 6 | 6 | 1.000 |
| gemma3:4b | r512-dense-bge | r256-hybrid-bge-k60-rerank | +21% [+0%, +41%] | 3 | 10 | 0.092 |
| gemma3:4b vs llama3.1:8b | r512-dense-bge | same | -6% [-21%, +9%] | 5 | 3 | 0.727 |
| gemma3:4b vs llama3.1:8b | r512-hybrid-bge-k60 | same | -9% [-24%, +6%] | 5 | 2 | 0.453 |
| gemma3:4b vs llama3.1:8b | r256-hybrid-bge-k60-rerank | same | -3% [-24%, +15%] | 6 | 5 | 1.000 |

## Oracle-context diagnosis

Each retrieved run against the same model given the gold passages, at the same chunking. *Retrieval failures* were right with gold passages but wrong with retrieved ones. *Generation failures* were wrong even with gold passages.

| Model | Retrieval | Questions | Correct, retrieved | Correct, oracle | Both right | Retrieval failures | Generation failures | Right only when retrieved |
|---|---|---|---|---|---|---|---|---|
| llama3.1:8b | r512-dense-bge | 34 | 44% | 82% | 12 | 16 | 6 | 3 |
| llama3.1:8b | r512-hybrid-bge-k60 | 34 | 68% | 82% | 19 | 9 | 6 | 4 |
| llama3.1:8b | r256-hybrid-bge-k60-rerank | 34 | 62% | 85% | 19 | 10 | 5 | 2 |
| gemma3:4b | r512-dense-bge | 34 | 38% | 88% | 11 | 19 | 4 | 2 |
| gemma3:4b | r512-hybrid-bge-k60 | 34 | 59% | 88% | 18 | 12 | 4 | 2 |
| gemma3:4b | r256-hybrid-bge-k60-rerank | 34 | 59% | 88% | 19 | 11 | 4 | 1 |

## Reference answers to check

0 questions were answered badly (correctness 2 or lower) by every oracle run, despite having the gold passages. The reference answer or the question is the likely problem, so these need a human check before the results are final.

None.
