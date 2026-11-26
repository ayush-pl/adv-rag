# Advanced RAG-Bench

[![CI](https://github.com/ayush-pl/adv-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ayush-pl/adv-rag/actions/workflows/ci.yml)

An evaluation-first benchmark for retrieval-augmented generation (RAG) over technical documentation.
It measures retrieval and answer quality with human-verified gold labels and confidence
intervals, and only calls a difference real when a paired test says so.

The benchmark covers Kubernetes, Docker and PostgreSQL documentation, 195 human-verified questions
and 23 retrieval configurations. Two local LLMs generate answers, and a third, from another model
family, judges them after being calibrated against blind human scores.

## Results

Held-out **test split**: 116 questions, 101 answerable. Brackets are 95% bootstrap intervals.
Full tables are in [reports](reports/).

**Retrieval** ([retrieval_sweep_test.md](reports/retrieval_sweep_test.md))

| Configuration | Recall@5 | nDCG@10 | p50 latency |
|---|---|---|---|
| Dense, BGE-small | 0.738 [0.65, 0.82] | 0.654 | 15 ms |
| BM25 | 0.748 [0.66, 0.83] | 0.673 | 110 ms |
| Hybrid (RRF, k=60) | 0.817 [0.74, 0.89] | 0.746 | 134 ms |
| Hybrid + cross-encoder rerank | **0.861** [0.80, 0.92] | **0.810** | 381 ms |

All on recursive 512-token chunks. Hybrid beats dense (nDCG@10 +0.09, interval excludes zero), and
reranking adds +0.06 more. Recursive chunking beats fixed-size at 512 tokens. MiniLM, which reads
only 256 tokens, trails BGE on Recall@5 with 512-token chunks.

**Answers** ([generation_test.md](reports/generation_test.md)): share of answerable questions
answered correctly, as scored by the calibrated judge.

| Generator | Dense | Hybrid | Hybrid + rerank | Gold passages (oracle) |
|---|---|---|---|---|
| llama3.1:8b | 50% | 55% | 62% | 70–74% |
| gemma3:4b | 52% | 50% | 57% | 77–83% |

- Dense → hybrid + rerank is significant for llama (+12 points, McNemar p = 0.04).
- With the gold passages, both models do far better, so **retrieval is the main bottleneck**:
  21–36 questions per run were right with gold passages and wrong with retrieved ones.
- gemma3:4b refuses unanswerable questions less reliably (47–80% refused, against 80–93%).

**How far to trust the judge** ([judge_calibration.md](reports/judge_calibration.md)). On 24 held-out
answers scored blind by a human, qwen2.5:7b agrees well on correctness (weighted kappa 0.82
[0.57, 0.92]) and relevance (0.85), and is slightly harsher than the human, so correct rates are
conservative. Its faithfulness scores did **not** agree (kappa 0.06), so they are reported as
indicative only.

## How it works

```
corpus ──► chunks ──► indexes ──► retrieval runs ──► answers ──► judgements ──► reports
  │                                     ▲                            ▲
  └── pinned snapshots         questions + gold spans      human calibration scores
```

- **Pinned corpus.** Kubernetes v1.37 and Docker docs at fixed git commits, PostgreSQL 15 and 16
  from checksummed release tarballs ([corpus_v1.yaml](data/manifests/corpus_v1.yaml)). Ingesting
  twice gives the same content hash.
- **Gold labels as character spans**, not chunk IDs, so one set of labels scores every chunking.
  A chunk counts as evidence when it covers enough of a span.
- **Staged, cached pipeline.** Each stage is stored under its inputs' hash, so changing the judge
  reruns only judging, and retrieval sweeps need no LLM.
- **Run records.** Every run stores its config, git commit, corpus and dataset hashes, model
  revisions and hardware.
- **Statistics.** Bootstrap intervals throughout, paired bootstrap for retrieval comparisons and
  McNemar's exact test for answer correctness.
- **Oracle runs** give the generator the gold passages, separating retrieval failures from
  generation failures.

### The dataset

195 questions across single-hop, multi-hop, config/code, table, version-specific, paraphrased and
unanswerable types, split into development (38), calibration (41) and test (116).

1. An LLM drafted questions with evidence quotes from the corpus.
2. A human verified each one: accepted, edited or rejected it, and confirmed the evidence span.
3. **Pooled review for missed evidence.** Gold labels are rarely complete, so the top results of
   seven diverse configurations were pooled and judged by a human (TREC-style; depth 3 on
   development, depth 2 on test, recorded in [pooling.json](data/datasets/v1/pooling.json)). This
   added evidence that would otherwise have counted against good retrievers.
4. **Reference audit.** Questions that every oracle run failed were rechecked by a human, and fixed
   or withdrawn.

Every decision is kept in [review_decisions.jsonl](data/datasets/v1/review_decisions.jsonl). The
test split was not looked at while tuning.

## Quick start

Needs [uv](https://docs.astral.sh/uv/). No GPU or models are needed for the tests or the
regression gate.

```bash
git clone https://github.com/ayush-pl/adv-rag.git
cd adv-rag
uv sync
uv run pytest
uv run python scripts/regression_gate.py check
```

## Reproducing the benchmark

Needs Python 3.12 (uv installs it), git, an NVIDIA GPU for reasonable speed and
[Ollama](https://ollama.com/) for answers and judging. Disk: about 6 GB for the PyTorch
environment and 650 MB for the pipeline data. Developed on Linux (WSL2) with an RTX 5060 8 GB.

```bash
# PyTorch with CUDA 13. Use --extra cpu instead of --extra cu130 for a CPU-only build.
uv sync --extra ml --extra cu130
uv run python scripts/check_env.py
```

### Retrieval (about 17 minutes on an RTX 5060 laptop GPU)

```bash
uv run python scripts/ingest.py                         # fetch pinned docs, normalize (1 min)
uv run python scripts/build_chunks.py                   # 4 chunking variants (0.5 min)
uv run python scripts/build_index.py                    # embed with MiniLM and BGE-small (8 min)
uv run python scripts/run_retrieval_sweep.py --split development   # 2 min
uv run python scripts/run_retrieval_sweep.py --split test          # 5.5 min
```

Each sweep rewrites `reports/retrieval_sweep_<split>.md`. Retrieval is deterministic: following
these steps from a fresh clone reproduced every metric in both committed reports exactly, with the
same corpus hash. Only latencies differ.

### Answers and judging (several hours)

```bash
ollama pull llama3.1:8b && ollama pull gemma3:4b && ollama pull qwen2.5:7b
uv run python scripts/retrieve_contexts.py              # save top-5 passages per question
uv run python scripts/generate_answers.py               # each model, retrieved and oracle
uv run python scripts/judge_answers.py
uv run python scripts/report_generation.py --split test
uv run python scripts/calibrate_judge.py report
```

Each script saves as it goes and resumes where it stopped. Generation runs at temperature 0 with a
fixed seed, but LLM output is not guaranteed to be identical across Ollama versions or hardware.
Generators and judge are set in [generation.yaml](configs/generation.yaml).

Rebuilding the dataset itself needs a human: see `draft_questions.py`, `review_questions.py`,
`review_missed_evidence.py`, `audit_references.py` and `finalize_dataset.py` in
[scripts](scripts/).

## API

A FastAPI service answers questions with citations, runs evaluations as background jobs and serves
experiments and reports. It needs the retrieval pipeline outputs in `data/` and a running Ollama.

```bash
uv run python -m ragbench.api --port 8000               # docs at http://localhost:8000/docs
```

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "What is the default restart policy for a Kubernetes Pod?"}'
```

Returns the answer, its citations (document, title, source URL, character span), whether the model
abstained, and per-stage latency. Other endpoints: `/health`, `/evaluate` with `/jobs/{id}`,
`/experiments`, `/experiments/compare` and `/reports`. Errors are structured and every response
carries an `X-Execution-ID`. Set `RAGBENCH_API_KEY` to require an `X-API-Key` header.

### Docker

```bash
docker compose up --build
```

The image runs CPU PyTorch, mounts `./data` read-only and reaches Ollama on the host at
`host.docker.internal`. For Docker Engine inside WSL with Ollama on Windows, turn on mirrored
networking (`networkingMode=mirrored` in `.wslconfig`) and add the host-network override:

```bash
docker compose -f docker-compose.yml -f docker-compose.wsl.yml up --build
```

Latency from the container is CPU-bound; published numbers come from native GPU runs.

## Regression gate

CI runs lint, unit tests, the regression gate and a Docker build on every push and pull request.

The gate reruns the real retrieval and evaluation code on a small committed fixture: the
development and calibration questions, the chunks real retrievers return for them, and stored
embedding vectors and recorded answers. So it needs no GPU, models or corpus. For each
configuration it compares Recall@5, MRR and nDCG@10 per question with a committed baseline:

- **pass**: the mean dropped by no more than 0.01;
- **fail**: it dropped more, and the 95% paired bootstrap interval lies below zero;
- **inconclusive**: it dropped more, but not consistently. CI warns and a human decides.

Recomputed evaluation values (correctness, abstention, citation scores) must match exactly, so a
change to a metric's definition needs a deliberate baseline update. Reversing the RRF ordering makes
the gate fail; making BM25 case-sensitive makes it inconclusive.

```bash
uv run python scripts/regression_gate.py check       # what CI runs
uv run python scripts/regression_gate.py baseline    # accept a deliberate change
uv run python scripts/regression_gate.py build       # rebuild the fixture (needs the pipeline)
```

## Project layout

```
configs/        chunking, embedders, retrieval sweep and generation settings
data/manifests/ pinned corpus sources and content hashes
data/datasets/  questions, splits, review history and human scores
src/ragbench/   ingestion, chunking, embeddings, retrieval, generation, evaluation,
                experiments, regression and api packages
scripts/        one script per pipeline stage
reports/        generated reports
tests/          unit tests and the regression fixture
```

Generated data (`data/raw`, `processed`, `chunks`, `indexes`, `runs`) is not committed.

## Limitations

- **Small test set.** 101 answerable test questions give intervals of about ±10 points on answer
  correctness, so per-type results are exploratory.
- **One human annotator** verified questions and scored answers, so inter-annotator agreement is
  not measured.
- **LLM-drafted questions** may echo the source wording, which favours keyword retrieval. Paraphrase
  questions are a partial check.
- **Pooled labels are still incomplete.** Passages no pooled configuration found were never judged.
- **Judge faithfulness is unreliable**, as above. Correctness and relevance were validated on 24
  held-out answers only.
- **Small models.** Generators are 4B and 8B models that fit in 8 GB of VRAM. Conclusions about the
  bottleneck may differ for larger ones.

## License

Code is under the [MIT License](LICENSE). The documentation excerpts in `tests/regression/fixture`
keep their own licenses (Kubernetes docs CC BY 4.0, Docker docs Apache 2.0, PostgreSQL docs the
PostgreSQL License), listed in its `manifest.json`.
