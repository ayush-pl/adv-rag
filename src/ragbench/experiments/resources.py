"""Load chunks, vectors and models once per process, and assemble retrievers from configs."""

from pathlib import Path

import yaml

from ragbench.chunking.chunk_manager import chunks_directory, load_chunks
from ragbench.chunking.models import Chunk, ChunkingConfig
from ragbench.embeddings.base import EmbedderConfig
from ragbench.embeddings.index_store import index_directory, load_vectors
from ragbench.experiments.config import ChunkingChoice, RetrievalConfig
from ragbench.experiments.tracker import read_lock
from ragbench.retrieval.base import Retriever
from ragbench.retrieval.bm25 import BM25Retriever
from ragbench.retrieval.dedup import DeduplicatingRetriever
from ragbench.retrieval.dense import DenseRetriever
from ragbench.retrieval.hybrid import HybridRetriever, RerankingRetriever


class Resources:
    def __init__(
        self,
        data_dir: Path = Path("data"),
        chunking_path: Path = Path("configs/chunking.yaml"),
        embeddings_path: Path = Path("configs/embeddings.yaml"),
        corpus_version: str = "v1",
    ) -> None:
        self.data_dir = data_dir
        chunking = yaml.safe_load(chunking_path.read_text(encoding="utf-8"))
        embeddings = yaml.safe_load(embeddings_path.read_text(encoding="utf-8"))
        self._tokenizer = chunking["tokenizer"]
        self._embedder_configs = {
            entry["model"]: EmbedderConfig(**entry) for entry in embeddings["embedders"]
        }
        self.reranker_config = embeddings["reranker"]
        lock = read_lock(data_dir / "manifests" / f"corpus_{corpus_version}.lock.json")
        self.corpus_hash: str = lock["corpus_hash"]

        self._chunks: dict[str, list[Chunk]] = {}
        self._embedders: dict = {}
        self._dense: dict[tuple[str, str], DenseRetriever] = {}
        self._bm25: dict[str, BM25Retriever] = {}
        self._reranker = None

    def embedder_config(self, model: str) -> EmbedderConfig:
        if model not in self._embedder_configs:
            raise ValueError(f"{model} is not in configs/embeddings.yaml")
        return self._embedder_configs[model]

    def chunks_dir(self, choice: ChunkingChoice) -> Path:
        config = ChunkingConfig(**self._tokenizer, **choice.model_dump())
        directory = chunks_directory(self.data_dir / "chunks", self.corpus_hash, config)
        if not (directory / "stats.json").exists():
            raise FileNotFoundError(f"No chunks for {config.config_id}. Run build_chunks.py.")
        return directory

    def chunks(self, choice: ChunkingChoice) -> list[Chunk]:
        key = choice.model_dump_json()
        if key not in self._chunks:
            self._chunks[key] = load_chunks(self.chunks_dir(choice))
        return self._chunks[key]

    def retriever(self, config: RetrievalConfig) -> Retriever:
        if config.method == "dense":
            retriever: Retriever = self._dense_retriever(config.chunking, config.embedder)
        elif config.method == "bm25":
            retriever = self._bm25_retriever(config.chunking)
        else:
            retriever = HybridRetriever(
                [
                    self._dense_retriever(config.chunking, config.embedder),
                    self._bm25_retriever(config.chunking),
                ],
                rrf_k=config.rrf_k,
                depth=config.hybrid_depth,
            )
        if config.dedup:
            retriever = DeduplicatingRetriever(retriever)
        if config.rerank:
            retriever = RerankingRetriever(
                retriever, self._cross_encoder(), config.rerank_candidates
            )
        return retriever

    def _dense_retriever(self, choice: ChunkingChoice, model: str) -> DenseRetriever:
        key = (choice.model_dump_json(), model)
        if key not in self._dense:
            config = self.embedder_config(model)
            directory = index_directory(self.data_dir / "indexes", self.chunks_dir(choice), config)
            if not (directory / "stats.json").exists():
                raise FileNotFoundError(f"No vectors for {model}. Run build_index.py.")
            self._dense[key] = DenseRetriever(
                self.chunks(choice), load_vectors(directory), self._embedder(model)
            )
        return self._dense[key]

    def _bm25_retriever(self, choice: ChunkingChoice) -> BM25Retriever:
        key = choice.model_dump_json()
        if key not in self._bm25:
            self._bm25[key] = BM25Retriever(self.chunks(choice))
        return self._bm25[key]

    def _embedder(self, model: str):
        if model not in self._embedders:
            from ragbench.embeddings.sentence_transformer import SentenceTransformerEmbedder

            self._embedders[model] = SentenceTransformerEmbedder(self.embedder_config(model))
        return self._embedders[model]

    def _cross_encoder(self):
        if self._reranker is None:
            from ragbench.retrieval.reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(
                self.reranker_config["model"], self.reranker_config["revision"]
            )
        return self._reranker
