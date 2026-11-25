import numpy as np
import pytest

from ragbench.chunking.models import Chunk
from ragbench.embeddings.base import Embedder, EmbedderConfig, TruncationStats
from ragbench.embeddings.index_store import build_vectors, index_directory, load_vectors
from ragbench.retrieval.base import MetadataFilter, Reranker, Retriever, ScoredChunk
from ragbench.retrieval.bm25 import BM25Retriever, tokenize
from ragbench.retrieval.dense import DenseRetriever
from ragbench.retrieval.fusion import reciprocal_rank_fusion
from ragbench.retrieval.hybrid import HybridRetriever, RerankingRetriever


def make_chunk(chunk_id: str, text: str = "text", source="A", version="1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        char_start=0,
        char_end=len(text),
        text=text,
        token_count=1,
        section="",
        source=source,
        source_url="https://example.org",
        title="Title",
        version=version,
    )


def ranking(*chunk_ids: str) -> list[ScoredChunk]:
    return [ScoredChunk(make_chunk(i), 1.0 / (n + 1)) for n, i in enumerate(chunk_ids)]


def ids(results: list[ScoredChunk]) -> list[str]:
    return [result.chunk.chunk_id for result in results]


class FakeEmbedder(Embedder):
    """Bag-of-words over a fixed vocabulary, which makes dense results predictable."""

    VOCABULARY = ["pod", "docker", "restart", "table"]

    @property
    def dimension(self) -> int:
        return len(self.VOCABULARY)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(
            [[float(word in text.lower()) for word in self.VOCABULARY] for text in texts],
            dtype=np.float32,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def truncation_stats(self, texts: list[str]) -> TruncationStats:
        return TruncationStats(
            max_tokens=10,
            texts=len(texts),
            truncated_texts=0,
            truncated_rate=0.0,
            longest_text_tokens=1,
        )


CORPUS = [
    make_chunk("pod-v1", "pod restart policy", source="K8s", version="1"),
    make_chunk("pod-v2", "pod restart policy", source="K8s", version="2"),
    make_chunk("docker", "docker restart flag", source="Docker", version="1"),
    make_chunk("table", "table partitioning guide", source="PG", version="1"),
]


# BM25Okapi's IDF is exactly 0 for a term in half of the documents, so its tests and any small
# benchmark fixture need terms that are rarer than that.
BM25_CORPUS = [*CORPUS, make_chunk("x", "unrelated words"), make_chunk("y", "other content")]


def dense_retriever() -> DenseRetriever:
    embedder = FakeEmbedder()
    return DenseRetriever(CORPUS, embedder.embed_documents([c.text for c in CORPUS]), embedder)


class TestDense:
    def test_returns_most_similar_first_with_ties_broken_by_id(self):
        results = dense_retriever().retrieve("pod restart", 3)
        assert ids(results) == ["pod-v1", "pod-v2", "docker"]
        assert results[0].score == pytest.approx(1.0)

    def test_top_k_above_corpus_size_returns_all(self):
        assert len(dense_retriever().retrieve("pod", 100)) == len(CORPUS)

    def test_metadata_filter_limits_candidates(self):
        version_two = MetadataFilter(versions=frozenset({"2"}))
        assert ids(dense_retriever().retrieve("pod", 5, version_two)) == ["pod-v2"]
        docker_only = MetadataFilter(sources=frozenset({"Docker"}))
        assert ids(dense_retriever().retrieve("restart", 5, docker_only)) == ["docker"]

    def test_filter_matching_nothing_returns_empty(self):
        assert dense_retriever().retrieve("pod", 5, MetadataFilter(versions=frozenset())) == []

    def test_mismatched_vectors_are_rejected(self):
        with pytest.raises(ValueError, match="4 chunks but 1 vectors"):
            DenseRetriever(CORPUS, np.zeros((1, 4), dtype=np.float32), FakeEmbedder())


class TestBM25:
    def test_tokenize_lowercases_and_keeps_identifiers(self):
        assert tokenize("Set restart_policy: Always!") == ["set", "restart_policy", "always"]

    def test_keyword_match_ranks_first_and_nonmatches_are_excluded(self):
        results = BM25Retriever(BM25_CORPUS).retrieve("partitioning", 5)
        assert ids(results) == ["table"]

    def test_tied_scores_are_broken_by_id(self):
        assert ids(BM25Retriever(BM25_CORPUS).retrieve("policy", 2)) == ["pod-v1", "pod-v2"]

    def test_tie_at_cutoff_is_deterministic(self):
        assert ids(BM25Retriever(BM25_CORPUS).retrieve("policy", 1)) == ["pod-v1"]

    def test_metadata_filter_limits_candidates(self):
        version_two = MetadataFilter(versions=frozenset({"2"}))
        assert ids(BM25Retriever(BM25_CORPUS).retrieve("policy", 5, version_two)) == ["pod-v2"]

    def test_empty_corpus_is_rejected(self):
        with pytest.raises(ValueError, match="at least one chunk"):
            BM25Retriever([])


class TestReciprocalRankFusion:
    def test_scores_match_hand_computed_values(self):
        # a: 1/(60+1) + 1/(60+2), b: 1/(60+2) + 1/(60+1), c: 1/(60+3)
        fused = reciprocal_rank_fusion([ranking("a", "b", "c"), ranking("b", "a")], top_k=3)
        assert ids(fused) == ["a", "b", "c"]
        assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
        assert fused[1].score == pytest.approx(1 / 61 + 1 / 62)
        assert fused[2].score == pytest.approx(1 / 63)

    def test_agreement_across_lists_beats_top_rank_in_one(self):
        # b: 1/62 + 1/62 = 0.03226 beats a: 1/61 = 0.01639
        fused = reciprocal_rank_fusion([ranking("a", "b"), ranking("c", "b")], top_k=1)
        assert ids(fused) == ["b"]

    def test_k_changes_the_outcome(self):
        rankings = [ranking("a", "x", "y", "z"), ranking("b", "x"), ranking("c", "y", "x")]
        # x at ranks 2, 2, 3. Small k favours the rank-1 items only if their sum is larger.
        low = reciprocal_rank_fusion(rankings, top_k=1, rrf_k=1)
        high = reciprocal_rank_fusion(rankings, top_k=1, rrf_k=1000)
        assert low[0].score == pytest.approx(1 / 3 + 1 / 3 + 1 / 4)
        assert high[0].score == pytest.approx(1 / 1002 + 1 / 1002 + 1 / 1003)
        assert ids(low) == ids(high) == ["x"]

    def test_uses_rank_not_original_score(self):
        scored_high = [ScoredChunk(make_chunk("a"), 9999.0)]
        fused = reciprocal_rank_fusion([scored_high], top_k=1)
        assert fused[0].score == pytest.approx(1 / 61)

    def test_top_k_limits_output_and_empty_input_is_safe(self):
        assert len(reciprocal_rank_fusion([ranking("a", "b", "c")], top_k=2)) == 2
        assert reciprocal_rank_fusion([], top_k=2) == []
        assert reciprocal_rank_fusion([[]], top_k=2) == []

    def test_invalid_k_is_rejected(self):
        with pytest.raises(ValueError, match="rrf_k"):
            reciprocal_rank_fusion([ranking("a")], top_k=1, rrf_k=0)


class FixedRetriever(Retriever):
    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.calls: list[tuple[int, MetadataFilter | None]] = []

    def retrieve(self, query, top_k, metadata_filter=None):
        self.calls.append((top_k, metadata_filter))
        return self.results[:top_k]


class ReverseReranker(Reranker):
    """Scores candidates by reversed input position."""

    def rerank(self, query, candidates, top_n):
        count = len(candidates)
        rescored = [
            ScoredChunk(c.chunk, float(count - i)) for i, c in enumerate(reversed(candidates))
        ]
        return rescored[:top_n]


class TestComposites:
    def test_hybrid_fuses_and_passes_filter_and_depth(self):
        first = FixedRetriever(ranking("a", "b"))
        second = FixedRetriever(ranking("b", "c"))
        chosen = MetadataFilter(versions=frozenset({"1"}))
        results = HybridRetriever([first, second], depth=50).retrieve("q", 2, chosen)
        assert ids(results)[0] == "b"
        assert first.calls == [(50, chosen)]
        assert second.calls == [(50, chosen)]

    def test_hybrid_depth_is_at_least_top_k(self):
        first = FixedRetriever(ranking("a"))
        HybridRetriever([first], depth=5).retrieve("q", 10)
        assert first.calls[0][0] == 10

    def test_real_dense_and_bm25_hybrid(self):
        results = HybridRetriever([dense_retriever(), BM25Retriever(CORPUS)]).retrieve("docker", 1)
        assert ids(results) == ["docker"]

    def test_reranking_fetches_candidates_then_returns_top_n(self):
        base = FixedRetriever(ranking("a", "b", "c", "d"))
        results = RerankingRetriever(base, ReverseReranker(), candidates=20).retrieve("q", 2)
        assert base.calls[0][0] == 20
        assert ids(results) == ["d", "c"]


class TestVectorStore:
    def config(self) -> EmbedderConfig:
        return EmbedderConfig(model="org/model", revision="abcdef1234")

    def test_slug_is_safe_for_paths_and_names_revision(self):
        assert self.config().slug == "org--model-abcdef12"

    def test_build_records_measurements_and_preserves_order(self, tmp_path):
        chunks_dir = tmp_path / "chunks" / "corpus" / "config"
        stats = build_vectors(CORPUS, FakeEmbedder(), self.config(), chunks_dir, tmp_path / "ix")

        directory = index_directory(tmp_path / "ix", chunks_dir, self.config())
        vectors = load_vectors(directory)
        assert vectors.shape == (4, 4)
        assert vectors.dtype == np.float32
        assert stats.vector_bytes == vectors.nbytes
        assert stats.truncation.max_tokens == 10
        np.testing.assert_allclose(vectors[2], FakeEmbedder().embed_documents([CORPUS[2].text])[0])

    def test_finished_build_is_reused(self, tmp_path):
        chunks_dir = tmp_path / "chunks" / "corpus" / "config"
        first = build_vectors(CORPUS, FakeEmbedder(), self.config(), chunks_dir, tmp_path / "ix")
        second = build_vectors([], FakeEmbedder(), self.config(), chunks_dir, tmp_path / "ix")
        assert first == second
