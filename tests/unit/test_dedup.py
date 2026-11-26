import pytest

from ragbench.chunking.models import Chunk
from ragbench.experiments.config import ChunkingChoice, RetrievalConfig
from ragbench.retrieval.base import Retriever, ScoredChunk
from ragbench.retrieval.dedup import DeduplicatingRetriever, duplicate_key, named_versions


def chunk(chunk_id: str, text: str, version: str = "16") -> Chunk:
    document_id = chunk_id.split("::")[0]
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        char_start=0,
        char_end=len(text),
        text=text,
        token_count=1,
        section="",
        source="S",
        source_url="https://example.org",
        title="T",
        version=version,
    )


class Fixed(Retriever):
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.asked: list[int] = []

    def retrieve(self, query, top_k, metadata_filter=None):
        self.asked.append(top_k)
        return [ScoredChunk(c, 1.0 / (i + 1)) for i, c in enumerate(self.chunks[:top_k])]


PG15_A = chunk("pg15/a::0-1", "Alpha  text.", "15")
PG16_A = chunk("pg16/a::0-1", "alpha text.", "16")
PG15_B = chunk("pg15/b::0-1", "Beta text.", "15")
PG16_B = chunk("pg16/b::0-1", "Beta text.", "16")
OTHER = chunk("k8s/c::0-1", "Gamma text.", "v1.37")


def ids(results: list[ScoredChunk]) -> list[str]:
    return [r.chunk.chunk_id for r in results]


def test_keeps_the_first_copy_and_fills_the_slot_with_new_passages():
    base = Fixed([PG15_A, PG16_A, PG15_B, PG16_B, OTHER])
    results = DeduplicatingRetriever(base).retrieve("What does alpha do?", 3)
    assert ids(results) == ["pg15/a::0-1", "pg15/b::0-1", "k8s/c::0-1"]
    # The base is asked for more, so removing duplicates still fills top_k.
    assert base.asked == [6]


def test_prefers_the_copy_from_the_version_the_question_names():
    base = Fixed([PG15_A, PG16_A, OTHER])
    results = DeduplicatingRetriever(base).retrieve("In PostgreSQL 16, what does alpha do?", 2)
    assert ids(results) == ["pg16/a::0-1", "k8s/c::0-1"]
    # The surviving copy keeps the first copy's rank and score.
    assert results[0].score == 1.0


def test_version_must_be_a_whole_word():
    assert named_versions("PostgreSQL 16 roles", {"15", "16"}) == {"16"}
    assert named_versions("a 160 GB disk on 15.2", {"15", "16"}) == set()
    assert named_versions("Kubernetes v1.37 pods", {"v1.37"}) == {"v1.37"}


def test_duplicates_ignore_whitespace_and_case():
    assert duplicate_key(" Alpha\n text. ") == duplicate_key("alpha text.")


def test_overfetch_must_be_positive():
    with pytest.raises(ValueError, match="overfetch"):
        DeduplicatingRetriever(Fixed([]), overfetch=0)


def test_dedup_changes_the_config_id_only_when_on():
    chunking = ChunkingChoice(strategy="recursive", chunk_size=512, overlap=50)
    plain = RetrievalConfig(
        name="x", chunking=chunking, method="hybrid", embedder="BAAI/bge-small-en-v1.5"
    )
    # The ID of r512-hybrid-bge-k60 from before the option existed. Saved runs are found by it.
    assert plain.config_id == "83e17a5e"
    assert plain.model_copy(update={"dedup": True}).config_id != plain.config_id
