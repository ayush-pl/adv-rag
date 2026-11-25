import pytest
from pydantic import ValidationError

from ragbench.chunking.chunk_manager import build_chunks, chunks_directory, load_chunks
from ragbench.chunking.fixed import FixedChunker
from ragbench.chunking.models import ChunkingConfig
from ragbench.chunking.recursive import RecursiveChunker
from ragbench.chunking.tokenizer import RegexTokenCounter, TokenOffsets
from ragbench.ingestion.cleaner import content_hash, extract_sections
from ragbench.ingestion.models import Document

COUNTER = RegexTokenCounter()


def make_document(content: str, document_id: str = "src/page") -> Document:
    return Document(
        document_id=document_id,
        source="Source",
        source_url="https://example.org/page",
        title="Page",
        version="1",
        content=content,
        content_hash=content_hash(content),
        sections=extract_sections(content),
        ingestion_date="2026-11-01",
    )


def words(count: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


def make_config(**overrides) -> ChunkingConfig:
    fields = {
        "strategy": "recursive",
        "chunk_size": 20,
        "overlap": 0,
        "tokenizer": "test",
        "tokenizer_revision": "r1",
    }
    return ChunkingConfig(**{**fields, **overrides})


def test_token_offsets_count_only_tokens_wholly_inside_span():
    offsets = TokenOffsets(starts=[0, 4, 8], ends=[3, 7, 11])
    assert offsets.count(0, 11) == 3
    assert offsets.count(4, 7) == 1
    assert offsets.count(5, 11) == 1
    assert offsets.count(3, 4) == 0


@pytest.mark.parametrize("chunker_class", [FixedChunker, RecursiveChunker])
class TestBothChunkers:
    def test_text_matches_offsets_and_size_limit_holds(self, chunker_class):
        doc = make_document(f"# Title\n\n{words(100)}\n\n## Part\n\n{words(50, 'x')}\n")
        chunks = chunker_class(20, 5, COUNTER).chunk(doc)
        assert chunks
        for chunk in chunks:
            assert chunk.text == doc.content[chunk.char_start : chunk.char_end]
            assert 0 < chunk.token_count <= 20

    def test_ids_are_unique_and_stable(self, chunker_class):
        doc = make_document(f"# Title\n\n{words(100)}\n")
        first = [c.chunk_id for c in chunker_class(20, 5, COUNTER).chunk(doc)]
        second = [c.chunk_id for c in chunker_class(20, 5, COUNTER).chunk(doc)]
        assert first == second
        assert len(set(first)) == len(first)
        assert first[0].startswith("src/page::")

    def test_metadata_is_preserved(self, chunker_class):
        chunk = chunker_class(20, 0, COUNTER).chunk(make_document("# Title\n\nSome words.\n"))[0]
        assert (chunk.document_id, chunk.source, chunk.version) == ("src/page", "Source", "1")
        assert chunk.source_url == "https://example.org/page"

    def test_empty_and_markup_only_documents_give_no_chunks(self, chunker_class):
        for content in ("", "\n\n", "---\n"):
            assert chunker_class(20, 0, COUNTER).chunk(make_document(content)) == []

    def test_overlap_must_be_below_size(self, chunker_class):
        with pytest.raises(ValueError, match="overlap"):
            chunker_class(10, 10, COUNTER)


class TestFixedChunker:
    def test_windows_have_exact_size_and_overlap(self):
        doc = make_document(words(24))
        chunks = FixedChunker(10, 3, COUNTER).chunk(doc)
        assert [c.text.split()[0] for c in chunks] == ["w0", "w7", "w14"]
        assert [c.token_count for c in chunks] == [10, 10, 10]

    def test_no_tiny_trailing_chunk_when_last_window_reaches_end(self):
        chunks = FixedChunker(10, 3, COUNTER).chunk(make_document(words(10)))
        assert len(chunks) == 1


class TestRecursiveChunker:
    def chunk(self, content: str, size: int = 20, overlap: int = 0):
        return RecursiveChunker(size, overlap, COUNTER).chunk(make_document(content))

    def test_new_section_starts_new_chunk_with_its_heading(self):
        chunks = self.chunk("# A\n\none two\n\n## B\n\nthree four\n")
        assert [c.text for c in chunks] == ["# A\n\none two", "## B\n\nthree four"]

    def test_consecutive_headings_stay_with_following_text(self):
        chunks = self.chunk("# A\n\n## B\n\ntext\n")
        assert [c.text for c in chunks] == ["# A\n\n## B\n\ntext"]

    def test_section_path_reflects_heading_levels(self):
        chunks = self.chunk("# A\n\none\n\n## B\n\ntwo\n\n## C\n\nthree\n\n# D\n\nfour\n")
        assert [c.section for c in chunks] == ["A", "A > B", "A > C", "D"]

    def test_code_block_is_kept_whole_and_blank_lines_do_not_split_it(self):
        code = "```\nline one\n\nline two\n```"
        chunks = self.chunk(f"{words(15)}\n\n{code}\n\nafter\n")
        assert sum(code in c.text for c in chunks) == 1

    def test_table_is_kept_whole(self):
        table = "| a | b |\n| c | d |"
        chunks = self.chunk(f"{words(15)}\n\n{table}\n")
        assert chunks[-1].text == table

    def test_table_directly_after_text_is_a_separate_block(self):
        table = "| a | b |\n| c | d |"
        chunks = self.chunk(f"{words(15)}\n{table}\n")
        assert chunks[-1].text == table

    def test_oversized_code_block_splits_on_lines(self):
        lines = [f"line {i}" for i in range(30)]
        chunks = self.chunk("```\n" + "\n".join(lines) + "\n```\n", size=12)
        for chunk in chunks:
            assert chunk.token_count <= 12
            for line in chunk.text.splitlines():
                assert line == "```" or line in lines

    def test_oversized_paragraph_splits_on_sentences(self):
        sentences = [f"Sentence number {i} is here." for i in range(10)]
        chunks = self.chunk(" ".join(sentences) + "\n", size=15)
        for chunk in chunks:
            assert chunk.token_count <= 15
            assert chunk.text.endswith(".")

    def test_single_oversized_sentence_falls_back_to_token_windows(self):
        chunks = self.chunk(words(50) + "\n", size=20)
        assert [c.token_count for c in chunks] == [20, 20, 10]

    def test_overlap_repeats_trailing_units_without_exceeding_size(self):
        paragraphs = [f"para{i} a b c" for i in range(6)]
        chunks = self.chunk("\n\n".join(paragraphs) + "\n", size=12, overlap=4)
        assert len(chunks) > 1
        for previous, following in zip(chunks, chunks[1:], strict=False):
            assert following.char_start < previous.char_end
            assert following.token_count <= 12

    def test_no_overlap_across_sections(self):
        chunks = self.chunk("# A\n\none two\n\n# B\n\nthree\n", overlap=5)
        assert chunks[1].char_start >= chunks[0].char_end


class TestChunkManager:
    def test_config_rejects_overlap_at_or_above_size(self):
        with pytest.raises(ValidationError):
            make_config(chunk_size=10, overlap=10)

    def test_config_id_changes_with_any_field(self):
        ids = {
            make_config().config_id,
            make_config(overlap=1).config_id,
            make_config(tokenizer_revision="r2").config_id,
        }
        assert len(ids) == 3

    def test_build_writes_chunks_and_statistics(self, tmp_path):
        doc = make_document(f"# Title\n\n{words(50)}\n")
        config = make_config()
        stats = build_chunks([doc], config, "sha256-abc", tmp_path, COUNTER)

        loaded = load_chunks(chunks_directory(tmp_path, "sha256-abc", config))
        assert stats.chunks == len(loaded) > 1
        assert stats.tokens_max <= 20
        assert stats.documents == 1
        assert stats.config == config

    def test_finished_build_is_reused(self, tmp_path):
        doc = make_document(f"# Title\n\n{words(50)}\n")
        config = make_config()
        first = build_chunks([doc], config, "sha256-abc", tmp_path, COUNTER)
        second = build_chunks([], config, "sha256-abc", tmp_path, COUNTER)
        assert first == second

    def test_interrupted_build_is_rebuilt(self, tmp_path):
        doc = make_document(f"# Title\n\n{words(50)}\n")
        config = make_config()
        directory = chunks_directory(tmp_path, "sha256-abc", config)
        directory.mkdir(parents=True)
        (directory / "chunks.jsonl").write_text("partial", encoding="utf-8")
        stats = build_chunks([doc], config, "sha256-abc", tmp_path, COUNTER)
        assert len(load_chunks(directory)) == stats.chunks

    def test_no_chunks_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="no chunks"):
            build_chunks([], make_config(), "sha256-abc", tmp_path, COUNTER)
