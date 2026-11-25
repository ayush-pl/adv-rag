"""Gold evidence defined on document text, so it applies to every chunking configuration.

Chunk IDs change whenever chunking changes, so labels are character spans. A retrieved chunk
is matched to the spans it overlaps at evaluation time.
"""

from pydantic import BaseModel, Field, model_validator

from ragbench.chunking.models import Chunk

# A chunk counts as matching a span when the overlap covers at least this share of the shorter
# of the two. Using the shorter handles both spans larger than a chunk and the reverse.
DEFAULT_MIN_OVERLAP = 0.5


class GoldSpan(BaseModel):
    document_id: str
    char_start: int = Field(ge=0)
    char_end: int
    # The labelled text, which lets validation detect spans that no longer match the corpus.
    quote: str

    @model_validator(mode="after")
    def span_is_not_empty(self) -> "GoldSpan":
        if self.char_end <= self.char_start:
            raise ValueError(f"Empty span {self.char_start}-{self.char_end}")
        return self


class EvidenceGroup(BaseModel):
    """One required piece of evidence. Any span in the group satisfies it.

    Alternatives cover the same fact stated in more than one place, such as a page shared by
    PostgreSQL 15 and 16. A multi-hop question has one group per hop.
    """

    spans: list[GoldSpan] = Field(min_length=1)
    # 2 means the evidence directly answers the question, 1 means it supports the answer.
    grade: int = Field(default=2, ge=1, le=2)


def chunk_matches_span(
    chunk: Chunk, span: GoldSpan, min_overlap: float = DEFAULT_MIN_OVERLAP
) -> bool:
    if chunk.document_id != span.document_id:
        return False
    overlap = min(chunk.char_end, span.char_end) - max(chunk.char_start, span.char_start)
    if overlap <= 0:
        return False
    shorter = min(chunk.char_end - chunk.char_start, span.char_end - span.char_start)
    return overlap / shorter >= min_overlap


def match_ranking(
    ranked: list[Chunk], groups: list[EvidenceGroup], min_overlap: float = DEFAULT_MIN_OVERLAP
) -> list[set[int]]:
    """For each ranked chunk, return the indices of the evidence groups it satisfies."""
    return [
        {
            index
            for index, group in enumerate(groups)
            if any(chunk_matches_span(chunk, span, min_overlap) for span in group.spans)
        }
        for chunk in ranked
    ]
