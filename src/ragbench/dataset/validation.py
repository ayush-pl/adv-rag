"""Dataset integrity checks. A benchmark built on a broken label is silently wrong."""

import re
from collections import Counter

from ragbench.dataset.models import Question
from ragbench.ingestion.models import Document


def validate_questions(questions: list[Question], documents: dict[str, Document]) -> list[str]:
    """Return every problem found. An empty list means the questions are trustworthy."""
    problems: list[str] = []

    for question_id, count in Counter(q.question_id for q in questions).items():
        if count > 1:
            problems.append(f"{question_id}: duplicate question ID")

    first_with_text: dict[str, str] = {}
    known_ids = {q.question_id for q in questions}
    for question in questions:
        qid = question.question_id
        text_key = _normalize(question.question)
        if text_key in first_with_text:
            problems.append(f"{qid}: same text as {first_with_text[text_key]}")
        first_with_text.setdefault(text_key, qid)

        if not question.human_verified:
            problems.append(f"{qid}: not human verified")
        source = question.provenance.source_question_id
        if source is not None and source not in known_ids:
            problems.append(f"{qid}: source question {source} is missing")

        for group in question.evidence:
            for span in group.spans:
                problems.extend(_check_span(qid, span, documents))
    return problems


def validate_splits(splits: dict[str, list[Question]]) -> list[str]:
    """Check for leakage: no question, or paraphrase of one, may span two splits."""
    problems: list[str] = []
    split_of: dict[str, str] = {}
    for name, questions in splits.items():
        for question in questions:
            if question.question_id in split_of:
                problems.append(f"{question.question_id}: in both {split_of[question.question_id]}")
            split_of[question.question_id] = name

    for name, questions in splits.items():
        for question in questions:
            source = question.provenance.source_question_id
            if source is not None and split_of.get(source) != name:
                problems.append(
                    f"{question.question_id}: in {name} but its source {source} "
                    f"is in {split_of.get(source)}"
                )
    return problems


def _check_span(qid: str, span, documents: dict[str, Document]) -> list[str]:
    document = documents.get(span.document_id)
    if document is None:
        return [f"{qid}: unknown document {span.document_id}"]
    if span.char_end > len(document.content):
        return [f"{qid}: span {span.char_start}-{span.char_end} is past the end of the document"]
    if document.content[span.char_start : span.char_end] != span.quote:
        return [f"{qid}: quote no longer matches {span.document_id} at {span.char_start}"]
    return []


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()
