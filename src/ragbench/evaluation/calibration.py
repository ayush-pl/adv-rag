"""Human scoring of a sample of answers, to measure how far the LLM judge can be trusted.

The reviewer scores blind: the sheet does not say which system wrote an answer, and the
judge's scores are not shown. A separate key file links each sample to its source.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from pydantic import BaseModel

from ragbench.dataset.models import Question
from ragbench.evaluation.judge import DIMENSIONS
from ragbench.generation.prompts import Passage
from ragbench.generation.rag_pipeline import GeneratedAnswer

SHEET = "Score answers"
RUBRIC = "Rubric"


class CalibrationItem(BaseModel):
    sample_id: str
    # The generation file and question the answer came from. Kept out of the sheet.
    run: str
    question_id: str


class HumanScore(BaseModel):
    sample_id: str
    scores: dict[str, int]
    note: str = ""


def judgeable(question: Question, answer: GeneratedAnswer) -> bool:
    """Only answers the judge scores can be compared with human scores."""
    return question.answerable and not answer.abstained and answer.parsed


def sample_items(
    answers_by_run: dict[str, list[GeneratedAnswer]],
    questions: dict[str, Question],
    size: int,
    seed: int = 0,
    per_question: int = 2,
    prefix: str = "s",
) -> list[CalibrationItem]:
    """Spread the sample evenly over runs, and over questions, so it covers a range of quality."""
    rng = random.Random(seed)
    pools = {
        run: rng.sample(eligible, len(eligible))
        for run, answers in sorted(answers_by_run.items())
        if (eligible := [a for a in answers if judgeable(questions[a.question_id], a)])
    }
    chosen: list[tuple[str, str]] = []
    used: dict[str, int] = {}
    while len(chosen) < size and any(pools.values()):
        for run in sorted(pools):
            while pools[run] and len(chosen) < size:
                answer = pools[run].pop()
                if used.get(answer.question_id, 0) < per_question:
                    used[answer.question_id] = used.get(answer.question_id, 0) + 1
                    chosen.append((run, answer.question_id))
                    break
    rng.shuffle(chosen)
    return [
        CalibrationItem(sample_id=f"{prefix}-{number:03}", run=run, question_id=question_id)
        for number, (run, question_id) in enumerate(chosen, start=1)
    ]


HEADERS = [
    ("Sample", 8),
    ("Question", 35),
    ("Reference answer", 35),
    ("Answer to score", 45),
    ("Passages the system was given", 90),
    ("Correctness (1-5)", 11),
    ("Faithfulness (1-5)", 11),
    ("Relevance (1-5)", 11),
    ("Note (optional)", 30),
]
_SCORE_COLUMNS = {name: 6 + i for i, name in enumerate(DIMENSIONS)}
_NOTE_COLUMN = 9
_EDITABLE_FILL = PatternFill("solid", start_color="FFF4CC")
_HEADER_FILL = PatternFill("solid", start_color="1F3A5F")

RUBRIC_TEXT = [
    "Score each answer on three things, from 1 to 5. Judge the answer only, not the passages.",
    "",
    "Correctness: does the answer agree with the reference answer?",
    "  5 fully correct and complete, 4 correct but missing a minor detail,",
    "  3 partly correct or with a notable error, 2 mostly wrong, 1 wrong.",
    "  If the reference answer itself is wrong, score against what the passages say,",
    "  and write 'reference wrong' in the note.",
    "",
    "Faithfulness: is every claim in the answer supported by the passages shown?",
    "  5 every claim supported, 4 one minor unsupported detail, 3 some claims unsupported,",
    "  2 mostly unsupported, 1 contradicts or ignores the passages.",
    "  A claim can be true and still unfaithful, if the passages don't say it.",
    "",
    "Relevance: does the answer address the question asked?",
    "  5 directly and fully, 3 partly or with much irrelevant material, 1 not at all.",
    "",
    "Don't reward length: a short correct answer can earn 5.",
    "Score every row before looking at anything else, so your scores stay independent.",
]


def export_sheet(
    items: list[CalibrationItem],
    answers: dict[tuple[str, str], GeneratedAnswer],
    passages: dict[tuple[str, str], list[Passage]],
    questions: dict[str, Question],
    output: Path,
) -> None:
    workbook = Workbook()
    rubric = workbook.active
    rubric.title = RUBRIC
    for line in RUBRIC_TEXT:
        rubric.append([line])
    rubric.column_dimensions["A"].width = 100
    rubric["A1"].font = Font(bold=True)

    sheet = workbook.create_sheet(SHEET)
    sheet.append([header for header, _ in HEADERS])
    for item in items:
        key = (item.run, item.question_id)
        question = questions[item.question_id]
        shown = "\n\n".join(
            f"[{n}] {p.title}\n{p.text}" for n, p in enumerate(passages[key], start=1)
        )
        sheet.append(
            [
                item.sample_id,
                question.question,
                question.expected_answer,
                answers[key].answer,
                shown[:32_000],
                "",
                "",
                "",
                "",
            ]
        )

    last_row = len(items) + 1
    for index, (_, width) in enumerate(HEADERS, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
        header = sheet.cell(row=1, column=index)
        header.font = Font(bold=True, color="FFFFFF")
        header.fill = _HEADER_FILL
        header.alignment = Alignment(wrap_text=True, vertical="center")
        for row in range(2, last_row + 1):
            cell = sheet.cell(row=row, column=index)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if index in (*_SCORE_COLUMNS.values(), _NOTE_COLUMN):
                cell.fill = _EDITABLE_FILL
    for row in range(2, last_row + 1):
        sheet.row_dimensions[row].height = 200
    sheet.freeze_panes = "E2"
    for column in _SCORE_COLUMNS.values():
        letter = get_column_letter(column)
        validation = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
        validation.add(f"{letter}2:{letter}{last_row}")
        sheet.add_data_validation(validation)
    workbook.active = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)


@dataclass
class ScoreImport:
    scores: list[HumanScore]
    incomplete: int
    problems: list[str]


def read_sheet(path: Path, items: list[CalibrationItem]) -> ScoreImport:
    known = {item.sample_id for item in items}
    sheet = load_workbook(path, read_only=True, data_only=True)[SHEET]
    rows = sheet.iter_rows(min_row=1, values_only=True)
    if [str(v or "").strip() for v in next(rows)][: len(HEADERS)] != [h for h, _ in HEADERS]:
        raise ValueError("The sheet's columns were changed. Export a fresh sheet.")

    result = ScoreImport([], 0, [])
    for row_number, values in enumerate(rows, start=2):
        values = list(values) + [None] * (len(HEADERS) - len(values))
        sample_id = str(values[0] or "").strip()
        if not sample_id:
            continue
        if sample_id not in known:
            result.problems.append(f"Row {row_number}: unknown sample {sample_id}")
            continue
        raw = {name: values[column - 1] for name, column in _SCORE_COLUMNS.items()}
        if all(value in (None, "") for value in raw.values()):
            result.incomplete += 1
            continue
        scores = {}
        for name, value in raw.items():
            text = str(value or "").strip().removesuffix(".0")
            if text not in {"1", "2", "3", "4", "5"}:
                result.problems.append(
                    f"Row {row_number} ({sample_id}): {name} must be 1 to 5, not {value!r}"
                )
                break
            scores[name] = int(text)
        else:
            note = str(values[_NOTE_COLUMN - 1] or "").strip()
            result.scores.append(HumanScore(sample_id=sample_id, scores=scores, note=note))
    return result


def write_items(path: Path, items: list[CalibrationItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8", newline="\n"
    )


def read_items(path: Path) -> list[CalibrationItem]:
    with path.open(encoding="utf-8") as lines:
        return [CalibrationItem.model_validate(json.loads(line)) for line in lines if line.strip()]


def write_scores(path: Path, scores: list[HumanScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(score.model_dump_json() + "\n" for score in scores), encoding="utf-8", newline="\n"
    )


def read_scores(path: Path) -> list[HumanScore]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as lines:
        return [HumanScore.model_validate(json.loads(line)) for line in lines if line.strip()]
