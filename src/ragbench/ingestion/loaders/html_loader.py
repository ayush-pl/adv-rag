"""Loader for DocBook-style HTML, as used by the PostgreSQL docs. Outputs Markdown-like text."""

import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from bs4.element import Comment, NavigableString, Tag

from ragbench.ingestion.loaders.base import DocumentLoader
from ragbench.ingestion.models import Document

_BOILERPLATE = "script, style, .navheader, .navfooter, .toc, a.id_link"
_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_BLOCK_TAGS = {"p", "div", "section", "blockquote", "pre", "table", "ul", "ol", "dl", "dt", "dd"}
_BLOCK_TAGS.update(_HEADINGS)
_CODE_TAGS = {"code", "tt", "kbd", "samp"}


class HtmlLoader(DocumentLoader):
    suffix = ".html"

    def load(self, path: Path) -> Document:
        # PostgreSQL pages are XHTML with an XML declaration. The HTML parser handles them fine.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
        for element in soup.select(_BOILERPLATE):
            element.decompose()
        if soup.body is None:
            raise ValueError("page has no body")

        title_tag = soup.find("title")
        title = _inline(title_tag) if title_tag else path.stem
        content = "\n\n".join(_blocks(soup.body))
        if not content.lstrip().startswith("#"):
            content = f"# {title}\n\n{content}"

        slug = path.relative_to(self.content_root).as_posix().removesuffix(".html")
        return self._build_document(slug, f"{slug}.html", title, content)


def _blocks(parent: Tag) -> list[str]:
    """Render a container's children as a list of Markdown blocks."""
    blocks: list[str] = []
    inline_parts: list[str] = []

    def flush() -> None:
        text = _collapse("".join(inline_parts))
        if text:
            blocks.append(text)
        inline_parts.clear()

    for child in parent.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            inline_parts.append(str(child))
        elif isinstance(child, Tag):
            if not _is_block(child):
                inline_parts.append(_inline_tag(child))
                continue
            flush()
            blocks.extend(_render_block(child))
    flush()
    return blocks


def _render_block(tag: Tag) -> list[str]:
    if tag.name in _HEADINGS:
        heading = _inline(tag)
        return [f"{'#' * _HEADINGS[tag.name]} {heading}"] if heading else []
    if tag.name == "pre":
        return [f"```\n{tag.get_text().strip(chr(10))}\n```"]
    if tag.name == "table":
        return _render_table(tag)
    if tag.name in {"ul", "ol"}:
        return _render_list(tag)
    return _blocks(tag)


def _render_list(tag: Tag) -> list[str]:
    items = []
    for item in tag.find_all("li", recursive=False):
        lines = "\n\n".join(_blocks(item)).splitlines()
        if lines:
            items.append("\n".join([f"- {lines[0]}"] + [f"  {line}" for line in lines[1:]]))
    return ["\n".join(items)] if items else []


def _render_table(table: Tag) -> list[str]:
    rows = []
    for row in table.find_all("tr"):
        # Rows of a nested table belong to that table's own cell, not to this one.
        if row.find_parent("table") is table:
            cells = row.find_all(["th", "td"], recursive=False)
            rows.append([_table_cell(cell) for cell in cells])
    rows = [row for row in rows if row]
    if not rows:
        return []

    width = max(len(row) for row in rows)
    rendered = [" | ".join(row + [""] * (width - len(row))) for row in rows]
    separator = " | ".join(["---"] * width)
    lines = [rendered[0], separator, *rendered[1:]]
    return ["\n".join(f"| {line} |" for line in lines)]


def _table_cell(cell: Tag) -> str:
    return _collapse(cell.get_text(" ")).replace("|", "\\|")


def _is_block(tag: Tag) -> bool:
    return tag.name in _BLOCK_TAGS or tag.find(_BLOCK_TAGS) is not None


def _inline(element: Tag) -> str:
    """Render inline content on one line, marking code spans with backticks."""
    parts: list[str] = []
    for child in element.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            parts.append(_inline_tag(child))
    return _collapse("".join(parts))


def _inline_tag(tag: Tag) -> str:
    if tag.name == "br":
        return " "
    if tag.name in _CODE_TAGS or "literal" in tag.get("class", []):
        code = _collapse(tag.get_text())
        return f"`{code}`" if code else ""
    return _inline(tag)


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
