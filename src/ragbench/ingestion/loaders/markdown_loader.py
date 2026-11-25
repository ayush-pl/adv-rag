"""Loader for Hugo-flavoured Markdown, as used by the Kubernetes and Docker docs."""

import re
from pathlib import Path

import yaml

from ragbench.ingestion.cleaner import map_prose
from ragbench.ingestion.loaders.base import DocumentLoader, SkipDocument
from ragbench.ingestion.models import Document

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_SHORTCODE = re.compile(r"\{\{[<%]\s*(/?)\s*([\w-]*)\s*(.*?)\s*/?\s*[%>]\}\}", re.DOTALL)
_ARGUMENT = re.compile(r"""([\w-]+)\s*=\s*"([^"]*)\"""")
_FIRST_STRING = re.compile(r'"([^"]*)"')
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_REF_DEFINITION = re.compile(r"^\[[^\]]+\]:\s+\S.*$", re.MULTILINE)
_HEADING_ATTRS = re.compile(r"^(#{1,6}\s.*?)\s*\{[#.][^}]*\}\s*$", re.MULTILINE)
_HTML_TAG = re.compile(
    r"</?(?:a|b|br|code|details|div|em|i|img|li|ol|p|span|strong|sub|summary|sup|table|tbody"
    r"|td|th|thead|tr|ul)(?:\s[^<>]*)?/?>",
    re.IGNORECASE,
)
_ALERT = re.compile(r"\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]")

_ADMONITIONS = {"note": "Note:", "caution": "Caution:", "warning": "Warning:"}
_HEADING_NAMES = {
    "prerequisites": "Before you begin",
    "whatsnext": "What's next",
    "objectives": "Objectives",
    "cleanup": "Clean up",
    "synopsis": "Synopsis",
    "options": "Options",
}


class MarkdownLoader(DocumentLoader):
    suffix = ".md"

    def load(self, path: Path) -> Document:
        raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        front_matter, body = _split_front_matter(raw)
        if front_matter.get("draft"):
            raise SkipDocument("marked as draft")

        body = map_prose(_COMMENT.sub("", body), self._clean_prose)
        title = str(front_matter.get("title") or "").strip() or _first_heading(body) or path.stem
        if not body.lstrip().startswith("# "):
            body = f"# {title}\n\n{body}"

        slug = _slug(path.relative_to(self.content_root).as_posix())
        return self._build_document(slug, f"{slug}/" if slug else "", title, body)

    def _clean_prose(self, prose: str) -> str:
        prose = _SHORTCODE.sub(self._replace_shortcode, prose)
        prose = _IMAGE.sub("", prose)
        prose = _LINK.sub(r"\1", prose)
        prose = _REF_LINK.sub(r"\1", prose)
        prose = _REF_DEFINITION.sub("", prose)
        prose = _HEADING_ATTRS.sub(r"\1", prose)
        prose = _HTML_TAG.sub("", prose)
        return _ALERT.sub(lambda m: m.group(1).capitalize() + ":", prose)

    def _replace_shortcode(self, match: re.Match[str]) -> str:
        closing, name, raw_args = match.group(1), match.group(2), match.group(3)
        if closing:
            return ""

        args = dict(_ARGUMENT.findall(raw_args))
        if name in _ADMONITIONS:
            return _ADMONITIONS[name]
        if name == "glossary_tooltip":
            return args.get("text") or args.get("term_id", "").replace("-", " ")
        if name == "heading":
            first = _FIRST_STRING.search(raw_args)
            key = first.group(1) if first else ""
            return _HEADING_NAMES.get(key, key.replace("-", " ").capitalize())
        if name == "feature-state":
            version = args.get("for_k8s_version", "")
            return f"FEATURE STATE: Kubernetes {version} [{args.get('state', '')}]"
        if name == "tab" and "name" in args:
            return f"{args['name']}:"
        if name == "code_sample":
            sample = self._read_code_sample(args)
            if sample is not None:
                return sample

        # Nothing else carries text we can resolve without running Hugo: includes, refs, layout.
        self.dropped_markup[name or "unnamed"] += 1
        return ""

    def _read_code_sample(self, args: dict[str, str]) -> str | None:
        if self.source.examples_dir is None or "file" not in args:
            return None
        examples_root = (self.source_root / self.source.examples_dir).resolve()
        sample_path = (examples_root / args["file"]).resolve()
        if not sample_path.is_relative_to(examples_root) or not sample_path.is_file():
            return None
        language = args.get("language", "")
        code = sample_path.read_text(encoding="utf-8").strip()
        return f"\n```{language}\n{code}\n```\n"


def _split_front_matter(raw: str) -> tuple[dict, str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    parsed = yaml.safe_load(match.group(1))
    return (parsed if isinstance(parsed, dict) else {}), raw[match.end() :]


def _first_heading(body: str) -> str | None:
    match = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
    return match.group(1) if match else None


def _slug(relative_path: str) -> str:
    """Map a content path to its URL path: `a/b.md` -> `a/b`, `a/_index.md` -> `a`."""
    without_suffix = relative_path.removesuffix(".md")
    parent, _, stem = without_suffix.rpartition("/")
    return parent if stem in {"_index", "index"} else without_suffix
