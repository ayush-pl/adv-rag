import tarfile
from pathlib import Path

import pytest

from ragbench.ingestion.cleaner import extract_sections, normalize_text, split_fences
from ragbench.ingestion.corpus_manager import build_corpus, load_documents
from ragbench.ingestion.loaders.base import SkipDocument
from ragbench.ingestion.loaders.html_loader import HtmlLoader
from ragbench.ingestion.loaders.markdown_loader import MarkdownLoader
from ragbench.ingestion.models import CorpusManifest, SourceConfig
from ragbench.ingestion.snapshot import extract_content

PADDING = "Enough body text to pass the minimum content length for a real page. " * 4


def markdown_source(**overrides) -> SourceConfig:
    fields = {
        "id": "k8s",
        "name": "Kubernetes Documentation",
        "version": "v1",
        "format": "markdown",
        "url_base": "https://example.org/docs/",
        "content_dir": "docs",
        "fetch": "git",
        "commit": "abc",
    }
    return SourceConfig(**{**fields, **overrides})


def html_source() -> SourceConfig:
    return SourceConfig(
        id="pg-17",
        name="PostgreSQL Documentation",
        version="17",
        format="html",
        url_base="https://example.org/pg/17/",
        content_dir="html",
        fetch="tarball",
        sha256="abc",
    )


def write_page(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_markdown(tmp_path: Path, body: str, relative="docs/a/page.md", **source_overrides):
    loader = MarkdownLoader(markdown_source(**source_overrides), tmp_path, "2026-11-01")
    text = f"---\ntitle: My Page\n---\n{body}\n\n{PADDING}"
    return loader, loader.load(write_page(tmp_path, relative, text))


def load_html(tmp_path: Path, body: str):
    loader = HtmlLoader(html_source(), tmp_path, "2026-11-01")
    text = f"<html><head><title>My Page</title></head><body>{body}<p>{PADDING}</p></body></html>"
    return loader.load(write_page(tmp_path, "html/page.html", text))


class TestCleaner:
    def test_split_fences_separates_code_from_prose(self):
        text = "before\n```yaml\na: 1\n```\nafter\n"
        assert split_fences(text) == [
            (False, "before\n"),
            (True, "```yaml\na: 1\n```\n"),
            (False, "after\n"),
        ]

    def test_shorter_fence_does_not_close_longer_one(self):
        text = "````\n```\ncode\n````\n"
        assert split_fences(text) == [(True, text)]

    def test_normalize_leaves_code_whitespace_alone(self):
        text = "a  \n\n\n\nb\n```\nx  \n\n\n\ny\n```\n"
        assert normalize_text(text) == "a\n\nb\n```\nx  \n\n\n\ny\n```\n"

    def test_normalize_removes_table_padding_outside_code_only(self):
        table = "| Flag      | Text  |\n|---------|-------|\n| `no`      | x |\n"
        code = "```\n| keep      | padding |\n```\n"
        assert normalize_text(table + code) == ("| Flag | Text |\n|---|---|\n| `no` | x |\n" + code)

    def test_sections_ignore_comments_in_code_and_have_correct_offsets(self):
        content = "# Title\n\n```bash\n# not a heading\n```\n\n## Real\n"
        sections = extract_sections(content)
        assert [(s.heading, s.level) for s in sections] == [("Title", 1), ("Real", 2)]
        assert content[sections[1].char_start :].startswith("## Real")


class TestMarkdownLoader:
    def test_metadata_and_url(self, tmp_path):
        _, doc = load_markdown(tmp_path, "Body.")
        assert doc.document_id == "k8s/a/page"
        assert doc.source_url == "https://example.org/docs/a/page/"
        assert doc.title == "My Page"
        assert doc.content.startswith("# My Page\n")
        assert doc.ingestion_date == "2026-11-01"

    def test_index_page_maps_to_parent_path(self, tmp_path):
        _, doc = load_markdown(tmp_path, "Body.", relative="docs/a/_index.md")
        assert doc.document_id == "k8s/a"

    def test_shortcodes_keep_their_text(self, tmp_path):
        body = (
            'A {{< glossary_tooltip text="Pod" term_id="pod" >}} on a '
            '{{< glossary_tooltip term_id="worker-node" >}}.\n\n'
            '{{< feature-state for_k8s_version="v1.30" state="stable" >}}\n\n'
            "{{< note >}}\nCareful.\n{{< /note >}}\n\n"
            '## {{% heading "whatsnext" %}}\n'
        )
        _, doc = load_markdown(tmp_path, body)
        assert "A Pod on a worker node." in doc.content
        assert "FEATURE STATE: Kubernetes v1.30 [stable]" in doc.content
        assert "Note:\nCareful." in doc.content
        assert "## What's next" in doc.content
        assert "{{" not in doc.content

    def test_unresolvable_shortcodes_are_counted(self, tmp_path):
        loader, doc = load_markdown(tmp_path, '{{% include "shared.md" %}}\n\nBody.')
        assert "include" not in doc.content
        assert loader.dropped_markup["include"] == 1

    def test_links_images_and_comments_reduce_to_text(self, tmp_path):
        body = (
            '<!-- overview -->\n[restart policy]({{< ref "/x" >}}), [UID](/docs/uids/#a) '
            "and ![diagram](a.png) and [ref][1].\n\n[1]: https://example.org\n"
            "\n## Heading {#anchor}\n<br>"
        )
        _, doc = load_markdown(tmp_path, body)
        assert "restart policy, UID and  and ref." in doc.content
        assert "## Heading\n" in doc.content
        for leftover in ("overview", "/docs/uids", "a.png", "example.org", "<br>"):
            assert leftover not in doc.content

    def test_code_blocks_are_untouched(self, tmp_path):
        code = '```yaml\nurl: "[a](b)" # {{ .Values.x }}\nimage: <br>\n```'
        _, doc = load_markdown(tmp_path, f"Intro\n\n{code}")
        assert code in doc.content

    def test_code_sample_is_inlined(self, tmp_path):
        write_page(tmp_path, "examples/pods/p.yaml", "kind: Pod\n")
        body = '{{% code_sample language="yaml" file="pods/p.yaml" %}}'
        _, doc = load_markdown(tmp_path, body, examples_dir="examples")
        assert "```yaml\nkind: Pod\n```" in doc.content

    def test_code_sample_cannot_escape_examples_dir(self, tmp_path):
        write_page(tmp_path, "secret.txt", "private")
        (tmp_path / "examples").mkdir()
        body = '{{% code_sample file="../secret.txt" %}}'
        loader, doc = load_markdown(tmp_path, body, examples_dir="examples")
        assert "private" not in doc.content
        assert loader.dropped_markup["code_sample"] == 1

    def test_stub_and_draft_are_skipped(self, tmp_path):
        loader = MarkdownLoader(markdown_source(), tmp_path, "2026-11-01")
        stub = write_page(tmp_path, "docs/stub.md", "---\ntitle: Stub\n---\n")
        draft = write_page(tmp_path, "docs/d.md", f"---\ndraft: true\n---\n{PADDING}")
        for path in (stub, draft):
            with pytest.raises(SkipDocument):
                loader.load(path)

    def test_exclude_patterns_and_stable_order(self, tmp_path):
        for name in ("docs/b.md", "docs/a.md", "docs/contribute/x.md"):
            write_page(tmp_path, name, "x")
        loader = MarkdownLoader(markdown_source(exclude=["contribute/*"]), tmp_path, "d")
        assert [p.name for p in loader.discover()] == ["a.md", "b.md"]


class TestHtmlLoader:
    def test_removes_navigation_and_anchor_links(self, tmp_path):
        body = (
            '<div class="navheader">Prev Next</div>'
            '<h2>7.2. Tables<a class="id_link" href="#x">#</a></h2>'
            '<div class="navfooter">Home</div>'
        )
        doc = load_html(tmp_path, body)
        assert "## 7.2. Tables\n" in doc.content
        for leftover in ("Prev", "Home", "#x"):
            assert leftover not in doc.content

    def test_removes_table_of_contents(self, tmp_path):
        doc = load_html(tmp_path, '<div class="toc"><dl><dt>9.4.1. format</dt></dl></div>')
        assert "9.4.1" not in doc.content

    def test_preserves_code_block_layout(self, tmp_path):
        doc = load_html(tmp_path, '<pre class="programlisting">SELECT 1\n  FROM t;\n</pre>')
        assert "```\nSELECT 1\n  FROM t;\n```" in doc.content

    def test_table_becomes_pipe_table_with_one_row_per_line(self, tmp_path):
        body = (
            "<table><thead><tr><th>Name</th><th>Type</th></tr></thead><tbody>"
            "<tr><td>a|b</td><td><code>int</code>\n value</td></tr></tbody></table>"
        )
        doc = load_html(tmp_path, body)
        assert "| Name | Type |\n| --- | --- |\n| a\\|b | int value |" in doc.content

    def test_inline_code_gets_backticks_and_whitespace_collapses(self, tmp_path):
        doc = load_html(
            tmp_path, '<p>Use\n  <code class="literal">SELECT</code>, <b>then</b>go.</p>'
        )
        assert "Use `SELECT`, thengo." in doc.content

    def test_lists_and_url(self, tmp_path):
        doc = load_html(tmp_path, "<ul><li><p>one</p></li><li>two</li></ul>")
        assert "- one\n- two" in doc.content
        assert doc.document_id == "pg-17/page"
        assert doc.source_url == "https://example.org/pg/17/page.html"


class TestTarballExtraction:
    def make_archive(self, tmp_path: Path) -> Path:
        write_page(tmp_path, "src/pg-1/doc/html/a.html", "page")
        write_page(tmp_path, "src/pg-1/doc/other.txt", "other")
        archive = tmp_path / "archive.tar.bz2"
        with tarfile.open(archive, "w:bz2") as tar:
            tar.add(tmp_path / "src" / "pg-1", arcname="pg-1")
        return archive

    def test_extracts_only_content_dir_without_release_folder(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        extract_content(self.make_archive(tmp_path), out, "doc/html")
        assert (out / "doc/html/a.html").read_text(encoding="utf-8") == "page"
        assert not (out / "doc/other.txt").exists()

    def test_missing_content_dir_is_an_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="no files under doc/missing"):
            extract_content(self.make_archive(tmp_path), tmp_path, "doc/missing")


class TestCorpusManager:
    def test_source_with_no_documents_is_an_error(self, tmp_path):
        (tmp_path / "docs").mkdir()
        manifest = CorpusManifest(
            corpus_version="v1", snapshot_date="2026-11-01", sources=[markdown_source()]
        )
        with pytest.raises(ValueError, match="k8s produced no documents"):
            build_corpus(manifest, {"k8s": tmp_path}, tmp_path / "out", tmp_path / "lock.json")

    def make_corpus(self, tmp_path: Path) -> tuple[CorpusManifest, dict[str, Path]]:
        root = tmp_path / "raw"
        write_page(root, "docs/good.md", f"---\ntitle: Good\n---\n{PADDING}")
        write_page(root, "docs/same.md", f"---\ntitle: Good\n---\n{PADDING}")
        write_page(root, "docs/stub.md", "---\ntitle: Stub\n---\n")
        write_page(root, "docs/broken.md", "---\ntitle: [unclosed\n---\nbody")
        manifest = CorpusManifest(
            corpus_version="v1", snapshot_date="2026-11-01", sources=[markdown_source()]
        )
        return manifest, {"k8s": root}

    def test_records_every_outcome(self, tmp_path):
        manifest, roots = self.make_corpus(tmp_path)
        out = tmp_path / "out"
        lock = build_corpus(manifest, roots, out, tmp_path / "lock.json")

        stats = lock.sources[0]
        assert (stats.documents, stats.failed, stats.skipped, stats.duplicates) == (1, 1, 1, 1)
        assert [d.document_id for d in load_documents(out)] == ["k8s/good"]
        issues = (out / "issues.jsonl").read_text(encoding="utf-8")
        assert "broken.md" in issues and "stub.md" in issues and "same content as" in issues

    def test_rebuild_is_byte_identical(self, tmp_path):
        manifest, roots = self.make_corpus(tmp_path)
        outputs = []
        for name in ("first", "second"):
            lock_path = tmp_path / f"{name}.json"
            build_corpus(manifest, roots, tmp_path / name, lock_path)
            outputs.append(
                (lock_path.read_bytes(), (tmp_path / name / "documents.jsonl").read_bytes())
            )
        assert outputs[0] == outputs[1]

    def test_same_page_in_two_versions_is_kept(self, tmp_path):
        root = tmp_path / "raw"
        write_page(root, "docs/page.md", f"---\ntitle: Same\n---\n{PADDING}")
        manifest = CorpusManifest(
            corpus_version="v1",
            snapshot_date="2026-11-01",
            sources=[markdown_source(id="k8s-1"), markdown_source(id="k8s-2")],
        )
        roots = {"k8s-1": root, "k8s-2": root}
        lock = build_corpus(manifest, roots, tmp_path / "out", tmp_path / "lock.json")
        assert lock.documents == 2
