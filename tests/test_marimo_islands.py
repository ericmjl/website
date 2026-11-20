"""Tests for marimo island placeholder injection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "packages" / "lektor-marimo-islands"
sys.path.insert(0, str(PACKAGE))

from lektor_marimo_islands import fragment_path, inject_marimo_islands  # noqa: E402


class Record:
    """Stand-in for a Lektor page record."""

    def __init__(self, source_filename: Path) -> None:
        self.source_filename = str(source_filename)


class InjectTests(unittest.TestCase):
    def test_missing_marker_is_left_alone(self) -> None:
        record = Record(Path("/tmp/contents.lr"))
        body = "<p>No island here.</p>"
        self.assertEqual(inject_marimo_islands(body, record), body)

    def test_fragment_is_inlined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "contents.lr"
            page.write_text("title: demo\n", encoding="utf-8")
            fragment = page.parent / "demo.islands.html"
            fragment.write_text("<marimo-island>hi</marimo-island>\n", encoding="utf-8")
            body = (
                "<p>before</p>\n"
                '<div class="marimo-islands" data-fragment="demo.islands.html"></div>\n'
                "<p>after</p>"
            )
            rendered = str(inject_marimo_islands(body, Record(page)))
        self.assertIn("<marimo-island>hi</marimo-island>", rendered)
        self.assertNotIn("data-fragment", rendered)
        self.assertIn("<p>after</p>", rendered)

    def test_parent_traversal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fragment_path("/tmp/post/contents.lr", "../secret.html")

    def test_missing_fragment_fails(self) -> None:
        page = Path("/tmp/does-not-exist-post/contents.lr")
        body = '<div class="marimo-islands" data-fragment="missing.islands.html"></div>'
        with self.assertRaises(RuntimeError):
            inject_marimo_islands(body, Record(page))


if __name__ == "__main__":
    unittest.main()
