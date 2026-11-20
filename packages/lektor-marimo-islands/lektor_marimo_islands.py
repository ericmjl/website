"""Inline committed marimo island fragments into Lektor pages.

Blog posts opt in with a placeholder::

    <div class="marimo-islands" data-fragment="notebook.islands.html"></div>

The fragment is a sibling of ``contents.lr``, produced by
``scripts/build_marimo_islands.py``. Markdown never sees the fragment, so
custom elements and the islands JSON payload stay intact.
"""

from __future__ import annotations

import re
from pathlib import Path

from lektor.context import get_ctx
from lektor.pluginsystem import Plugin
from markupsafe import Markup

_PLACEHOLDER = re.compile(
    r"<div\b"
    r"(?=[^>]*\bclass=[\"']marimo-islands[\"'])"
    r"(?=[^>]*\bdata-fragment=[\"']([^\"']+)[\"'])"
    r"[^>]*>\s*</div>",
    re.IGNORECASE,
)


class MarimoIslandsPlugin(Plugin):
    """Register the ``inject_marimo_islands`` Jinja filter."""

    name = "Marimo Islands"
    description = "Inline marimo island fragments into Lektor pages."

    def on_setup_env(self, **extra: object) -> None:
        """Add the islands filter to the Jinja environment.

        :param extra: Lektor hook arguments this plugin does not use.
        """
        del extra
        self.env.jinja_env.filters["inject_marimo_islands"] = inject_marimo_islands


def fragment_path(source_filename: str, name: str) -> Path:
    """Resolve a fragment name next to a ``contents.lr`` file.

    :param source_filename: Absolute path of the page's ``contents.lr``.
    :param name: File name from the ``data-fragment`` attribute.
    :returns: Absolute path of the fragment.
    :raises ValueError: If ``name`` is absolute or escapes the page directory.
    """
    relative = Path(name)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError(f"Unsafe marimo islands fragment name: {name}")
    return Path(source_filename).resolve().parent / relative


def inject_marimo_islands(html: str, record: object) -> Markup | str:
    """Replace marimo island placeholders with fragment HTML.

    :param html: Rendered page body.
    :param record: Lektor page record. ``source_filename`` is the contents file.
    :returns: Body HTML, marked safe when a fragment was inserted.
    :raises RuntimeError: If a placeholder is present but cannot be filled.
    :raises ValueError: If a fragment name escapes the page directory.
    """
    text = str(html)
    if "marimo-islands" not in text:
        return html

    source_filename = getattr(record, "source_filename", None)
    if not source_filename:
        raise RuntimeError(
            "Marimo islands placeholder found on a record with no source file."
        )

    def replace(match: re.Match[str]) -> str:
        path = fragment_path(source_filename, match.group(1))
        ctx = get_ctx()
        if ctx is not None:
            ctx.record_dependency(str(path))
        if not path.is_file():
            raise RuntimeError(
                f"Marimo islands fragment {path} is missing. "
                "Run: uv run scripts/build_marimo_islands.py <notebook.py>"
            )
        return path.read_text(encoding="utf-8")

    rendered = _PLACEHOLDER.sub(replace, text)
    if rendered == text:
        raise RuntimeError(
            "Found a marimo-islands marker but could not parse it. "
            'Use <div class="marimo-islands" data-fragment="name.islands.html"></div> '
            "on its own line."
        )
    return Markup(rendered)
