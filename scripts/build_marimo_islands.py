# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer",
#   "loguru",
# ]
# ///
"""Build a marimo islands HTML fragment from a notebook.

The fragment is what a blog post inlines. Generation runs in a uv environment
that includes marimo and the notebook's own PEP 723 dependencies. Cell code is
embedded for the browser; cells are not executed here, because anywidget
previews pick a new random id on every run and the browser executes the cells
anyway.

Usage:
    uv run scripts/build_marimo_islands.py path/to/notebook.py
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import typer
from loguru import logger

# Keep this aligned with the @marimo-team/islands CDN URL the generator emits.
MARIMO_VERSION = "0.25.2"
# jsDelivr's cached main.js for this version points at chunk hashes that 404.
# The page loads the copy unpacked by scripts/fetch_marimo_islands_assets.py.
ISLANDS_CDN_DIST = (
    f"https://cdn.jsdelivr.net/npm/@marimo-team/islands@{MARIMO_VERSION}/dist"
)
ISLANDS_LOCAL_DIST = f"/static/marimo-islands/{MARIMO_VERSION}"

app = typer.Typer(add_completion=False, no_args_is_help=True)


def pep723_dependencies(notebook: Path) -> list[str]:
    """Return dependency strings from a notebook's PEP 723 script block.

    :param notebook: Path to a marimo ``.py`` notebook.
    :returns: Requirement strings, or an empty list when the block is absent.
    """
    text = notebook.read_text(encoding="utf-8")
    match = re.search(r"^# /// script\n(?P<body>(?:#.*\n)*)# ///", text, re.M)
    if match is None:
        return []
    body_lines: list[str] = []
    for line in match.group("body").splitlines():
        if line.startswith("# "):
            body_lines.append(line[2:])
        elif line.startswith("#"):
            body_lines.append(line[1:])
        else:
            body_lines.append(line)
    body = "\n".join(body_lines)
    dep_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", body, re.S)
    if dep_match is None:
        return []
    return re.findall(r"""["']([^"']+)["']""", dep_match.group(1))


def _requirement_name(requirement: str) -> str:
    """Return the distribution name from a PEP 508 requirement string.

    :param requirement: A requirement such as ``anywidget>=0.9``.
    :returns: The name portion, lowercased.
    """
    return re.split(r"[<>=\[]", requirement, maxsplit=1)[0].strip().lower()


def _uv_executable() -> str:
    """Return a uv binary path.

    :returns: ``uv`` on ``PATH``, or the user-local installer path.
    """
    from shutil import which

    found = which("uv")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "uv"
    if local.is_file():
        return str(local)
    raise RuntimeError("uv is not installed; see https://docs.astral.sh/uv/")


def _reexec_with_notebook_deps(notebook: Path, output: Path) -> None:
    """Re-run this script under uv with marimo and the notebook's dependencies.

    :param notebook: Notebook to generate from.
    :param output: Fragment path to write.
    """
    requirements = [f"marimo=={MARIMO_VERSION}", "typer", "loguru"]
    for requirement in pep723_dependencies(notebook):
        if _requirement_name(requirement) == "marimo":
            continue
        requirements.append(requirement)
    command = [_uv_executable(), "run"]
    for requirement in requirements:
        command.extend(["--with", requirement])
    command.extend(
        [
            "--",
            str(Path(__file__).resolve()),
            str(notebook),
            "--output",
            str(output),
        ]
    )
    env = os.environ.copy()
    env["MARIMO_ISLANDS_INNER"] = "1"
    logger.info("Generating islands with {}", ", ".join(requirements))
    raise SystemExit(subprocess.call(command, env=env))


def _loading_script() -> str:
    """Return the snippet that removes the loading line once a canvas exists.

    :returns: An HTML paragraph plus a classic script.
    """
    return """<p class="marimo-islands-status">Loading the interactive viewer...</p>
<script>
(function () {
  var script = document.currentScript;
  var host = script && script.closest(".marimo-islands-host");
  if (!host) return;
  function hasCanvas(root) {
    if (root.querySelector("canvas, .pv-container")) return true;
    var nodes = root.querySelectorAll("*");
    for (var i = 0; i < nodes.length; i++) {
      if (nodes[i].shadowRoot && hasCanvas(nodes[i].shadowRoot)) return true;
    }
    return false;
  }
  function settle() {
    if (!hasCanvas(host)) return;
    var status = host.querySelector(".marimo-islands-status");
    if (status) status.remove();
    clearInterval(timer);
  }
  // The 3Dmol canvas is created inside an open shadow root, which a
  // light-DOM MutationObserver does not see. Polling catches it.
  var timer = setInterval(settle, 200);
  settle();
})();
</script>"""


def localize_islands_runtime(fragment: str) -> str:
    """Point the islands runtime at the self-hosted dist.

    :param fragment: HTML produced by ``MarimoIslandGenerator``.
    :returns: The same HTML with the jsDelivr dist URL replaced.
    :raises RuntimeError: If the generator did not emit the expected URL.
    """
    if ISLANDS_CDN_DIST not in fragment:
        raise RuntimeError(
            "Islands head did not contain "
            f"{ISLANDS_CDN_DIST}. Refusing to write a fragment that "
            "depends on the jsDelivr cache."
        )
    localized = fragment.replace(ISLANDS_CDN_DIST, ISLANDS_LOCAL_DIST)
    # A title on a stylesheet makes it a preferred stylesheet. Drop it so
    # every browser applies the islands CSS as a normal persistent sheet.
    return localized.replace(' title="marimo-islands"', "")


def render_fragment(notebook: Path) -> str:
    """Render a notebook as an islands fragment for inlining into a page.

    :param notebook: Path to a marimo ``.py`` notebook.
    :returns: HTML containing the islands runtime tags, cells, and JSON payload.
    """
    from marimo import MarimoIslandGenerator

    generator = MarimoIslandGenerator.from_file(str(notebook), display_code=False)
    parts = [
        '<div class="marimo-islands-host">',
        _loading_script(),
        generator.render_head(version_override=MARIMO_VERSION),
    ]
    for stub in generator.stubs:
        parts.append(stub.render(display_code=False))
    parts.append(generator.render_payload_script())
    parts.append("</div>")
    fragment = "\n".join(parts).rstrip() + "\n"
    fragment = "\n".join(line.rstrip() for line in fragment.splitlines()) + "\n"
    return localize_islands_runtime(fragment)


@app.command()
def build(
    notebook: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Marimo notebook to embed.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Fragment path. Defaults to <notebook>.islands.html beside the notebook.",
    ),
) -> None:
    """Write an islands HTML fragment for a marimo notebook."""
    destination = output or notebook.with_suffix(".islands.html")
    if os.environ.get("MARIMO_ISLANDS_INNER") != "1":
        _reexec_with_notebook_deps(notebook, destination)
        return
    fragment = render_fragment(notebook)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(fragment, encoding="utf-8")
    logger.info("Wrote {} ({} bytes)", destination, destination.stat().st_size)


if __name__ == "__main__":
    app()
