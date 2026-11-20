# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Install a self-consistent marimo islands runtime under assets/.

jsDelivr caches ``@marimo-team/islands@0.25.2/dist/main.js`` for a year.
That cached file points at chunk names (including the Pyodide worker) that
the republished 0.25.2 package no longer contains, so those requests 404
and the island never leaves "Loading the interactive viewer...". unpkg and
the npm tarball agree with each other. This script unpacks that tarball,
checks its sha512, and writes it where the blog fragment loads it from.

The output directory is gitignored. ``pixi run build`` runs this first.
"""

from __future__ import annotations

import base64
import hashlib
import io
import shutil
import tarfile
import urllib.request
from pathlib import Path

VERSION = "0.25.2"
INTEGRITY = (
    "sha512-XGqFLhlswlC4DtgKqQ6FcD9jEdugq+w2UYHpDpwrFb3RYdySPaRpP8XOVW4V/"
    "dbTsjAuJv7NkAiC3zJO665HVw=="
)
TARBALL_URL = "https://registry.npmjs.org/@marimo-team/islands/-/islands-0.25.2.tgz"
DIST_PREFIX = "package/dist/"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "assets" / "static" / "marimo-islands" / VERSION


def _sha512(payload: bytes) -> str:
    """Return an npm-style sha512 integrity string.

    :param payload: Bytes to hash.
    :returns: ``sha512-`` plus the base64 digest.
    """
    digest = hashlib.sha512(payload).digest()
    return "sha512-" + base64.b64encode(digest).decode("ascii")


def _already_installed() -> bool:
    """Return whether DEST already matches INTEGRITY.

    :returns: True when the stamp and main.js are both present.
    """
    stamp = DEST / ".sha512"
    return (
        stamp.is_file()
        and (DEST / "main.js").is_file()
        and stamp.read_text().strip() == INTEGRITY
    )


def _safe_relative(name: str) -> Path:
    """Return the path of a dist member relative to DEST.

    :param name: Tar member name.
    :returns: Relative path under the dist directory.
    :raises ValueError: If the member escapes the dist prefix.
    """
    if not name.startswith(DIST_PREFIX):
        raise ValueError(name)
    relative = Path(name[len(DIST_PREFIX) :])
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError(name)
    return relative


def install() -> None:
    """Download, verify, and unpack the islands dist."""
    if _already_installed():
        print(f"marimo islands {VERSION} already installed at {DEST}")
        return
    print(f"Downloading {TARBALL_URL}")
    with urllib.request.urlopen(TARBALL_URL, timeout=120) as response:
        payload = response.read()
    got = _sha512(payload)
    if got != INTEGRITY:
        raise SystemExit(f"sha512 mismatch for islands {VERSION}: {got}")
    if DEST.exists():
        # A partial unpack should not be mixed with a new tarball.
        for child in DEST.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        DEST.mkdir(parents=True)
    written = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith(DIST_PREFIX):
                continue
            relative = _safe_relative(member.name)
            target = DEST / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise SystemExit(f"Could not read {member.name}")
            target.write_bytes(extracted.read())
            written += 1
    (DEST / ".sha512").write_text(INTEGRITY + "\n", encoding="utf-8")
    worker = list((DEST / "assets").glob("worker-*.js"))
    if not (DEST / "main.js").is_file() or not worker:
        raise SystemExit(f"islands {VERSION} unpack is missing main.js or the worker")
    print(f"Installed {written} files into {DEST}")


if __name__ == "__main__":
    install()
