"""Modal-hosted FastMCP server exposing Eric Ma's blog posts as MCP tools.

Deploy:

    modal --profile ericmjl deploy apis/blog_mcp/server.py

Serves a streamable-HTTP MCP endpoint at:

    https://ericmjl--website-mcp-serve.modal.run/

Tools:
    - list_posts(limit, offset, tag): list post metadata, newest first
    - get_post(slug): full markdown body + metadata for one post
    - search_posts(query, limit): weighted keyword search over title/tags/summary/body
    - list_tags(): tag -> post count

Data source: content/blog/*/contents.lr (Lektor .lr format), baked into the
Modal image at deploy time (contents.lr files only, images excluded).
"""

import difflib
import re
from pathlib import Path

import modal

BLOG_DIR = Path(__file__).resolve().parent.parent.parent / "content" / "blog"
CONTAINER_BLOG_DIR = "/app/content/blog"


def _only_lr(path) -> bool:
    """Ignore-filter for add_local_dir: keep dirs, keep only contents.lr files."""
    p = Path(path)
    if p.is_dir():
        return False
    return p.name != "contents.lr"


app = modal.App("website-mcp")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastmcp>=2.3")
    .add_local_dir(
        str(BLOG_DIR),
        CONTAINER_BLOG_DIR,
        ignore=_only_lr,
    )
)

# ---------------------------------------------------------------------------
# .lr parsing (faithful port of lektor.metaformat.tokenize + _process_buf)
# ---------------------------------------------------------------------------

SITE_BASE = "https://ericmjl.github.io"


def _line_is_dashes(line: str) -> bool:
    line = line.strip()
    return line == "-" * len(line) and len(line) >= 3


def _process_buf(buf: list[str]) -> str:
    lines = []
    for line in buf:
        if _line_is_dashes(line):  # escaped dash-lines lose their first dash
            line = line[1:]
        lines.append(line)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return "".join(lines)


def parse_lr(text: str) -> dict[str, str]:
    """Parse a Lektor .lr file into {field_name: value}, matching Lektor's
    metaformat.tokenize: a line of exactly ``---`` separates fields; blank or
    non-``key:`` lines between a separator and the next field are skipped;
    blank lines between a bare ``key:`` header and its content are skipped."""
    fields: dict[str, str] = {}
    key: str | None = None
    buf: list[str] = []
    want_newline = False

    def flush():
        nonlocal key, buf
        if key is not None:
            fields[key] = _process_buf(buf)
        key, buf = None, []

    for raw in text.splitlines():
        line = raw.rstrip("\r\n") + "\n"
        if line.rstrip() == "---":
            flush()
            want_newline = False
        elif key is not None:
            if want_newline:
                want_newline = False
                if not line.strip():
                    continue
            buf.append(line)
        else:
            bits = line.split(":", 1)
            if len(bits) == 2:
                key = bits[0].strip()
                first_bit = bits[1].strip("\t ")
                if first_bit.strip():
                    buf = [first_bit]
                else:
                    buf = []
                    want_newline = True
    flush()
    return fields


def _post_url(slug: str, pub_date: str | None, body: str = "") -> str | None:
    """Date-based blog URL: /blog/YYYY/M/D/<slug>/ (no zero padding)."""
    if not pub_date:
        # one legacy post carries its date only in the body text
        m = re.search(r"\bpub_date:\s*(\d{4}-\d{2}-\d{2})\b", body)
        if not m:
            return None
        pub_date = m.group(1)
    y, m, d = pub_date.split("-")
    return f"{SITE_BASE}/blog/{int(y)}/{int(m)}/{int(d)}/{slug}/"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_posts(blog_dir: str | Path = BLOG_DIR) -> list[dict]:
    """Parse every contents.lr under blog_dir into searchable post dicts."""
    base = Path(blog_dir)
    posts = []
    for lr in sorted(base.rglob("contents.lr")):
        if lr.parent == base:
            continue  # the blog collection index, not a post
        raw = lr.read_text(encoding="utf-8")
        fields = parse_lr(raw)
        slug = lr.parent.relative_to(base).as_posix()
        pub_date = fields.get("pub_date") or None
        tags = [t.strip() for t in fields.get("tags", "").split("\n") if t.strip()]
        body = fields.get("body", "")
        posts.append(
            {
                "slug": slug,
                "title": fields.get("title", slug),
                "author": fields.get("author", ""),
                "pub_date": pub_date,
                "tags": tags,
                "summary": fields.get("summary", ""),
                "body": body,
                "word_count": len(body.split()),
                "url": _post_url(slug, pub_date, body),
                "_title_lower": fields.get("title", "").lower(),
                "_tags_lower": " ".join(tags).lower(),
                "_summary_lower": fields.get("summary", "").lower(),
                "_body_lower": body.lower(),
            }
        )
    posts.sort(key=lambda p: p["pub_date"] or "0000-00-00", reverse=True)
    return posts


def _meta(post: dict, summary_chars: int = 200) -> dict:
    return {
        "slug": post["slug"],
        "title": post["title"],
        "pub_date": post["pub_date"],
        "tags": post["tags"],
        "summary": post["summary"][:summary_chars],
        "url": post["url"],
    }


def _score_post(post: dict, tokens: list[str]) -> int:
    score = 0
    for tok in tokens:
        score += 5 * post["_title_lower"].count(tok)
        score += 3 * post["_tags_lower"].count(tok)
        score += 2 * post["_summary_lower"].count(tok)
        score += 1 * post["_body_lower"].count(tok)
    return score


def _snippet(post: dict, tokens: list[str], width: int = 150) -> str:
    body_lower = post["_body_lower"]
    idx = -1
    for tok in tokens:
        idx = body_lower.find(tok)
        if idx != -1:
            break
    if idx == -1:
        return post["body"][:width].replace("\n", " ")
    start = max(0, idx - width // 2)
    end = min(len(post["body"]), idx + width)
    text = post["body"][start:end].replace("\n", " ")
    return ("…" if start > 0 else "") + text + ("…" if end < len(post["body"]) else "")


def _search(posts: list[dict], query: str, tag: str | None, limit: int) -> list[dict]:
    tokens = _tokenize(query)
    pool = posts
    if tag:
        t = tag.lower()
        pool = [p for p in posts if any(t in g.lower() for g in p["tags"])]
        if not tokens:
            return [_meta(p) for p in pool[:limit]]
    scored = ((_score_post(p, tokens), p) for p in pool)
    hits = [(s, p) for s, p in scored if s > 0]
    hits.sort(key=lambda sp: sp[0], reverse=True)
    return [
        {"score": s, "snippet": _snippet(p, tokens), **_meta(p)}
        for s, p in hits[:limit]
    ]


# ---------------------------------------------------------------------------
# MCP server (Modal ASGI endpoint)
# ---------------------------------------------------------------------------


@app.function(image=image, timeout=120)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def serve():
    from fastmcp import FastMCP

    posts = load_posts(CONTAINER_BLOG_DIR)
    by_slug = {p["slug"]: p for p in posts}

    mcp = FastMCP(
        "ericmjl-blog",
        instructions=(
            "Search and read Eric Ma's blog (https://ericmjl.github.io): "
            f"{len(posts)} posts on data science, Bayesian statistics, Python, "
            "AI agents, and scientific computing. Use search_posts to find "
            "relevant posts, list_posts to browse, and get_post for full text."
        ),
    )

    @mcp.tool
    def list_posts(limit: int = 20, offset: int = 0, tag: str | None = None) -> dict:
        """List blog posts, newest first. Optionally filter by tag substring."""
        pool = posts
        if tag:
            t = tag.lower()
            pool = [p for p in posts if any(t in g.lower() for g in p["tags"])]
        return {
            "total": len(pool),
            "posts": [_meta(p) for p in pool[offset : offset + limit]],
        }

    @mcp.tool
    def get_post(slug: str) -> dict:
        """Get one blog post's full markdown body plus metadata by slug."""
        post = by_slug.get(slug)
        if post is None:
            lowered = {k.lower(): v for k, v in by_slug.items()}
            post = lowered.get(slug.lower())
        if post is None:
            suggestions = difflib.get_close_matches(
                slug, list(by_slug), n=5, cutoff=0.5
            )
            raise ValueError(
                f"No post with slug '{slug}'. Closest matches: {suggestions}"
            )
        return {
            "slug": post["slug"],
            "title": post["title"],
            "author": post["author"],
            "pub_date": post["pub_date"],
            "tags": post["tags"],
            "summary": post["summary"],
            "url": post["url"],
            "word_count": post["word_count"],
            "body": post["body"],
        }

    @mcp.tool
    def search_posts(query: str, limit: int = 10, tag: str | None = None) -> list[dict]:
        """Keyword-search all posts (title and tags weighted highest).

        Returns ranked results with a matching snippet. Optionally filter
        to posts whose tags contain `tag`.
        """
        return _search(posts, query, tag, limit)

    @mcp.tool
    def list_tags() -> dict:
        """All tags with post counts, most-used first."""
        counts: dict[str, int] = {}
        for p in posts:
            for g in p["tags"]:
                counts[g] = counts.get(g, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    return mcp.http_app(path="/")
