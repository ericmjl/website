# /// script
# requires-python = ">=3.11"
# dependencies = ["llamabot", "litellm", "pydantic", "python-dotenv", "rich"]
# ///
# ruff: noqa: E501
"""
Generate a summary from a blog post slug, routed through Z.ai's GLM-5.2
(the same model used by generate_social.py), with an oMLX fallback.

llamabot's StructuredBot is not used because it rejects glm-5.2; instead this
calls litellm.completion() directly via the shared _glm helper.

Usage:
    uv run summary.py <blog_slug>

Configuration (read from .env or the environment):
    ZAI_API_KEY       (required for GLM) your Z.ai API key (coding plan)
    BLOGBOT_MODEL     (optional) litellm model string, defaults to anthropic/glm-5.2
    BLOGBOT_API_BASE  (optional) defaults to https://api.z.ai/api/anthropic
                      (the Anthropic-compatible coding-plan endpoint)
    BLOGBOT_API_KEY   (optional) oMLX fallback key; else read from ~/.omlx/settings.json
"""

import sys
from pathlib import Path

from _glm import generate_structured
from dotenv import load_dotenv
from llamabot.prompt_manager import prompt
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()


class Summary(BaseModel):
    content: str = Field(..., description="The content of the summary")


@prompt(role="system")
def sysprompt():
    """You are an expert blogger.

    You are given a blog post for which to write its summary field
    (the meta-description shown on the blog index and used for SEO).

    Notes:

    - First person, humble, and inviting.
    - Keep it short and concise.
    - Do NOT include a [URL] placeholder: this summary lives on the post
      itself, so there is no link to insert.
    - Do NOT end with "Read on!" / "read more" / similar: the HTML template
      (templates/macros/blog.html) already appends a "Read on..." link after
      the summary, so those would be duplicated. End at the question instead.
    """


@prompt(role="user")
def compose_post(text, title):
    """Here is my blog post titled "{{ title }}":

        {{ text }}

    I need a summary of the post in 100 words or less.
    Write in first person.
    Start with, "In this blog post".
    End with an engaging question. Do NOT append "Read on!" / "read more"
    or similar; the HTML template already renders the read-on link after
    the summary, so it would be duplicated.
    """


def find_repo_root() -> Path:
    current = Path(__file__).parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd()


def parse_contents_lr(file_path: Path) -> dict:
    content = file_path.read_text()
    fields = {}
    current_field = None
    current_value = []

    for line in content.split("\n"):
        if line == "---":
            if current_field and current_value:
                fields[current_field] = "\n".join(current_value).strip()
            current_field = None
            current_value = []
        elif current_field is None and ":" in line:
            field_name, field_value = line.split(":", 1)
            field_name = field_name.strip()
            field_value = field_value.strip()
            if field_value:
                fields[field_name] = field_value
            else:
                current_field = field_name
        elif current_field:
            current_value.append(line)

    if current_field and current_value:
        fields[current_field] = "\n".join(current_value).strip()

    return fields


def get_local_blog_post(slug: str) -> dict:
    repo_root = find_repo_root()
    content_dir = repo_root / "content"
    blog_path = content_dir / "blog" / slug / "contents.lr"

    if not blog_path.exists():
        raise FileNotFoundError(f"Blog post not found: {blog_path}")

    fields = parse_contents_lr(blog_path)

    return {
        "title": fields.get("title", ""),
        "body": fields.get("body", ""),
        "slug": slug,
        "path": str(blog_path),
    }


def list_blog_posts() -> list[dict]:
    repo_root = find_repo_root()
    content_dir = repo_root / "content"
    blog_dir = content_dir / "blog"
    posts = []

    for post_dir in sorted(
        blog_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if post_dir.is_dir():
            contents_lr = post_dir / "contents.lr"
            if contents_lr.exists():
                try:
                    fields = parse_contents_lr(contents_lr)
                    posts.append(
                        {
                            "slug": post_dir.name,
                            "title": fields.get("title", post_dir.name),
                        }
                    )
                except Exception:
                    posts.append({"slug": post_dir.name, "title": post_dir.name})

    return posts


def main():
    if len(sys.argv) < 2:
        console.print("[red]Error: Please provide a blog slug[/red]")
        console.print("Usage: uv run summary.py <blog_slug>")
        console.print("\nAvailable blog posts:")
        for post in list_blog_posts()[:10]:
            console.print(f"  - {post['slug']}")
        sys.exit(1)

    slug = sys.argv[1]
    console.print(f"[blue]Fetching blog post:[/blue] {slug}")

    try:
        post = get_local_blog_post(slug)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    console.print(f"[blue]Title:[/blue] {post['title']}")
    console.print("[blue]Generating summary...[/blue]")
    summary = generate_structured(
        compose_post(post["body"], post["title"]).content,
        Summary,
        sysprompt().content,
    )

    console.print(Panel(summary.content, title="Summary", border_style="green"))


if __name__ == "__main__":
    main()
