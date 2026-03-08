# /// script
# requires-python = ">=3.11"
# dependencies = ["llamabot", "pydantic", "rich"]
# ///
"""
Generate tags from a blog post slug.

Usage:
    uv run tags.py <blog_slug>
"""

import sys
from pathlib import Path

from llamabot import StructuredBot
from llamabot.prompt_manager import prompt
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.panel import Panel

console = Console()


class Tags(BaseModel):
    content: list[str] = Field(..., description="The list of tags")

    @model_validator(mode="after")
    def validate_tags(self):
        problematic_tags = [tag for tag in self.content if len(tag.split()) > 2]
        if problematic_tags:
            raise ValueError(
                f"Tags must be 2 words or less: {', '.join(problematic_tags)}"
            )
        problematic_tags = [tag for tag in self.content if "-" in tag]
        if problematic_tags:
            raise ValueError(
                f"Tags cannot contain hyphens: {', '.join(problematic_tags)}"
            )
        if len(self.content) < 10:
            raise ValueError("There should be at least 10 tags.")
        one_word_tags = [tag for tag in self.content if len(tag.split()) == 1]
        if len(one_word_tags) < 7:
            raise ValueError("There should be at least 7 one-word tags.")
        return self


@prompt(role="system")
def sysprompt():
    """You are an expert blogger and social media manager.

    You are given a blog post for which to write a social media post.

    Notes:

    - First person, humble, and inviting.
    - Keep it short and concise.
    - Include a placeholder [URL] where the blog post URL should go.
    """


@prompt(role="user")
def compose_post(text):
    """Generate 10 tags for this blog post.
    Maximum two words.
    All lowercase.
    No `#` symbol is needed.
    Spaces are okay (e.g., "web development" not "webdevelopment").
    Here is the blog post:

        {{ text }}
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
        console.print("Usage: uv run tags.py <blog_slug>")
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
    console.print("[blue]Generating tags...[/blue]")
    bot = StructuredBot(sysprompt(), model="gpt-4.1", pydantic_model=Tags)
    tags = bot(compose_post(post["body"]))
    content = "\n".join(tags.content)

    console.print(Panel(content, title="Tags", border_style="green"))


if __name__ == "__main__":
    main()
