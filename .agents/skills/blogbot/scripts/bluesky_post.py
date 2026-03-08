# /// script
# requires-python = ">=3.11"
# dependencies = ["llamabot", "pydantic", "rich"]
# ///
"""
Generate a BlueSky post from a blog post slug.

Usage:
    uv run bluesky_post.py <blog_slug>
"""

import sys
from pathlib import Path
from typing import List, Optional

from llamabot import StructuredBot
from llamabot.prompt_manager import prompt
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.panel import Panel

console = Console()


class BlueSkyPost(BaseModel):
    strong_hook: str = Field(..., description="Context-Lean Setup, bold and direct")
    clear_stance: str = Field(..., description="Direct, clear stance or insight")
    value_delivery: str = Field(
        ..., description="Deliver value, emotion, or entertainment"
    )
    call_to_action: Optional[str] = Field(None, description="Optional CTA, no URL")
    hashtags: List[str] = Field(..., max_items=2, description="Max 2 hashtags")
    url: str = Field(default="[URL]", description="URL placeholder")

    @model_validator(mode="after")
    def validate_content(self):
        errors = []
        if len(self.hashtags) > 2:
            errors.append("BlueSky post can have a maximum of 2 hashtags.")
        for hashtag in self.hashtags:
            if not hashtag.startswith("#"):
                errors.append(f"Hashtag '{hashtag}' must start with '#.'")
        total_content = self.format_post(with_url=False)
        if len(total_content) > 283:
            errors.append("Total content must be 283 characters or less.")
        if len(total_content) < 100:
            errors.append("Total content must be at least 100 characters.")
        if errors:
            raise ValueError(", ".join(errors))
        return self

    def format_post(self, with_url: bool = True) -> str:
        post_content = f"{self.strong_hook} {self.clear_stance} {self.value_delivery}"
        if self.call_to_action:
            post_content += f" {self.call_to_action}"
        if with_url:
            post_content += f" {self.url}"
        post_content += f" {' '.join(self.hashtags)}"
        return post_content


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
def compose_post(text, title):
    """This is a blog post titled "{{ title }}" that I just wrote:

        {{ text }}

    Please compose for me a BlueSky post that entices my followers to read it.
    Open with a question that the post answers.
    Include a call to action to interact with the post.
    Include hashtags inline (all lowercase).
    DO NOT include the URL - it will be added automatically.
    Write in first-person, humble, and inviting tone.
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
        console.print("Usage: uv run bluesky_post.py <blog_slug>")
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
    console.print("[blue]Generating BlueSky post...[/blue]")
    bot = StructuredBot(sysprompt(), model="gpt-4.1", pydantic_model=BlueSkyPost)
    social_post = bot(compose_post(post["body"], post["title"]))
    content = social_post.format_post()

    console.print(Panel(content, title="BlueSky Post", border_style="green"))
    console.print(f"[dim]Character count: {len(content)}[/dim]")


if __name__ == "__main__":
    main()
