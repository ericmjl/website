# /// script
# requires-python = ">=3.11"
# dependencies = ["llamabot", "pydantic", "rich"]
# ///
"""
Generate a LinkedIn post from a blog post slug.

Usage:
    uv run linkedin_post.py <blog_slug>
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


class LinkedInPostSection(BaseModel):
    content: str = Field(..., description="Section content")
    section_type: Optional[str] = Field(
        None, description="Type: story, insight, lesson"
    )


class LinkedInHook(BaseModel):
    line1: str = Field(..., description="Context-Lean Setup, 60 chars or less")
    line2: str = Field(..., description="Scroll-Stop Interjection")
    line3: str = Field(..., description="Curiosity Gap")


class LinkedInAuthorityElement(BaseModel):
    story_type: str = Field(
        ..., description="Type: personal_story, lesson_learned, failure"
    )
    content: str = Field(..., description="Authority-building content")
    specific_example: Optional[str] = Field(None, description="Concrete example")


class LinkedInPost(BaseModel):
    hook: LinkedInHook = Field(..., description="Three-line hook")
    authority_elements: List[LinkedInAuthorityElement] = Field(default_factory=list)
    main_content: List[LinkedInPostSection] = Field(..., description="Main sections")
    call_to_action: str = Field(..., description="Call to action")
    ending_question: str = Field(..., description="Question to generate discussion")
    hashtags: List[str] = Field(..., max_items=5, description="Hashtags with #")

    @model_validator(mode="after")
    def validate_content(self):
        for hashtag in self.hashtags:
            if not hashtag.startswith("#"):
                raise ValueError(f"Hashtag '{hashtag}' must start with '#'.")
        if len(self.hook.line1) > 60:
            raise ValueError("Hook line 1 should be under 60 characters")
        return self

    def format_post(self) -> str:
        post_content = f"{self.hook.line1}\n{self.hook.line2}\n{self.hook.line3}\n\n"
        for element in self.authority_elements:
            post_content += f"{element.content}\n\n"
            if element.specific_example:
                post_content += f"{element.specific_example}\n\n"
        for section in self.main_content:
            post_content += f"{section.content}\n\n"
        post_content += f"{self.call_to_action}\n\n"
        post_content += f"{self.ending_question}\n\n"
        post_content += " ".join([h.lower() for h in self.hashtags])
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
    """This is a blog post titled "{{ title }}" that I just wrote.

        {{ text }}

    Please compose for me a LinkedIn post that follows the provided JSON structure.
    Include [URL] as a placeholder for the blog post link.
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
        console.print("Usage: uv run linkedin_post.py <blog_slug>")
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
    console.print("[blue]Generating LinkedIn post...[/blue]")
    bot = StructuredBot(sysprompt(), model="gpt-4.1", pydantic_model=LinkedInPost)
    social_post = bot(compose_post(post["body"], post["title"]))
    content = social_post.format_post()

    console.print(Panel(content, title="LinkedIn Post", border_style="green"))


if __name__ == "__main__":
    main()
