# /// script
# requires-python = ">=3.11"
# dependencies = ["llamabot", "pydantic", "rich"]
# ///
"""
Generate a Substack post from a blog post slug.

Usage:
    uv run substack_post.py <blog_slug>
"""

import sys
from pathlib import Path
from typing import List

from llamabot import StructuredBot
from llamabot.prompt_manager import prompt
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

console = Console()


class SubstackSection(BaseModel):
    content: str = Field(..., description="Section content")


class TitleVariant(BaseModel):
    title: str = Field(..., description="Title text")
    variant_type: str = Field(
        ..., description="Type: question_based, emotional_appeal, etc."
    )
    rationale: str = Field(..., description="Why this variant might work")


class SubstackPost(BaseModel):
    title: str = Field(..., description="Primary recommended title")
    title_variants: List[TitleVariant] = Field(..., min_items=2, max_items=4)
    subtitle: str = Field(..., description="Brief subtitle")
    hook_introduction: str = Field(..., description="Strong opening hook")
    purpose_statement: str = Field(..., description="What this post is about")
    main_content: SubstackSection = Field(..., description="Main content")
    key_takeaway: str = Field(..., description="Key takeaway")
    engagement_question: str = Field(..., description="Question to invite replies")
    call_to_action: str = Field(..., description="Call to action")
    signoff: str = Field(..., description="Authentic sign-off")

    def format_post(self) -> str:
        post_content = f"# {self.title}\n\n"
        post_content += "## Title Variants for Testing\n\n"
        for i, variant in enumerate(self.title_variants, 1):
            post_content += f"**Variant {i}: {variant.title}**\n\n"
            post_content += f"*Type: {variant.variant_type}*\n\n"
            post_content += f"*Rationale: {variant.rationale}*\n\n"
        post_content += "---\n\n"
        post_content += f"*{self.subtitle}*\n\n"
        post_content += f"{self.hook_introduction}\n\n"
        post_content += f"{self.purpose_statement}\n\n"
        post_content += f"{self.main_content.content}\n\n"
        post_content += f"{self.key_takeaway}\n\n"
        post_content += f"{self.engagement_question}\n\n"
        post_content += f"*{self.call_to_action}*\n\n"
        post_content += f"{self.signoff}"
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

    Compose a Substack post following best practices.
    Match the tone and style of the original blog post.
    Generate a primary title and 2-4 title variants for A/B testing.
    Include [URL] as a placeholder for the blog post link.
    Start with a clear hook and purpose.
    End with a call to action and authentic sign-off.
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
        console.print("Usage: uv run substack_post.py <blog_slug>")
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
    console.print("[blue]Generating Substack post...[/blue]")
    bot = StructuredBot(sysprompt(), model="gpt-4.1", pydantic_model=SubstackPost)
    social_post = bot(compose_post(post["body"], post["title"]))
    content = social_post.format_post()

    console.print(Panel(content, title="Substack Post", border_style="green"))


if __name__ == "__main__":
    main()
