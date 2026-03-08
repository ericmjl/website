# /// script
# requires-python = ">=3.11"
# dependencies = ["llamabot", "pydantic", "rich", "pillow", "requests"]
# ///
"""
Generate a DALL-E banner image from a blog post slug and save as logo.webp.

Usage:
    uv run banner.py <blog_slug>
"""

import sys
from io import BytesIO
from pathlib import Path

import requests
from llamabot import ImageBot, StructuredBot
from llamabot.prompt_manager import prompt
from PIL import Image
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

console = Console()


class DallEImagePrompt(BaseModel):
    content: str = Field(..., description="The DALL-E image prompt")


@prompt(role="system")
def dalle_sysprompt():
    """
    **As 'Prompt Designer',**
    your role is to create highly detailed and imaginative prompts for DALL-E,
    designed to generate banner images for blog posts in a watercolor style,
    with a 16:4 aspect ratio.

    You will be given a chunk of text or a summary that comes from the blog post.
    Your task is to translate the key concepts, ideas,
    and themes from the text into an image prompt.

    **Guidelines for creating the prompt:**
    - Use vivid and descriptive language to specify the image's mood, colors,
      composition, and style.
    - Vary your approach significantly between prompts - avoid repetitive patterns,
      elements, or compositions that could make images look similar.
    - Explore diverse watercolor techniques: washes, wet-on-wet, dry brush,
      salt effects, splattering, or layered glazes.
    - Consider different artistic styles within watercolor: impressionistic,
      expressionistic, minimalist, detailed botanical, atmospheric, or abstract.
    - Vary the color palettes: warm vs cool tones, monochromatic vs complementary,
      muted vs vibrant, or seasonal color schemes.
    - Mix different compositional approaches: centered focal points, rule of thirds,
      diagonal compositions, or asymmetrical balance.
    - Incorporate varied symbolic elements: natural objects, architectural forms,
      organic shapes, geometric patterns, or conceptual representations.
    - Focus on maximizing the use of imagery and symbols to represent ideas,
      avoiding any inclusion of text or character symbols in the image.
    - If the text is vague or lacks detail, make thoughtful and creative assumptions
      to create a compelling visual representation.

    The prompt should be suitable for a variety of blog topics,
    evoking an emotional or intellectual connection to the content.
    Ensure the description specifies the watercolor art style,
    the wide 16:4 banner aspect ratio,
    and your chosen artistic approach.

    Do **NOT** include any text or character symbols in the image description.
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


def download_and_convert_image(url: str, output_path: Path) -> None:
    """Download image from URL, convert to WebP, and save."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    if img.mode in ("RGBA", "LA", "P"):
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = rgb_img
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img.save(output_path, "WEBP", quality=95)


def main():
    if len(sys.argv) < 2:
        console.print("[red]Error: Please provide a blog slug[/red]")
        console.print("Usage: uv run banner.py <blog_slug>")
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
    console.print("[blue]Generating DALL-E prompt...[/blue]")
    dalle_prompt_bot = StructuredBot(
        dalle_sysprompt(), model="gpt-4.1", pydantic_model=DallEImagePrompt
    )
    dalle_prompt = dalle_prompt_bot(post["body"]).content

    console.print(Panel(dalle_prompt, title="DALL-E Prompt", border_style="yellow"))
    console.print("[blue]Generating banner image...[/blue]")

    bannerbot = ImageBot(size="1792x1024")
    banner_url = bannerbot(dalle_prompt, return_url=True)

    console.print("[blue]Downloading and converting to WebP...[/blue]")
    blog_dir = Path(post["path"]).parent
    output_path = blog_dir / "logo.webp"
    download_and_convert_image(banner_url, output_path)

    console.print(Panel(str(output_path), title="Saved", border_style="green"))


if __name__ == "__main__":
    main()
