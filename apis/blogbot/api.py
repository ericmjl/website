from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llamabot import ImageBot, StructuredBot
from loguru import logger
from PIL import Image
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

from .models import (
    BlueSkyPost,
    DallEImagePrompt,
    LinkedInPost,
    SubstackPost,
    Summary,
    Tags,
)
from .prompts import (
    bannerbot_dalle_prompter_sysprompt,
    compose_bluesky_post,
    compose_feedback_revision,
    compose_linkedin_post,
    compose_substack_post,
    compose_summary,
    compose_tags,
    socialbot_sysprompt,
)
from .scraper import get_latest_blog_posts, get_post_body

app = FastAPI()
app.mount("/static", StaticFiles(directory="apis/blogbot/static"), name="static")

bannerbot = ImageBot(size="1792x1024")

templates = Jinja2Templates(directory="apis/blogbot/templates")

# Add urlencode filter to Jinja2
templates.env.filters["urlencode"] = lambda u: quote(str(u), safe="")


def get_blog_post_directory(blog_url: str) -> Optional[Path]:
    """Extract the blog post slug from URL and find the corresponding directory.

    Args:
        blog_url: URL like https://ericmjl.github.io/blog/2025/11/16/how-i-replaced-307-lines-of-agent-code-with-4-lines/
                  or http://localhost:5959/blog/2025/11/16/how-i-replaced-307-lines-of-agent-code-with-4-lines/

    Returns:
        Path to the blog post directory in content/blog/, or None if not found
    """
    try:
        # Parse the URL to get the path
        from urllib.parse import urlparse

        parsed = urlparse(blog_url)
        path = parsed.path.strip("/")

        # Remove 'blog/' prefix if present
        if path.startswith("blog/"):
            path = path[5:]  # Remove "blog/"

        # Split the path - format is typically YYYY/MM/DD/slug/
        parts = [p for p in path.split("/") if p]

        # The slug is the last non-empty part
        if not parts:
            return None

        slug = parts[-1]

        # Find the directory in content/blog/
        content_blog_path = Path("content/blog")
        blog_dir = content_blog_path / slug

        if blog_dir.exists() and blog_dir.is_dir():
            return blog_dir
        else:
            logger.warning(f"Blog directory not found: {blog_dir}")
            return None
    except Exception as e:
        logger.error(f"Error parsing blog URL {blog_url}: {e}")
        return None


@app.get("/download-logo")
async def download_logo(
    banner_url: Optional[str] = Query(None),
    blog_url: Optional[str] = Query(None),
    save: bool = Query(
        False,
        description=(
            "If True, save directly to blog post directory instead of downloading"
        ),
    ),
):
    """Download or save the logo/banner as webp format.

    If banner_url is provided, downloads/saves that image.
    Otherwise downloads the static logo.
    If save=True and blog_url is provided, saves directly to the blog post directory.
    """
    if banner_url:
        # Fetch the banner image from the URL
        try:
            response = requests.get(banner_url, timeout=30)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))

            # Convert RGBA to RGB if necessary
            # (WebP supports both, but RGB is more compatible)
            if img.mode in ("RGBA", "LA", "P"):
                # Create a white background for transparency
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                rgb_img.paste(
                    img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None
                )
                img = rgb_img
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Convert to WebP
            webp_buffer = BytesIO()
            # Ensure we're saving as WebP format
            img.save(webp_buffer, format="WEBP", quality=95, method=6)
            webp_data = webp_buffer.getvalue()

            # Verify it's actually WebP format (WebP files start with RIFF header)
            if not webp_data.startswith(b"RIFF") or b"WEBP" not in webp_data[:20]:
                logger.warning(
                    "Image may not have been converted to WebP correctly, retrying..."
                )
                # Try again with explicit format
                webp_buffer = BytesIO()
                img.save(webp_buffer, format="WEBP", quality=95)
                webp_data = webp_buffer.getvalue()
                # Check again
                if not webp_data.startswith(b"RIFF") or b"WEBP" not in webp_data[:20]:
                    logger.error("Failed to convert image to WebP format")
                    raise ValueError("WebP conversion failed")

            # If save=True and blog_url is provided, save to blog post directory
            if save and blog_url:
                blog_dir = get_blog_post_directory(blog_url)
                if blog_dir:
                    logo_path = blog_dir / "logo.webp"
                    logo_path.write_bytes(webp_data)
                    # Return HTML response with success message
                    return HTMLResponse(
                        content=f"""
                        <div class="alert alert-success" role="alert">
                            <h5>Logo saved successfully!</h5>
                            <p>Saved to: <code>{logo_path}</code></p>
                            <p>
                                The logo has been saved as <code>logo.webp</code>
                                in your blog post directory.
                            </p>
                        </div>
                        """
                    )
                else:
                    return HTMLResponse(
                        content=f"""
                        <div class="alert alert-danger" role="alert">
                            <h5>Error saving logo</h5>
                            <p>
                                Could not find blog post directory for:
                                <code>{blog_url}</code>
                            </p>
                            <p>
                                Please check that the blog post exists in
                                <code>content/blog/</code>
                            </p>
                        </div>
                        """,
                        status_code=404,
                    )

            # Otherwise, return as download
            return Response(
                content=webp_data,
                media_type="image/webp",
                headers={"Content-Disposition": 'attachment; filename="logo.webp"'},
            )
        except Exception as e:
            logger.error(f"Error fetching banner image: {e}")
            # Fall back to static logo
            return await _download_static_logo()
    else:
        # Use static logo
        return await _download_static_logo()


async def _download_static_logo():
    """Download the static logo as webp format with filename logo.png"""
    logo_path = Path("apis/blogbot/static/img/bars.svg")

    # Convert SVG to PNG using svglib and reportlab
    drawing = svg2rlg(str(logo_path))
    with NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        renderPM.drawToFile(drawing, tmp_file.name, fmt="PNG")
        # Read the PNG file
        with open(tmp_file.name, "rb") as f:
            png_bytes = f.read()
        # Clean up temp file
        Path(tmp_file.name).unlink()

    # Convert PNG to WebP
    img = Image.open(BytesIO(png_bytes))
    webp_buffer = BytesIO()
    img.save(webp_buffer, format="WEBP")
    webp_buffer.seek(0)

    return Response(
        content=webp_buffer.read(),
        media_type="image/webp",
        headers={"Content-Disposition": 'attachment; filename="logo.png"'},
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    base_url = "http://localhost:5959/blog/"
    latest_posts = get_latest_blog_posts(base_url)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "latest_posts": latest_posts, "base_url": base_url},
    )


@app.get("/update_posts", response_class=HTMLResponse)
async def update_posts(request: Request, base_url: str):
    latest_posts = get_latest_blog_posts(base_url)
    return templates.TemplateResponse(
        "blog_post_select.html",
        {"request": request, "latest_posts": latest_posts, "base_url": base_url},
    )


@app.post("/generate/{post_type}", response_class=HTMLResponse)
async def generate_post(
    request: Request, post_type: str, blog_url: Annotated[str, Form()]
):
    _, body = get_post_body(blog_url)
    title_variants = None  # Initialize for non-Substack posts

    if post_type == "linkedin":
        bot = StructuredBot(
            socialbot_sysprompt(), model="gpt-4.1", pydantic_model=LinkedInPost
        )
        logger.info("Generating LinkedIn post...")
        social_post = bot(compose_linkedin_post(body, blog_url))
        logger.info("Post generated!")
        content = social_post.format_post()
    elif post_type == "bluesky":
        bot = StructuredBot(
            socialbot_sysprompt(), model="gpt-4.1", pydantic_model=BlueSkyPost
        )
        logger.info("Generating BlueSky post...")
        social_post = bot(compose_bluesky_post(body, blog_url))
        logger.info("Post generated!")
        content = social_post.format_post()
    elif post_type == "substack":
        bot = StructuredBot(
            socialbot_sysprompt(), model="gpt-4.1", pydantic_model=SubstackPost
        )
        social_post = bot(compose_substack_post(body, blog_url))
        # Format content without title variants (they'll be shown separately in UI)
        content = social_post.format_post(include_title_variants=False)
        # Pass title variants separately for UI display
        title_variants = [
            {
                "title": variant.title,
                "variant_type": variant.variant_type,
                "rationale": variant.rationale,
            }
            for variant in social_post.title_variants
        ]
    elif post_type == "summary":
        bot = StructuredBot(
            socialbot_sysprompt(), model="gpt-4.1", pydantic_model=Summary
        )
        social_post = bot(compose_summary(body, blog_url))
        content = social_post.content
    elif post_type == "tags":
        bot = StructuredBot(socialbot_sysprompt(), model="gpt-4.1", pydantic_model=Tags)
        tags = bot(compose_tags(body))
        content = "\n".join(tags.content)
    elif post_type == "banner":
        dalle_prompt_bot = StructuredBot(
            bannerbot_dalle_prompter_sysprompt(),
            model="gpt-4.1",
            pydantic_model=DallEImagePrompt,
        )
        dalle_prompt = dalle_prompt_bot(body).content
        banner_url = bannerbot(dalle_prompt, return_url=True)
        return templates.TemplateResponse(
            "banner_result.html",
            {"request": request, "banner_url": banner_url, "blog_url": blog_url},
        )

    template_context = {
        "request": request,
        "content": content,
        "post_type": post_type,
        "blog_url": blog_url,
    }
    # Add title variants for Substack posts
    if post_type == "substack":
        template_context["title_variants"] = title_variants

    return templates.TemplateResponse("result.html", template_context)


# SEARCH-RELATED APIs BELOW
# ------------------------- #


@app.get("/search")
def search_home(request: Request):
    return templates.TemplateResponse("search.html", context={"request": request})


site_path = Path("site").resolve()
blog_path = site_path / "blog"


@lru_cache
def get_blog_texts(blog_path) -> dict:
    # List all index.html files under the blog directory, recursively
    source_files = [f for f in blog_path.rglob("index.html")]

    sources = {}
    # print a random source_file
    for source_file in source_files:
        soup = BeautifulSoup(source_file.read_text(), "html.parser")
        post_body = soup.find("div", id="post_body")
        if post_body is not None:
            sources[source_file] = post_body.get_text()
    return sources


def find_publication_date(source):
    publication_date = None
    for line in source.split("\n"):
        if "pub_date" in line:
            publication_date = line.split(":")[1].strip()
    return publication_date


# convert publication_date into URL. Split on "-" and join with "/",
# and remove "0" from month and day.
def convert_pubdate_to_url(pubdate):
    """Convert publication date to URL.

    :param pubdate: Publication date in YYYY-MM-DD format.
    :return: Publication date in YYYY/MM/DD format with leading zeros removed.
    """

    if pubdate is None:
        return ""
    ymd = pubdate.split("-")
    ymd[1] = ymd[1].lstrip("0")
    ymd[2] = ymd[2].lstrip("0")
    return "/".join(ymd)


@app.post("/search/term", response_class=HTMLResponse)
async def search(search_term: Annotated[str, Form()]):
    sources = get_blog_texts(blog_path=blog_path)
    results = []
    for k, v in sources.items():
        if search_term in v:
            results.append(k)
    if not results:
        return "<b>No results found.</b>"
    output = "<ul>"
    for r in results:
        pubdate = find_publication_date(sources[r])
        pubdate_url = convert_pubdate_to_url(pubdate)
        rel_url = r.relative_to(blog_path)
        output += f"<li><a href='https://ericmjl.github.io/blog/{pubdate_url}/{rel_url}'>{rel_url}</a></li>"
    output += "</ul>"
    return output


@app.post("/iterate/{post_type}", response_class=HTMLResponse)
async def iterate_post(
    request: Request,
    post_type: str,
    blog_url: Annotated[str, Form()],
    original_content: Annotated[str, Form()],
    feedback: Annotated[str, Form()],
):
    """Iterate on a social media post based on user feedback."""
    _, body = get_post_body(blog_url)

    # Get the appropriate model based on post type
    if post_type == "linkedin":
        model = LinkedInPost
    elif post_type == "bluesky":
        model = BlueSkyPost
    elif post_type == "substack":
        model = SubstackPost
    elif post_type == "summary":
        model = Summary
    elif post_type == "tags":
        model = Tags
    else:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "content": "Invalid post type",
                "post_type": post_type,
                "blog_url": blog_url,
            },
        )

    # Create bot with appropriate model
    bot = StructuredBot(socialbot_sysprompt(), model="gpt-4.1", pydantic_model=model)

    # Generate revised content
    revised_post = bot(
        compose_feedback_revision(
            original_content=original_content,
            feedback_request=feedback,
            post_type=post_type,
            blog_text=body,
            blog_url=blog_url,
        )
    )

    # Format the revised content
    title_variants = None  # Initialize for non-Substack posts
    if hasattr(revised_post, "format_post"):
        if post_type == "substack":
            # For Substack, format without title variants (they'll be shown separately)
            content = revised_post.format_post(include_title_variants=False)
            # Extract title variants for UI display
            title_variants = [
                {
                    "title": variant.title,
                    "variant_type": variant.variant_type,
                    "rationale": variant.rationale,
                }
                for variant in revised_post.title_variants
            ]
        else:
            content = revised_post.format_post()
    else:
        content = revised_post.content

    template_context = {
        "request": request,
        "content": content,
        "post_type": post_type,
        "blog_url": blog_url,
    }
    # Add title variants for Substack posts
    if post_type == "substack" and title_variants:
        template_context["title_variants"] = title_variants

    return templates.TemplateResponse("result.html", template_context)
