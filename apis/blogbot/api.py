from functools import lru_cache
from pathlib import Path
from typing import Annotated

from bs4 import BeautifulSoup
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llamabot import ImageBot, StructuredBot

from .models import LinkedInPost, SubstackPost, Summary, Tags, TwitterPost
from .prompts import (
    bannerbot_sysprompt,
    compose_linkedin_post,
    compose_substack_post,
    compose_summary,
    compose_tags,
    compose_twitter_post,
    socialbot_sysprompt,
)
from .scraper import get_latest_blog_posts, get_post_body

app = FastAPI()
app.mount("/static", StaticFiles(directory="apis/blogbot/static"), name="static")

bannerbot = ImageBot(size="1792x1024")

templates = Jinja2Templates(directory="apis/blogbot/templates")


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

    if post_type == "linkedin":
        bot = StructuredBot(
            socialbot_sysprompt(), model="gpt-4-turbo", pydantic_model=LinkedInPost
        )
        print("Generating LinkedIn post...")
        social_post = bot(compose_linkedin_post(body, blog_url))
        print("Post generated!")
        content = social_post.format_post()
    elif post_type == "twitter":
        bot = StructuredBot(
            socialbot_sysprompt(), model="gpt-4-turbo", pydantic_model=TwitterPost
        )
        print("Generating Twitter post...")
        social_post = bot(compose_twitter_post(body, blog_url))
        print("Post generated!")
        content = social_post.format_post()
    elif post_type == "substack":
        bot = StructuredBot(
            socialbot_sysprompt(), model="gpt-4-turbo", pydantic_model=SubstackPost
        )
        social_post = bot(compose_substack_post(body, blog_url))
        content = social_post.format_post()
    elif post_type == "summary":
        bot = StructuredBot(
            socialbot_sysprompt(), model="gpt-4-turbo", pydantic_model=Summary
        )
        social_post = bot(compose_summary(body, blog_url))
        content = social_post.content
    elif post_type == "tags":
        bot = StructuredBot(
            socialbot_sysprompt(), model="gpt-4-turbo", pydantic_model=Tags
        )
        tags = bot(compose_tags(body))
        content = "\n".join(tags.content)
    elif post_type == "banner":
        bot = StructuredBot(
            "You are an expert blogger.", model="gpt-4-turbo", pydantic_model=Summary
        )
        summary = bot(compose_summary(body, blog_url)).content
        prompt = f"{bannerbot_sysprompt()}\n\nBlog post summary: {summary}"
        banner_url = bannerbot(prompt, return_url=True)
        return templates.TemplateResponse(
            "banner_result.html", {"request": request, "banner_url": banner_url}
        )

    return templates.TemplateResponse(
        "result.html", {"request": request, "content": content, "post_type": post_type}
    )


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
