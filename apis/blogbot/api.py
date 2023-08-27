import json
from pathlib import Path
from typing import Annotated

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llamabot import SimpleBot
from .prompts import (
    compose_linkedin_post,
    compose_twitter_post,
    compose_tags,
    compose_summary,
)
from .scraper import get_latest_blog_posts, get_post_body

app = FastAPI()
app.mount("/static", StaticFiles(directory="apis/blogbot/static"), name="static")

bot = SimpleBot(
    "You are an expert blogger. Whenever you use hashtags, they are always lowercase."
)

templates = Jinja2Templates(directory="apis/blogbot/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    latest_blog_posts = get_latest_blog_posts(base_url="http://localhost:5959/blog/")
    print(latest_blog_posts)
    return templates.TemplateResponse(
        "social.html",
        context={"request": request, "latest_blog_posts": latest_blog_posts},
    )


@app.post("/{social_media}", response_class=HTMLResponse)
async def social_media(
    request: Request, blog_url: Annotated[str, Form()], social_media: str
):
    if social_media == "linkedin":
        prompt = compose_linkedin_post
    elif social_media == "twitter":
        prompt = compose_twitter_post
    elif social_media == "tags":
        prompt = compose_tags
    elif social_media == "summary":
        prompt = compose_summary
    response, post_body = get_post_body(blog_url)
    if response.status_code == 200:
        response = bot(prompt(post_body))
        text = json.loads(response.content)["post_text"]
    else:
        text = "Error"
    return text


# SEARCH-RELATED APIs BELOW
# ------------------------- #


@app.get("/search")
def search_home(request: Request):
    return templates.TemplateResponse("search.html", context={"request": request})


site_path = Path("site").resolve()
blog_path = site_path / "blog"


from functools import lru_cache


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


# convert publication_date into URL. Split on "-" and join with "/", and remove "0" from month and day.
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
