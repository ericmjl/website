import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote

from bs4 import BeautifulSoup
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llamabot import SimpleBot
from loguru import logger

from .prompts import (
    compose_linkedin_post,
    compose_patreon_post,
    compose_substack_post,
    compose_summary,
    compose_tags,
    compose_twitter_post,
)
from .scraper import get_latest_blog_posts, get_post_body

app = FastAPI()
app.mount("/static", StaticFiles(directory="apis/blogbot/static"), name="static")

social_bot = SimpleBot(
    "You are an expert blogger.",
    model="gpt-4-0125-preview",
    json_mode=True,
)

tagbot = SimpleBot(
    ("You are an expert tagger of blog posts. "
     "Return lowercase tags for the following blog post."),
    model="gpt-4-0125-preview",
    json_mode=True,
)

templates = Jinja2Templates(directory="apis/blogbot/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    latest_blog_posts = get_latest_blog_posts(base_url="http://localhost:5959/blog/")
    return templates.TemplateResponse(
        "social.html",
        context={"request": request, "latest_blog_posts": latest_blog_posts},
    )


@app.get("/latest-blog-posts", response_class=HTMLResponse)
async def get_latest_blog_posts_api(request: Request):
    base_url = request.query_params.get("base_url")
    base_url = unquote(base_url)
    print(base_url)
    latest_blog_posts: dict[str, str] = get_latest_blog_posts(base_url=base_url)
    print(latest_blog_posts)
    response = ""
    for url, title in latest_blog_posts.items():
        response += f"<option value='{url}'>{title}</option>\n"
    print(response)
    return response


@app.post("/{post_type}", response_class=HTMLResponse)
async def social_media(
    request: Request, blog_url: Annotated[str, Form()], post_type: str
):
    bot = social_bot
    if post_type == "linkedin":
        prompt = compose_linkedin_post
    elif post_type == "twitter":
        prompt = compose_twitter_post
    elif post_type == "tags":
        prompt = compose_tags
        bot = tagbot
    elif post_type == "summary":
        prompt = compose_summary
    elif post_type == "patreon":
        prompt = compose_patreon_post
    elif post_type == "substack":
        prompt = compose_substack_post
    response, post_body = get_post_body(blog_url)
    text = "Error"
    if response.status_code == 200:
        social_post = bot(prompt(post_body))
        # Post-processing

        try:
            bot_text = social_post.content.replace("'response_text'", '"response_text"')
            logger.info(bot_text)
            bot_text = bot_text.replace("\n", "\\n").replace("\t", "\\t")
            logger.info(bot_text)
            text = json.loads(bot_text)["response_text"]
            logger.info(text)

            if post_type == "tags":
                text = "\n".join(line for line in text)
        except Exception as e:
            text = f"Error: {e}"
            text += "\n\n"
            text += "Generated text: \n\n"
            text += social_post.content
    return text


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
