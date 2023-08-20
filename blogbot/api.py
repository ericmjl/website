import json
from typing import Annotated

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llamabot.prompt_library.blog import (
    blogging_bot,
    compose_linkedin_post,
    compose_twitter_post,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="blogbot/static"), name="static")

bot = blogging_bot()

templates = Jinja2Templates(directory="blogbot/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("layout.html", context={"request": request})


def get_content(url):
    response = requests.get(url)
    webpage_text = "Error"
    if response.status_code == 200:
        html = response.content
        soup = BeautifulSoup(html, "html.parser")
        webpage_text = soup.get_text()
    return response, webpage_text


@app.post("/{social_media}", response_class=HTMLResponse)
async def compose(
    request: Request, blog_url: Annotated[str, Form()], social_media: str
):
    if social_media == "linkedin":
        prompt = compose_linkedin_post
    elif social_media == "twitter":
        prompt = compose_twitter_post
    response, webpage_text = get_content(blog_url)
    if response.status_code == 200:
        response = bot(prompt(webpage_text))
        text = json.loads(response.content)["post_text"]
    else:
        text = "Error"
    return text
