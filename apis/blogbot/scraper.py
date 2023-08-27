"""Scrapers for the blog bot."""
import requests
from bs4 import BeautifulSoup


def get_latest_blog_posts(base_url: str = "https://ericmjl.github.io/blog/") -> dict:
    """Get latest blog posts"""
    response = requests.get(base_url)
    html = response.content
    soup = BeautifulSoup(html, "html.parser")

    # Find the <a> tags nested within the <header> tag
    header = soup.find_all("header")
    titles = [h.get_text().strip("\n") for h in header]
    anchors = [h.find("a", href=True) for h in header]
    hrefs = [base_url + a.get("href") for a in anchors]
    # dictionary of link to title
    links_to_titles = dict(zip(hrefs, titles))
    return links_to_titles


def get_post_body(url: str):
    """Get the body of a blog post."""
    response = requests.get(url)
    webpage_text = "Error"
    if response.status_code == 200:
        html = response.content
        soup = BeautifulSoup(html, "html.parser").find("span", id="post_body")
        if soup is not None:
            webpage_text = soup.get_text()

    return response, webpage_text
