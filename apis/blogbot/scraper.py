"""Scrapers for the blog bot."""

from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def extract_rel_path(href: str) -> str:
    """Normalize an anchor href to a host-agnostic relative post path.

    The returned path never carries a scheme/host and never carries a leading
    ``blog/`` segment (the base_url already ends with ``/blog/``), so it can be
    safely appended to any base_url.

    Examples:
        "2026/6/17/slug/"                              -> "2026/6/17/slug"
        "http://localhost:5959/blog/2026/6/17/slug/"   -> "2026/6/17/slug"
        "/blog/2026/6/17/slug/"                        -> "2026/6/17/slug"
    """
    if href.startswith("http"):
        href = urlparse(href).path
    href = href.strip("/")
    if href.startswith("blog/"):
        href = href[len("blog/") :]
    return href.strip("/")


def build_blog_url(base_url: str, blog_path: str) -> str:
    """Construct the canonical blog URL from a base_url and a relative path.

    This is the single source of truth for the URL embedded in generated posts.
    Building it server-side from the *selected* base_url guarantees the output
    URL honors the user's choice, independent of whichever host happened to
    populate the post dropdown (which may be stale or racy via htmx swapping).
    """
    path = blog_path.strip("/")
    return f"{base_url.rstrip('/')}/{path}/"


def get_latest_blog_posts(base_url: str = "https://ericmjl.github.io/blog/") -> dict:
    """Get latest blog posts.

    Returns a dict mapping a host-agnostic relative post path (e.g.
    ``2026/6/17/slug``) to the post title. Keys are intentionally free of any
    scheme/host so the same key produces a correct URL against any base_url.
    """
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        html = response.content
        soup = BeautifulSoup(html, "html.parser")

        # Find the <a> tags nested within the <header> tag
        header = soup.find_all("header")
        if not header:
            print(f"Warning: No <header> tags found on {base_url}")
            return {}

        titles = [h.get_text().strip("\n") for h in header]
        anchors = [h.find("a", href=True) for h in header]
        # Filter out None values in case some headers don't have links
        anchors = [a for a in anchors if a is not None]

        if not anchors:
            print(f"Warning: No links found in <header> tags on {base_url}")
            return {}

        rel_paths = [extract_rel_path(a.get("href")) for a in anchors]
        # dictionary of relative path to title
        return dict(zip(rel_paths, titles))
    except requests.RequestException as e:
        print(f"Error fetching blog posts from {base_url}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error parsing blog posts from {base_url}: {e}")
        return {}


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
