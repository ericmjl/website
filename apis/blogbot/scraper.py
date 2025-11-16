"""Scrapers for the blog bot."""

import requests
from bs4 import BeautifulSoup


def get_latest_blog_posts(base_url: str = "https://ericmjl.github.io/blog/") -> dict:
    """Get latest blog posts"""
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

        hrefs = [
            base_url.rstrip("/") + "/" + a.get("href").lstrip("/")
            if not a.get("href").startswith("http")
            else a.get("href")
            for a in anchors
        ]
        # dictionary of link to title
        links_to_titles = dict(zip(hrefs, titles))
        return links_to_titles
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
