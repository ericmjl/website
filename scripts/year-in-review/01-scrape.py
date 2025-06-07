"""Blog post scraper and analyzer for year-in-review summaries.

This script scrapes blog posts from ericmjl.github.io/blog and saves them to a CSV file,
with options to filter by date range.

Usage:
    $ uv run 01-year-in-review-scrape.py \
        --start-date 2023-01-01 \
        --end-date 2023-12-31 \
        --output data/2023_posts.csv

Arguments:
    --start-date, -s  Start date in YYYY-MM-DD format (inclusive)
    --end-date, -e    End date in YYYY-MM-DD format (inclusive)
    --output, -o      Path to save the output CSV file [default: data/blog_posts.csv]

Examples:
    # Scrape all posts from 2023
    $ uv run year-in-review.py -s 2023-01-01 -e 2023-12-31

    # Scrape all posts up to a specific date
    $ uv run year-in-review.py -e 2024-01-01

    # Scrape all posts since a specific date
    $ uv run year-in-review.py -s 2023-06-01

    # Specify custom output path
    $ uv run year-in-review.py -s 2023-01-01 -o path/to/output.csv

The output CSV will contain columns for:
- title: The blog post title
- date: Publication date
- tags: List of post tags
- url: Full URL to the blog post
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "beautifulsoup4",
#   "requests",
#   "pandas",
#   "loguru",
#   "typer",
# ]
# ///

from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urljoin

import pandas as pd
import requests
import typer
from bs4 import BeautifulSoup
from loguru import logger


class BlogPost(NamedTuple):
    """Represents a blog post with its metadata."""

    title: str
    date: datetime
    tags: list[str]
    url: str


def get_soup(url: str) -> BeautifulSoup:
    """
    Get BeautifulSoup object from URL.

    :param url: URL to fetch
    :return: BeautifulSoup object
    """
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_blog_post(post_div) -> BlogPost:
    """
    Parse a blog post div and extract relevant information.

    :param post_div: BeautifulSoup div containing blog post information
    :return: BlogPost object
    """
    # Extract title from the header span with name="title"
    title_span = post_div.find("span", {"name": "title"})
    title = title_span.text.strip()

    # Extract URL from the header link and clean it up
    header = post_div.find("header")
    url = header.find("a").get("href")

    # Clean up the URL by resolving relative paths
    if not url.startswith("http"):
        # First ensure we have a clean base URL
        base_url = "https://ericmjl.github.io/blog/"
        # Remove any relative path components (../../)
        clean_url = url.replace("../", "")
        # Remove any leading slashes
        clean_url = clean_url.lstrip("/")
        # Combine with base URL
        url = urljoin(base_url, clean_url)

    # Extract date from the span with id="pub_date"
    date_str = post_div.find("span", {"id": "pub_date"}).text.strip()
    date = datetime.strptime(date_str, "%Y-%m-%d")

    # Extract tags from the spans with class="boxed"
    tag_spans = post_div.find_all("span", class_="boxed")
    tags = [span.find("a").text.strip() for span in tag_spans]

    return BlogPost(title=title, date=date, tags=tags, url=url)


def scrape_blog_posts(
    base_url: str = "https://ericmjl.github.io/blog/",
    start_date: datetime = None,
    end_date: datetime = None,
) -> list[BlogPost]:
    """
    Scrape blog posts from Eric's website within the date range.

    :param base_url: Base URL of the blog
    :param start_date: Optional date to stop scraping (inclusive)
    :param end_date: Optional date to start scraping from (inclusive)
    :return: List of BlogPost objects
    """
    posts = []
    current_url = base_url
    page_num = 1
    reached_date = False

    while True:
        logger.info(f"Scraping page {page_num} from {current_url}")
        soup = get_soup(current_url)

        # Find all blog post divs (they're in terminal-card class)
        post_divs = soup.find_all("div", class_="terminal-card")
        logger.info(f"Found {len(post_divs)} posts on page {page_num}")

        if not post_divs:
            logger.warning(f"No posts found on page {page_num}")
            break

        for div in post_divs:
            post = parse_blog_post(div)
            logger.debug(f"Parsed post: {post.title} from {post.date}")

            # Skip posts after end_date
            if end_date and post.date > end_date:
                logger.debug(
                    f"Skipping post from {post.date} (after end_date {end_date})"
                )
                continue

            # Stop if we've reached the start date
            if start_date and post.date < start_date:
                logger.info(
                    f"Reached start date {start_date} with post from {post.date}"
                )
                reached_date = True
                break

            posts.append(post)

        if reached_date:
            break

        # Find next page link - look for all pagination links
        pagination_links = soup.find_all("a", class_="btn")
        logger.debug(f"Found {len(pagination_links)} pagination links")

        # Find the next page link
        next_page = None
        for link in pagination_links:
            href = link.get("href", "")
            # Check if this is a page number link
            if href and "page/" in href:
                try:
                    page_num_str = href.split("page/")[1].strip("/")
                    link_page_num = int(page_num_str)
                    # Only consider this link if it's the next page
                    if link_page_num == page_num + 1:
                        next_page = link
                        break
                except (IndexError, ValueError):
                    continue

        if not next_page:
            logger.warning(f"No next page link found on page {page_num}")
            # Try constructing the next page URL manually
            next_url = f"/blog/page/{page_num + 1}/"
            logger.info(f"Manually constructing next URL: {next_url}")
        else:
            next_url = next_page.get("href")

        # Construct the next page URL correctly
        if not next_url.startswith("http"):
            # Handle both /blog/page/N/ and page/N/ formats
            if not next_url.startswith("/"):
                next_url = "/" + next_url
            if not next_url.startswith("/blog/"):
                next_url = "/blog" + next_url
            current_url = f"https://ericmjl.github.io{next_url}"
        else:
            current_url = next_url

        page_num += 1

    logger.info(f"Scraped {len(posts)} posts total")
    return posts


def save_to_csv(posts: list[BlogPost], output_path: Path) -> None:
    """
    Save blog posts to CSV file.

    :param posts: List of BlogPost objects
    :param output_path: Path to save CSV file
    """
    df = pd.DataFrame([post._asdict() for post in posts])
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(posts)} posts to {output_path}")


def main(
    start_date: str = typer.Option(
        None,
        "--start-date",
        "-s",
        help="Start date in YYYY-MM-DD format (inclusive)",
    ),
    end_date: str = typer.Option(
        None,
        "--end-date",
        "-e",
        help="End date in YYYY-MM-DD format (inclusive)",
    ),
    output: Path = typer.Option(
        "data/blog_posts.csv",
        "--output",
        "-o",
        help="Output CSV file path",
    ),
):
    """
    Scrape blog posts from ericmjl.github.io/blog within a date range.
    """
    # Parse dates if provided
    start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    # Log the date range
    date_range = f"from {start.date()}" if start else "from the beginning"
    if end:
        date_range += f" to {end.date()}"
    logger.info(f"Scraping posts {date_range}")

    # Scrape posts
    posts = scrape_blog_posts(start_date=start, end_date=end)

    # Ensure output directory exists
    output.parent.mkdir(exist_ok=True)

    # Save results
    save_to_csv(posts, output)


if __name__ == "__main__":
    typer.run(main)
