"""Tag blog posts with categories using an LLM.

This script reads a CSV file containing blog posts, uses an LLM to categorize each post
into predefined categories, and saves the tagged posts to a new CSV file.

Usage:
    uv run scripts/02-year-in-review-tagging.py [OPTIONS] INPUT_PATH

    If no input path is specified, defaults to data/blog_posts.csv

Options:
    --output-path PATH    Path to save tagged posts CSV. Defaults to
                         'tagged_blog_posts.csv' in same directory as input file.
    --help               Show this message and exit.

Input:
    CSV file with at least the following columns:
    - title: The title of the blog post
    - url: The relative URL path of the blog post

Output:
    Creates CSV file with additional columns:
    - categories: List of assigned categories
    - explanation: Reasoning for the category assignments

Example:
    # Basic usage with default paths
    uv run scripts/02-year-in-review-tagging.py

    # Specify custom input and output paths
    uv run scripts/02-year-in-review-tagging.py \
        data/posts.csv \
        --output-path results/tagged.csv

Notes:
    - Requires an OpenAI API key set in the environment
    - The script will log progress and any errors during execution
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "llamabot",
#   "loguru",
#   "requests",
#   "beautifulsoup4",
#   "typer",
# ]
# ///

from enum import Enum, auto
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
import typer
from bs4 import BeautifulSoup
from llamabot import StructuredBot
from loguru import logger
from pydantic import BaseModel


class BlogCategory(Enum):
    """Categories for blog posts."""

    LLMS = auto()
    DS_PRACTICE_LEADERSHIP = auto()
    DS_TOOLING = auto()
    BIOLOGY_CHEMISTRY = auto()
    CAREER_ADVICE = auto()


class BlogTags(BaseModel):
    """Structure for blog post tags."""

    categories: List[BlogCategory]
    explanation: str


def create_tagger_bot() -> StructuredBot:
    """Create a StructuredBot for tagging blog posts.

    :returns: A configured StructuredBot instance.
    """
    system_prompt = """You are a blog post categorizer.
    Given a blog post title and content,
    categorize it into one or more of these categories:
    - LLMs: Posts about large language models, their applications, and implementation
    - Data Science Practice and Leadership: Posts about data science best practices,
      team leadership, and methodology
    - Data Science Tooling: Posts about tools, libraries, and technical implementations
    - Biology and Chemistry: Posts related to biological or chemical applications
    - Career Advice: Posts offering career guidance and professional development

    Provide clear reasoning for your categorization."""

    return StructuredBot(
        system_prompt=system_prompt,
        pydantic_model=BlogTags,
    )


def get_blog_content(url: str) -> str:
    """Scrape blog post content from the given URL.

    :param url: URL of the blog post
    :returns: Extracted blog post content
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the main content in the post_body span
        content = soup.find("span", id="post_body")

        # Also get the summary if available
        summary = soup.find("div", class_="post-summary")

        text_content = []
        if summary:
            text_content.append(summary.get_text(strip=True))
        if content:
            text_content.append(content.get_text(strip=True))

        return "\n\n".join(text_content)
    except Exception as e:
        logger.error(f"Error scraping content from {url}: {str(e)}")
        return ""


def tag_blog_posts(csv_path: Path) -> pd.DataFrame:
    """Tag blog posts with categories using StructuredBot.

    :param csv_path: Path to the CSV file containing blog posts
    :returns: DataFrame with added category tags
    """
    logger.info(f"Reading blog posts from {csv_path}")
    df = pd.read_csv(csv_path)

    tagger = create_tagger_bot()
    tagged_posts = []
    base_url = "https://ericmjl.github.io/blog/"  # Adjust this to your blog's base URL

    for idx, row in df.iterrows():
        # Construct full URL and get content
        full_url = urljoin(base_url, row["url"]) if "url" in row else None
        content = get_blog_content(full_url) if full_url else ""

        # Prepare prompt with both title and content
        prompt = f"Title: {row['title']}\n\nContent: {content}"
        logger.info(f"Tagging post: {row['title']}")

        try:
            tags = tagger(prompt)
            tagged_posts.append(
                {
                    "title": row["title"],
                    "categories": [cat.name for cat in tags.categories],
                    "explanation": tags.explanation,
                }
            )
            logger.success(f"Successfully tagged: {row['title']}")
        except Exception as e:
            logger.error(f"Error tagging post {row['title']}: {str(e)}")
            tagged_posts.append(
                {
                    "title": row["title"],
                    "categories": [],
                    "explanation": f"Error: {str(e)}",
                }
            )

    tagged_df = pd.DataFrame(tagged_posts)
    return pd.merge(df, tagged_df, on="title", how="left")


def main(
    input_path: Path = typer.Argument(
        Path("data/blog_posts.csv"),
        help="Path to input CSV file containing blog posts",
        exists=True,
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        help=(
            "Path to save tagged posts CSV. Defaults to 'tagged_blog_posts.csv' in "
            "same directory as input"
        ),
    ),
) -> None:
    """Tag blog posts with categories using an LLM."""
    if output_path is None:
        output_path = input_path.parent / "tagged_blog_posts.csv"

    tagged_df = tag_blog_posts(input_path)
    tagged_df.to_csv(output_path, index=False)
    logger.success(f"Tagged posts saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)
