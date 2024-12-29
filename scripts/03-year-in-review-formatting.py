"""Format blog post data into markdown for year-in-review post.

This script takes the tagged blog posts CSV file and formats it into markdown,
either as a table or as categorized lists.

Usage with uv run:

    # Generate markdown table
    uv run scripts/03-year-in-review-formatting.py --format table

    # Generate categorized lists
    uv run scripts/03-year-in-review-formatting.py --format categories

The script expects the tagged blog posts data to be in data/tagged_blog_posts.csv.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "loguru",
#   "typer",
# ]
# ///

from pathlib import Path

import pandas as pd
import typer
from loguru import logger


def load_blog_posts() -> pd.DataFrame:
    """Load the tagged blog posts CSV file.

    :returns: DataFrame containing blog post data
    """
    csv_path = Path("data/tagged_blog_posts.csv")
    return pd.read_csv(csv_path)


def format_categories(categories_str: str) -> str:
    """Format the categories string into a readable format.

    :param categories_str: String representation of categories list
    :returns: Formatted categories string
    """
    # Convert string representation of list to actual list
    categories = eval(categories_str)

    # Define category name mappings
    category_mappings = {
        "DS_PRACTICE_LEADERSHIP": "Data Science Practice & Leadership",
        "DS_TOOLING": "Data Science Tooling",
        "BIOLOGY_CHEMISTRY": "Biology & Chemistry",
        "LLMS": "LLMs",
        "CAREER_ADVICE": "Career Advice",
    }

    # Map categories to their display names
    formatted = [category_mappings[cat] for cat in categories]
    return ", ".join(formatted)


def create_markdown_table(df: pd.DataFrame) -> str:
    """Create a markdown-formatted table from the blog posts data.

    :param df: DataFrame containing blog post data
    :returns: Markdown-formatted table as string
    """
    # Create table header
    table = "| Date | Title | Categories |\n"
    table += "|------|-------|------------|\n"

    # Sort by date ascending (earliest first)
    df = df.sort_values("date", ascending=True)

    # Add each row
    for _, row in df.iterrows():
        title_with_link = f"[{row['title']}]({row['url']})"
        categories = format_categories(row["categories"])
        table += f"| {row['date']} | {title_with_link} | {categories} |\n"

    return table


def create_category_lists(df: pd.DataFrame) -> str:
    """Create markdown-formatted category lists from the blog posts data.

    :param df: DataFrame containing blog post data
    :returns: Markdown-formatted category lists as string
    """
    # Get unique categories
    all_categories = set()
    for categories_str in df["categories"]:
        categories = eval(categories_str)
        all_categories.update(categories)

    # Define category name mappings (same as in format_categories)
    category_mappings = {
        "DS_PRACTICE_LEADERSHIP": "Data Science Practice & Leadership",
        "DS_TOOLING": "Data Science Tooling",
        "BIOLOGY_CHEMISTRY": "Biology & Chemistry",
        "LLMS": "LLMs",
        "CAREER_ADVICE": "Career Advice",
    }

    # Sort categories by their display names
    sorted_categories = sorted(all_categories, key=lambda x: category_mappings[x])

    # Create markdown output
    output = ""
    for category in sorted_categories:
        display_name = category_mappings[category]
        output += f"\n### {display_name}\n\n"

        # Filter posts for this category and sort by date
        category_posts = df[df["categories"].apply(lambda x: category in eval(x))]
        category_posts = category_posts.sort_values("date", ascending=True)

        # Add bullet points for each post
        for _, row in category_posts.iterrows():
            date = row["date"]
            title = row["title"]
            url = row["url"]
            output += f"- [{title} ({date})]({url})\n"

    return output


def main(
    format: str = typer.Option("table", help="Output format: 'table' or 'categories'"),
):
    """Generate the markdown output in specified format.

    :param format: Output format - either 'table' or 'categories'
    """
    logger.info("Loading blog posts data...")
    df = load_blog_posts()

    logger.info(f"Generating markdown {format}...")
    if format == "table":
        output = create_markdown_table(df)
    elif format == "categories":
        output = create_category_lists(df)
    else:
        raise ValueError("Format must be either 'table' or 'categories'")

    # Print to stdout
    print(output)


if __name__ == "__main__":
    typer.run(main)
