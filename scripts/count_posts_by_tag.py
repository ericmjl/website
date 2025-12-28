# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "lektor",
# ]
# ///
"""Count the number of blog posts tagged with a given tag.

Usage:
    pixi run python scripts/count_posts_by_tag.py <tag>
    pixi run python scripts/count_posts_by_tag.py --all

Examples:
    pixi run python scripts/count_posts_by_tag.py python
    pixi run python scripts/count_posts_by_tag.py --all
"""

import argparse
from collections import Counter
from pathlib import Path

from lektor.environment import Environment
from lektor.project import Project


def get_pad():
    """Get a Lektor pad for the project."""
    project = Project.discover(Path(__file__).parent.parent)
    env = Environment(project)
    return env.new_pad()


def count_posts_by_tag(tag: str) -> tuple[int, list[str]]:
    """Count blog posts with a given tag.

    Returns a tuple of (count, list of post titles).
    """
    pad = get_pad()
    blog = pad.get("/blog")

    matching_posts = []
    for post in blog.children:
        tags = post["tags"] or []
        if tag.lower() in [t.lower() for t in tags]:
            matching_posts.append(post["title"])

    return len(matching_posts), matching_posts


def count_all_tags() -> Counter:
    """Count occurrences of all tags across blog posts."""
    pad = get_pad()
    blog = pad.get("/blog")

    tag_counts = Counter()
    for post in blog.children:
        tags = post["tags"] or []
        for tag in tags:
            tag_counts[tag.lower()] += 1

    return tag_counts


def main():
    parser = argparse.ArgumentParser(
        description="Count blog posts by tag using Lektor's parser."
    )
    parser.add_argument("tag", nargs="?", help="Tag to search for")
    parser.add_argument("--all", action="store_true", help="Show counts for all tags")
    parser.add_argument(
        "--top", type=int, default=20, help="Number of top tags to show (default: 20)"
    )
    parser.add_argument("--list", action="store_true", help="List matching post titles")

    args = parser.parse_args()

    if args.all:
        tag_counts = count_all_tags()
        print(f"Tag counts across {sum(tag_counts.values())} total tag usages:\n")
        for tag, count in tag_counts.most_common(args.top):
            print(f"  {tag}: {count}")
        if len(tag_counts) > args.top:
            print(f"\n  ... and {len(tag_counts) - args.top} more tags")
    elif args.tag:
        count, posts = count_posts_by_tag(args.tag)
        print(f"Posts tagged with '{args.tag}': {count}")
        if args.list and posts:
            print("\nMatching posts:")
            for title in sorted(posts):
                print(f"  - {title}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
