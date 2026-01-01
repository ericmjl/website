# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "lektor",
# ]
# ///
"""Build a pre-built lunr.js search index from all site content.

This script reads all content pages from the Lektor content directory,
extracts searchable fields (title, summary, tags, body), and generates:
1. A documents JSON file with page metadata for displaying results
2. A pre-built lunr.js index for fast client-side search

Usage:
    pixi run python scripts/build_search_index.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def parse_lr_file(filepath: Path) -> dict | None:
    """Parse a Lektor .lr file and extract fields."""
    content = filepath.read_text(encoding="utf-8")

    # Split by field separator
    parts = re.split(r"\n---\n", content)

    fields = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Find the field name and value
        match = re.match(r"^(\w+):\s*", part)
        if match:
            field_name = match.group(1)
            field_value = part[match.end() :].strip()
            fields[field_name] = field_value

    return fields


def strip_markdown(text: str) -> str:
    """Remove markdown formatting for plain text search."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Remove images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove headers
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Remove blockquotes
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_url_for_page(page_path: Path, content_dir: Path, fields: dict) -> str:
    """Generate the URL for a content page."""
    # Get the relative path from content_dir
    rel_path = page_path.parent.relative_to(content_dir)
    parts = list(rel_path.parts)

    if not parts:
        # Root content (homepage)
        return "/"

    # Special handling for blog posts with date-based URLs
    if parts[0] == "blog" and len(parts) > 1:
        slug = parts[-1]
        pub_date = fields.get("pub_date", "")
        if pub_date:
            try:
                year, month, day = pub_date.split("-")
                return f"/blog/{year}/{int(month)}/{int(day)}/{slug}/"
            except ValueError:
                pass
        return f"/blog/{slug}/"

    # For all other pages, use the directory structure
    return "/" + "/".join(parts) + "/"


def get_page_type(page_path: Path, content_dir: Path) -> str:
    """Determine the type/section of a page."""
    rel_path = page_path.parent.relative_to(content_dir)
    parts = list(rel_path.parts)

    if not parts:
        return "home"

    return parts[0]


def get_all_pages(content_dir: Path) -> list[dict]:
    """Read all content pages and extract searchable content."""
    pages = []

    # Directories to skip (non-content directories)
    skip_dirs = {"blog-tags", ".lektor"}

    # Find all contents.lr files recursively
    for contents_file in content_dir.rglob("contents.lr"):
        # Skip certain directories
        rel_path = contents_file.parent.relative_to(content_dir)
        if rel_path.parts and rel_path.parts[0] in skip_dirs:
            continue

        fields = parse_lr_file(contents_file)
        if not fields:
            continue

        # Skip if no title (probably not a real page)
        title = fields.get("title", "")
        if not title:
            # Try to get title from the directory name as fallback
            if contents_file.parent != content_dir:
                title = contents_file.parent.name.replace("-", " ").title()
            else:
                continue

        # Generate unique ID from path
        rel_path = contents_file.parent.relative_to(content_dir)
        page_id = str(rel_path).replace("/", "-") if str(rel_path) != "." else "home"

        # Get URL
        url = get_url_for_page(contents_file, content_dir, fields)

        # Get page type/section
        page_type = get_page_type(contents_file, content_dir)

        # Extract and clean body text for searching
        body = fields.get("body", "")
        body_text = strip_markdown(body)
        # Truncate body for index size (first 5000 chars should be enough)
        body_text = body_text[:5000]

        # Parse tags (if present)
        tags_raw = fields.get("tags", "")
        tags = [t.strip() for t in tags_raw.split("\n") if t.strip()]

        # Get summary or description
        summary = fields.get("summary", "") or fields.get("description", "")
        summary = strip_markdown(summary)

        # Get pub_date if available
        pub_date = fields.get("pub_date", "")

        page = {
            "id": page_id,
            "url": url,
            "title": title,
            "summary": summary,
            "body": body_text,
            "tags": tags,
            "pub_date": pub_date,
            "type": page_type,
        }
        pages.append(page)

    # Sort: blog posts by pub_date descending, then other pages alphabetically
    def sort_key(p):
        if p["type"] == "blog" and p.get("pub_date"):
            # Blog posts sorted by date (newest first)
            return (0, p["pub_date"], "")
        else:
            # Other pages sorted alphabetically by title
            return (1, "", p["title"].lower())

    pages.sort(key=sort_key, reverse=True)

    # Re-sort so blog posts are newest first but other content comes after
    blog_posts = [p for p in pages if p["type"] == "blog"]
    other_pages = [p for p in pages if p["type"] != "blog"]
    blog_posts.sort(key=lambda p: p.get("pub_date", ""), reverse=True)
    other_pages.sort(key=lambda p: p["title"].lower())

    return blog_posts + other_pages


def build_lunr_index(pages: list[dict], output_dir: Path) -> None:
    """Build the lunr.js index using Node.js."""
    # Create a documents file for the browser (without full body text)
    documents = []
    for page in pages:
        documents.append(
            {
                "id": page["id"],
                "url": page["url"],
                "title": page["title"],
                "summary": page["summary"],
                "tags": page["tags"],
                "pub_date": page["pub_date"],
                "type": page["type"],
            }
        )

    # Write documents JSON
    docs_path = output_dir / "search-documents.json"
    docs_path.write_text(json.dumps(documents, indent=2), encoding="utf-8")
    print(f"Written {len(documents)} documents to {docs_path}")

    # Create index data for lunr (with body for indexing)
    index_data = []
    for page in pages:
        index_data.append(
            {
                "id": page["id"],
                "title": page["title"],
                "summary": page["summary"],
                "body": page["body"],
                "tags": " ".join(page["tags"]),
                "type": page["type"],
            }
        )

    # Write index data for Node.js script
    index_data_path = output_dir / "search-index-data.json"
    index_data_path.write_text(json.dumps(index_data), encoding="utf-8")

    # Build the index using Node.js
    build_script = """
const lunr = require('lunr');
const fs = require('fs');

const data = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));

const idx = lunr(function () {
    this.ref('id');
    this.field('title', { boost: 10 });
    this.field('summary', { boost: 5 });
    this.field('tags', { boost: 3 });
    this.field('type', { boost: 2 });
    this.field('body');

    data.forEach(function (doc) {
        this.add(doc);
    }, this);
});

console.log(JSON.stringify(idx));
"""

    # Write the build script
    build_script_path = output_dir / "build-lunr-index.js"
    build_script_path.write_text(build_script, encoding="utf-8")

    # Run the script with Node.js
    try:
        result = subprocess.run(
            ["node", str(build_script_path), str(index_data_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        index_json = result.stdout

        # Write the pre-built index
        index_path = output_dir / "search-index.json"
        index_path.write_text(index_json, encoding="utf-8")
        print(f"Written pre-built lunr index to {index_path}")

        # Clean up temporary files
        build_script_path.unlink()
        index_data_path.unlink()

    except subprocess.CalledProcessError as e:
        print(f"Error building lunr index: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Node.js not found. Please install Node.js and lunr: npm install -g lunr",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    # Find the project root (where this script lives in scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    content_dir = project_root / "content"
    output_dir = project_root / "assets" / "static" / "js"

    if not content_dir.exists():
        print(f"Content directory not found: {content_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading all content pages...")
    pages = get_all_pages(content_dir)

    # Count by type
    type_counts = {}
    for page in pages:
        t = page["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"Found {len(pages)} pages:")
    for t, count in sorted(type_counts.items()):
        print(f"  - {t}: {count}")

    print("Building search index...")
    build_lunr_index(pages, output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
