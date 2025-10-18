#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pathlib",
#     "pyyaml",
#     "click",
# ]
# ///

"""
Convert Lektor content.lr files to Markdown with YAML frontmatter.

This script processes all content.lr files in the blog directory and converts them
to index.md files with proper YAML frontmatter for static site generation.
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List

import click
import yaml


def parse_flowblocks(body_content: str) -> tuple[str, List[Dict[str, Any]]]:
    """Parse flowblocks from body content and return (clean_body, flowblocks)."""
    import re

    flowblocks = []

    # Find all flowblock patterns: #### blockname ####
    flowblock_pattern = r"####\s*(\w+)\s*####"

    # Split content by flowblock markers
    parts = re.split(flowblock_pattern, body_content)

    # First part is content before any flowblocks
    clean_body = parts[0].strip() if parts else ""

    # Process flowblock parts (name, content, name, content, ...)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            block_type = parts[i].strip()
            block_content = parts[i + 1].strip()

            # Parse block content into fields
            block_data = {"type": block_type}

            # Split by ---- separators for multiple field groups
            field_groups = block_content.split("\n----\n")

            for group in field_groups:
                group = group.strip()
                if not group:
                    continue

                # Parse fields in this group
                lines = group.split("\n")
                current_field = None
                current_content = []

                for line in lines:
                    if (
                        ":" in line
                        and not line.startswith(" ")
                        and not line.startswith("\t")
                    ):
                        # Save previous field
                        if current_field:
                            block_data[current_field] = "\n".join(
                                current_content
                            ).strip()

                        # Start new field
                        field_name, field_value = line.split(":", 1)
                        current_field = field_name.strip()
                        current_content = (
                            [field_value.strip()] if field_value.strip() else []
                        )
                    else:
                        # Continue current field content
                        if current_field:
                            current_content.append(line)

                # Don't forget the last field
                if current_field:
                    block_data[current_field] = "\n".join(current_content).strip()

            flowblocks.append(block_data)

    return clean_body, flowblocks


def parse_lektor_content(content: str) -> Dict[str, Any]:
    """Parse Lektor content.lr format into structured data."""
    sections = {}

    # Split content by --- separators on their own lines
    lines = content.split("\n")
    parts = []
    current_part = []

    for line in lines:
        if line.strip() == "---":
            if current_part:
                parts.append("\n".join(current_part))
                current_part = []
        else:
            current_part.append(line)

    # Don't forget the last part
    if current_part:
        parts.append("\n".join(current_part))

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Find the first line that contains ':'
        lines = part.split("\n")
        if not lines:
            continue

        first_line = lines[0].strip()
        if ":" in first_line:
            # Split on first colon to get field name and value
            field_name, field_value = first_line.split(":", 1)
            field_name = field_name.strip()
            field_value = field_value.strip()

            if field_value:
                # Single line value (e.g., "title: CuPy")
                sections[field_name] = field_value
            else:
                # Multi-line value (e.g., "body:" followed by content)
                section_content = "\n".join(lines[1:]).strip()
                sections[field_name] = section_content
        else:
            # This might be content without a header, skip for now
            continue

    return sections


def extract_tags(tags_content: str) -> List[str]:
    """Extract tags from the tags section."""
    if not tags_content:
        return []

    # Split by newlines and filter out empty lines
    tags = [tag.strip() for tag in tags_content.split("\n") if tag.strip()]
    return tags


def convert_to_yaml_frontmatter(sections: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    """Convert Lektor sections to YAML frontmatter and body content."""
    frontmatter = {}
    body_content = ""

    # Map Lektor fields to frontmatter - keep original field names for consistency
    field_mapping = {
        "title": "title",
        "author": "author",
        "pub_date": "date",
        "twitter_handle": "twitter_handle",
        "summary": "summary",
        "_model": "layout",  # Map _model to layout for static site generators
    }

    for lektor_field, yaml_field in field_mapping.items():
        if lektor_field in sections and sections[lektor_field]:
            frontmatter[yaml_field] = sections[lektor_field]

    # Handle tags specially
    if "tags" in sections:
        tags = extract_tags(sections["tags"])
        if tags:
            frontmatter["tags"] = tags

    # Handle other frontmatter fields
    other_fields = ["category", "visible", "lead", "sort_key"]
    for field in other_fields:
        if field in sections and sections[field]:
            frontmatter[field] = sections[field]

    # Handle body content with flowblocks
    if "body" in sections:
        raw_body = sections["body"]
        clean_body, flowblocks = parse_flowblocks(raw_body)

        # Extract description flowblocks and add to body content
        description_texts = []
        filtered_flowblocks = []

        for flowblock in flowblocks:
            if flowblock.get("type") == "description" and "text" in flowblock:
                description_texts.append(flowblock["text"])
            else:
                filtered_flowblocks.append(flowblock)

        # Add non-description flowblocks to frontmatter if they exist
        if filtered_flowblocks:
            frontmatter["flowblocks"] = filtered_flowblocks

        # Combine clean body with description texts
        body_parts = [clean_body] if clean_body else []
        body_parts.extend(description_texts)
        body_content = "\n\n".join(part for part in body_parts if part)

    return frontmatter, body_content


def generate_date_path(pub_date: str, slug: str) -> str:
    """Generate date-based path from publication date."""
    if not pub_date:
        return slug

    # Parse date (assuming YYYY-MM-DD format)
    try:
        year, month, day = pub_date.split("-")
        return f"{year}/{month}/{day}/{slug}"
    except (ValueError, AttributeError):
        return slug


def copy_attachments(source_dir: Path, output_path: Path) -> None:
    """Copy all attachment files (non-contents.lr files) to output directory."""
    # Get all files in source directory except contents.lr
    attachment_files = [
        f for f in source_dir.iterdir() if f.is_file() and f.name != "contents.lr"
    ]

    for attachment in attachment_files:
        dest_file = output_path / attachment.name
        shutil.copy2(attachment, dest_file)
        print(f"  Copied attachment: {attachment.name}")


def process_content_file(
    content_path: Path, output_dir: Path, content_base_dir: Path
) -> None:
    """Process a single content.lr file."""
    # Read the content.lr file
    with open(content_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse Lektor format
    sections = parse_lektor_content(content)

    # Convert to frontmatter and body
    frontmatter, body = convert_to_yaml_frontmatter(sections)

    # Calculate relative path from content base directory
    content_dir = content_path.parent
    relative_path = content_dir.relative_to(content_base_dir)

    # Extract slug from directory name
    slug = content_dir.name

    # Add slug to frontmatter
    frontmatter["slug"] = slug

    # Handle different content types differently
    if len(relative_path.parts) > 0 and relative_path.parts[0] == "blog":
        # For blog posts, use date-based paths if available
        if "date" in frontmatter:
            date_path = generate_date_path(frontmatter["date"], slug)
            frontmatter["url"] = f"/blog/{date_path}/"
            output_path = output_dir / "blog" / date_path
        else:
            frontmatter["url"] = f"/blog/{slug}/"
            output_path = output_dir / "blog" / slug
    elif len(relative_path.parts) == 0:
        # This is a root-level contents.lr file (like content/contents.lr)
        frontmatter["url"] = "/"
        output_path = output_dir
    else:
        # For other content types, mirror the directory structure
        frontmatter["url"] = f"/{relative_path}/"
        output_path = output_dir / relative_path

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy attachments to output directory
    copy_attachments(content_dir, output_path)

    # Write index.md file
    index_path = output_path / "index.md"

    with open(index_path, "w", encoding="utf-8") as f:
        # Write YAML frontmatter
        f.write("---\n")
        yaml.dump(frontmatter, f, default_flow_style=False, allow_unicode=True)
        f.write("---\n\n")

        # Write body content
        f.write(body)

    print(f"Converted: {content_path} -> {index_path}")


@click.command()
@click.option(
    "--content-dir",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=Path("content"),
    help="Directory containing content.lr files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("converted_content"),
    help="Output directory for converted files",
)
@click.option(
    "--pattern",
    "-p",
    default="**/contents.lr",
    help="File pattern to match (default: **/contents.lr)",
)
def main(content_dir: Path, output_dir: Path, pattern: str):
    """Convert Lektor content.lr files to Markdown with YAML frontmatter."""

    # Find all content.lr files
    content_files = list(content_dir.glob(pattern))

    if not content_files:
        print(f"No files found matching pattern '{pattern}' in {content_dir}")
        return

    print(f"Found {len(content_files)} content files to convert")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Process each file
    for content_file in content_files:
        try:
            process_content_file(content_file, output_dir, content_dir)
        except Exception as e:
            print(f"Error processing {content_file}: {e}")

    print(f"\nConversion complete! Files written to {output_dir}")


if __name__ == "__main__":
    main()
