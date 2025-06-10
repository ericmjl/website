# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyyaml",
#   "pyprojroot",
#   "tqdm",
# ]
# ///

import re
from pathlib import Path
from typing import Any, Dict

import yaml
from pyprojroot import here
from tqdm import tqdm


def parse_lr_content(content: str) -> Dict[str, Any]:
    """Parse .lr content into a dictionary with sections.

    Args:
        content: String containing the .lr file content

    Returns:
        Dictionary containing the parsed content with keys for each section
    """
    # Split content into sections using --- as separator
    # We use a simpler pattern that matches --- at the start of a line
    sections = re.split(r"\n---\n", content)

    # Remove empty sections and strip whitespace
    sections = [s.strip() for s in sections if s.strip()]

    # Parse each section
    result = {}
    for section in sections:
        # Split section into header and content
        if ":" in section:
            header, content = section.split(":", 1)
            header = header.strip()
            content = content.strip()

            # Handle multi-line content
            if "\n" in content:
                # If content starts with newline, remove it
                if content.startswith("\n"):
                    content = content[1:]
                # If content ends with newline, remove it
                if content.endswith("\n"):
                    content = content[:-1]

            result[header] = content
        else:
            # If no header found, treat as body content
            result["body"] = section

    return result


def parse_lr_file(file_path: Path) -> Dict[str, Any]:
    """Parse a .lr file into a dictionary with sections.

    Args:
        file_path: Path to the .lr file

    Returns:
        Dictionary containing the parsed content with keys for each section
    """
    content = file_path.read_text(encoding="utf-8")
    result = parse_lr_content(content)
    result["file_path"] = str(file_path)
    return result


def convert_lr_to_yaml(input_path: Path, output_path: Path) -> None:
    """Convert all .lr files in a directory to YAML format.

    Args:
        input_path: Path to the directory containing .lr files
        output_path: Path to save the YAML files
    """
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all .lr files
    lr_files = list(input_path.rglob("*.lr"))

    for lr_file in tqdm(lr_files, desc="Converting .lr files to YAML"):
        try:
            # Parse the .lr file
            parsed_content = parse_lr_file(lr_file)

            # Create output filename
            relative_path = lr_file.relative_to(input_path)
            output_file = output_path / f"{relative_path.stem}.yaml"

            # Ensure parent directories exist
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to YAML file
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(parsed_content, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Error processing {lr_file}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    blog_path = here() / "content" / "blog"
    output_path = here() / "content" / "blog_yaml"

    convert_lr_to_yaml(blog_path, output_path)
