#!/usr/bin/env python3
"""Resize images within the repository."""

import os
from pathlib import Path

from PIL import Image
from pyprojroot import here


def resize_image(image_path: Path, base_width: int) -> bool:
    with Image.open(image_path) as img:
        if img.size[0] > base_width:
            w_percent = base_width / float(img.size[0])
            h_size = int(float(img.size[1]) * float(w_percent))
            img = img.resize((base_width, h_size), Image.LANCZOS)
            webp_path = image_path.with_suffix(".webp")
            img.save(webp_path, format="WEBP")
            os.remove(image_path)
            return True
    return False


def resize_logos_in_tree(root_dir: Path, logo_name: str, max_width: int) -> bool:
    resized_any = False
    for path in root_dir.rglob(logo_name):
        if resize_image(path, max_width):
            print(f"Resized and converted to WEBP: {path}")
            resized_any = True
    return resized_any


if __name__ == "__main__":
    root_directory = here()  # This returns a Path object
    logo_filename = "logo.png"
    maximum_width = 1024  # Updated to 1024px

    if resize_logos_in_tree(root_directory, logo_filename, maximum_width):
        print("Some logos were resized. Commit rejected.")
        exit(1)
    else:
        print("All logos are of the maximum width. Commit accepted.")
        exit(0)
