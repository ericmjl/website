#!/usr/bin/env python3
"""Resize images within the repository."""


from PIL import Image
from pyprojroot import here


def resize_image(image_path, base_width):
    with Image.open(image_path) as img:
        if img.size[0] > base_width:
            w_percent = base_width / float(img.size[0])
            h_size = int(float(img.size[1]) * float(w_percent))
            img = img.resize((base_width, h_size), Image.LANCZOS)
            img.save(image_path)
            return True
    return False


def resize_logos_in_tree(root_dir, logo_name, max_width):
    resized_any = False
    for path in root_dir.rglob(logo_name):
        if resize_image(path, max_width):
            print(f"Resized: {path}")
            resized_any = True
    return resized_any


if __name__ == "__main__":
    root_directory = here()
    logo_filename = "logo.png"
    maximum_width = 600

    if resize_logos_in_tree(root_directory, logo_filename, maximum_width):
        print("Some logos were resized. Commit rejected.")
        exit(1)
    else:
        print("All logos are of the maximum width. Commit accepted.")
        exit(0)
