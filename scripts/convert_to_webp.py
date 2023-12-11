from pathlib import Path

from PIL import Image

# Glob all of the `logo.png` images under the `content` directory, recursively.
for path in Path("content").rglob("logo.png"):
    # Open the image using Pillow.
    image = Image.open(path)
    # Convert the image to WebP format.
    image.save(path.with_suffix(".webp"))
