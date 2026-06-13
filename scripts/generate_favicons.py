# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pillow",
# ]
# ///
"""Generate raster favicon assets for the website.

Renders the terminal-style ">_" mark (matching assets/static/favicon.svg)
to a multi-size favicon.ico and an apple-touch-icon WebP.

Usage:
    uv run scripts/generate_favicons.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = "#222225"
FG = "#62c4ff"
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"


def make_icon(size: int, rounded: bool = True) -> Image.Image:
    """Draw the ">_" mark on a dark square at the given pixel size."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = int(size * 0.1875) if rounded else 0
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=BG)

    font = ImageFont.truetype(FONT_PATH, int(size * 0.5))
    text = ">_"
    stroke = max(1, size // 48)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    x = (size - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (size - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=FG, stroke_width=stroke, stroke_fill=FG)
    return img


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    static = repo / "assets" / "static"

    favicon = make_icon(48)
    favicon.save(repo / "assets" / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    print(f"Wrote {repo / 'assets' / 'favicon.ico'}")

    # iOS applies its own corner mask, so keep the touch icon square.
    make_icon(180, rounded=False).save(static / "apple-touch-icon.webp")
    print(f"Wrote {static / 'apple-touch-icon.webp'}")


if __name__ == "__main__":
    main()
