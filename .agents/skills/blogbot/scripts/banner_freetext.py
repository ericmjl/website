# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "llamabot",
#   "pydantic",
#   "rich",
#   "pillow",
#   "openai",
#   "requests",
#   "python-dotenv",
# ]
# ///
"""Generate a DALL-E banner for the Substack jobs post from raw text (no slug).

Adapts blogbot's banner.py to take free text instead of a contents.lr slug,
and writes the image to ~/Downloads.
"""

import sys
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from banner import DallEImagePrompt, dalle_sysprompt  # noqa: E402
from llamabot import StructuredBot  # noqa: E402

load_dotenv(Path.home() / "github/website/.env")

import os  # noqa: E402
import sys  # noqa: E402

os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
sys.path.insert(0, str(Path.home() / "github/website"))

from apis.blogbot.images import generate_banner_image_bytes  # noqa: E402


def save_image_as_webp(image_bytes: bytes, output_path: Path) -> None:
    img = Image.open(BytesIO(image_bytes))
    if img.mode in ("RGBA", "LA", "P"):
        rgb = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        rgb.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = rgb
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img.save(output_path, "WEBP", quality=95)


def main():
    text = Path(sys.argv[1]).read_text()
    bot = StructuredBot(
        dalle_sysprompt(), model="gpt-4.1", pydantic_model=DallEImagePrompt
    )
    prompt = bot(text).content
    print("PROMPT:", prompt[:300])
    image_bytes = generate_banner_image_bytes(prompt)
    out = Path.home() / "Downloads" / "substack-jobs-banner.webp"
    save_image_as_webp(image_bytes, out)
    print("SAVED:", out)


if __name__ == "__main__":
    main()
