"""Generate blog banner images via the OpenAI Images API."""

import base64

import requests
from openai import OpenAI

BANNER_MODEL = "gpt-image-2"
BANNER_SIZE = "1536x1024"
BANNER_QUALITY = "medium"

TEXT_FREE_SUFFIX = (
    " No text, no letters, no words, no numbers, no labels, no captions, "
    "no signage, no typography, no watermarks anywhere in the image."
)


def generate_banner_image_bytes(prompt: str) -> bytes:
    """Generate a banner image and return raw image bytes.

    GPT Image models return base64-encoded data in ``b64_json``; DALL-E may
    return a temporary ``url`` instead.

    :param prompt: DALL-E / GPT Image prompt text.
    :return: Raw image bytes (typically PNG).
    :raises RuntimeError: If the API response contains no image data.
    """
    client = OpenAI()
    full_prompt = prompt.rstrip() + TEXT_FREE_SUFFIX
    response = client.images.generate(
        model=BANNER_MODEL,
        prompt=full_prompt,
        size=BANNER_SIZE,
        quality=BANNER_QUALITY,
        n=1,
    )
    item = response.data[0]
    if item.b64_json:
        return base64.b64decode(item.b64_json)
    if item.url:
        img_response = requests.get(item.url, timeout=120)
        img_response.raise_for_status()
        return img_response.content
    raise RuntimeError("No image data in OpenAI Images API response")
