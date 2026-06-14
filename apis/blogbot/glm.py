"""GLM-backed structured generation via the Z.ai coding-plan endpoint.

Used by the blogbot FastAPI app for text generation (LinkedIn, BlueSky,
Substack, summary, tags). Banner/image generation stays on OpenAI (DALL-E),
which still requires ``OPENAI_API_KEY``.
"""

import json
import os
import re

from litellm import completion
from loguru import logger
from pydantic import BaseModel

from .prompts import socialbot_sysprompt

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model reply, tolerating code fences."""
    match = _FENCE_RE.search(text)
    candidate = match.group(1) if match else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]
    return json.loads(candidate)


def generate_structured(
    user_prompt: str,
    model_cls: type[BaseModel],
    *,
    num_attempts: int = 8,
) -> BaseModel:
    """Generate a pydantic-validated object from GLM via litellm.

    The JSON schema is injected as plain text in the prompt (no
    ``response_format`` / tool use, which the Z.ai coding-plan endpoint does not
    handle for these custom models). Code fences are stripped from the reply and
    the result is validated against ``model_cls``, retrying on failure.
    """
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise RuntimeError("ZAI_API_KEY is not set. Add it to .env or the environment.")

    model = os.environ.get("BLOGBOT_MODEL", "anthropic/glm-5.2")
    api_base = os.environ.get("BLOGBOT_API_BASE", "https://api.z.ai/api/anthropic")

    schema = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    system_text = (
        f"{socialbot_sysprompt().content}\n\n"
        "Return ONLY a raw JSON object (no markdown, no code fences, no prose) "
        f"matching this JSON schema:\n{schema}"
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(num_attempts):
        logger.info(
            "Generating {} via {} (attempt {}/{})",
            model_cls.__name__,
            model,
            attempt + 1,
            num_attempts,
        )
        response = completion(
            model=model,
            api_base=api_base,
            api_key=api_key,
            messages=messages,
            drop_params=True,
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        try:
            return model_cls.model_validate(_extract_json(content))
        except Exception as err:
            last_error = err
            logger.warning("Validation failed ({}): {}", model_cls.__name__, err)
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That did not parse / validate: {err}. "
                        "Return ONLY the corrected raw JSON object."
                    ),
                }
            )
    raise RuntimeError(
        f"Failed to produce a valid {model_cls.__name__} after "
        f"{num_attempts} attempts: {last_error}"
    )
