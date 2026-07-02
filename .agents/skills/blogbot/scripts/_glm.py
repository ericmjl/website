"""Shared GLM-5.2 structured generation helper for blogbot scripts.

Calls litellm.completion() against Z.ai's Anthropic-compatible coding-plan
endpoint (model anthropic/glm-5.2) and falls back to a local oMLX server.
The JSON schema is injected as plain text in the prompt; the model's reply is
code-fence-stripped and revalidated against the given pydantic model, retrying
on failure.

llamabot's StructuredBot is intentionally NOT used: it rejects glm-5.2 via a
client-side capability guard (model name not recognized), which is why the GLM
scripts call litellm directly. The same approach is used by generate_social.py.

A Z.ai coding-plan key only works via the Anthropic-compatible endpoint
(https://api.z.ai/api/anthropic); the zai/ PaaS endpoint
(api.z.ai/api/paas/v4) returns "insufficient balance".
"""

import json
import os
import re
from pathlib import Path

from litellm import completion
from pydantic import BaseModel

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response, tolerating code fences."""
    match = _FENCE_RE.search(text)
    candidate = match.group(1) if match else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]
    return json.loads(candidate)


def _omlx_fallback_key() -> str:
    key = os.environ.get("BLOGBOT_API_KEY")
    if key:
        return key
    cfg_path = Path.home() / ".omlx" / "settings.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f).get("auth", {}).get("api_key", "")
    return ""


def generate_structured(
    user_prompt: str,
    model_cls: type[BaseModel],
    system_prompt: str,
    *,
    num_attempts: int = 8,
) -> BaseModel:
    """Generate a pydantic-validated object from GLM-5.2 via litellm.

    Control flow: a successful API call is ALWAYS followed by JSON extraction +
    validation + return. Only a validation failure, or both providers failing,
    triggers a retry. (This corrects the generate_social.generate_structured
    bug, where unconditional ``break`` statements exit the loop before the
    validation/return lines, making the success path unreachable.)
    """
    api_key = os.environ.get("ZAI_API_KEY")
    api_base = os.environ.get("BLOGBOT_API_BASE", "https://api.z.ai/api/anthropic")
    model = os.environ.get("BLOGBOT_MODEL", "anthropic/glm-5.2")

    fallback_api_base = "http://localhost:8426/v1"
    fallback_model = "openai/Qwen3.36-35B-A3B-8bit"
    fallback_api_key = _omlx_fallback_key()

    schema = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    system_text = (
        f"{system_prompt}\n\n"
        "Return ONLY a raw JSON object (no markdown, no code fences, no prose) "
        f"matching this JSON schema:\n{schema}"
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    for _ in range(num_attempts):
        response = None
        try:
            response = completion(
                model=model,
                api_base=api_base,
                api_key=api_key,
                messages=messages,
                drop_params=True,
                temperature=0,
            )
        except Exception as primary_err:
            try:
                response = completion(
                    model=fallback_model,
                    api_base=fallback_api_base,
                    api_key=fallback_api_key,
                    messages=messages,
                    drop_params=True,
                    temperature=0,
                )
            except Exception as fallback_err:
                last_error = fallback_err
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Both providers failed (primary: {primary_err}; "
                            f"fallback: {fallback_err})."
                        ),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Return ONLY the corrected raw JSON object.",
                    }
                )
                continue

        content = response.choices[0].message.content or ""
        try:
            return model_cls.model_validate(_extract_json(content))
        except Exception as err:
            last_error = err
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
            continue

    raise RuntimeError(
        f"Failed to produce a valid {model_cls.__name__} "
        f"after {num_attempts} attempts: {last_error}"
    )
