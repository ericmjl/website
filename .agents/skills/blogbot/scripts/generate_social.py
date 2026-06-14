# /// script
# requires-python = ">=3.11"
# dependencies = ["llamabot", "pydantic", "python-dotenv"]
# ///
# ruff: noqa: E501
"""
Generate LinkedIn + BlueSky posts for a blog post, using the same
battle-tested prompts/models as the apis/blogbot FastAPI app, but routed
through Z.ai's GLM models instead of OpenAI.

Outputs a single JSON object on stdout:
    {"slug", "title", "url", "linkedin", "bluesky"}

Usage:
    uv run generate_social.py <blog_slug>

Configuration (read from .env or the environment):
    ZAI_API_KEY       (required) your Z.ai API key (coding plan or PaaS)
    BLOGBOT_MODEL     (optional) litellm model string, defaults to anthropic/glm-5.2
    BLOGBOT_API_BASE  (optional) defaults to https://api.z.ai/api/anthropic
                      (the Anthropic-compatible coding-plan endpoint)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from litellm import completion
from llamabot.prompt_manager import prompt
from pydantic import BaseModel, Field, model_validator

load_dotenv()


# ---------------------------------------------------------------------------
# Prompts (duplicated verbatim from apis/blogbot/prompts.py)
# ---------------------------------------------------------------------------


@prompt(role="system")
def socialbot_sysprompt():
    """You are an expert blogger and social media manager.

    You are given a blog post for which to write a social media post.

    Notes:

    - First person, humble, and inviting.
    - Keep it short and concise.
    - Always include the URL of the blog post in the social media post.
    """


@prompt(role="user")
def compose_linkedin_post(text, url):
    """This is a blog post that I just wrote.

        {{ text }}

    It came from the following url: {{ url }}.

    Please compose for me a LinkedIn post that follows the provided JSON structure.
    """


@prompt(role="user")
def compose_bluesky_post(text, url):
    """This is a blog post that I just wrote:

        {{ text }}

    It came from the following url: {{ url }}.

    Please compose for me a BlueSky post
    that entices my followers on BlueSky to read it.
    I usually like to open off with a question that the post answers.
    Ensure that there is a call to action to interact with the post after reading it,
    such as reposting, commenting, or sharing it with others.
    Include hashtags inline with the BlueSky post.
    Hashtags should be all lowercase.
    DO NOT include the URL in your response - it will be added automatically at the end.
    Also ensure that it is written in first-person, humble, and inviting tone.
    """


# ---------------------------------------------------------------------------
# Models (duplicated verbatim from apis/blogbot/models.py)
# ---------------------------------------------------------------------------


class LinkedInPostSection(BaseModel):
    content: str = Field(
        ..., description="The content of a section in the LinkedIn post"
    )
    section_type: Optional[str] = Field(
        None, description="Type: story, insight, lesson, example"
    )


class LinkedInHook(BaseModel):
    line1: str = Field(
        ...,
        description=(
            "Context-Lean Setup: Start with as little context as possible. "
            "Jump straight into the action or core message. Avoid long-winded intros. "
            "Should be 60 characters or less for maximum impact."
        ),
    )
    line2: str = Field(
        ...,
        description=(
            "Scroll-Stop Interjection: Use a bold, surprising, or emotionally charged statement "
            "that interrupts scrolling. Could be a shocking fact, strong opinion, or unexpected insight. "
            "Jolt the viewer out of autopilot and make them pay attention."
        ),
    )
    line3: str = Field(
        ...,
        description=(
            "Curiosity Gap: Create a gap between what the viewer knows and wants to know. "
            "Tease an answer, reveal, or transformation without giving it away. "
            "Compel the viewer to click 'see more' to satisfy their curiosity."
        ),
    )


class LinkedInAuthorityElement(BaseModel):
    story_type: str = Field(
        ..., description="Type: personal_story, lesson_learned, failure, success"
    )
    content: str = Field(..., description="The authority-building content")
    specific_example: Optional[str] = Field(
        None, description="Concrete example demonstrating expertise"
    )


class LinkedInPost(BaseModel):
    hook: LinkedInHook = Field(
        ..., description="Three-line hook following the 3-Line Hack strategy"
    )
    authority_elements: List[LinkedInAuthorityElement] = Field(
        default_factory=list,
        description="Personal stories, lessons, or insights that build authority and trust",
    )
    main_content: List[LinkedInPostSection] = Field(
        ..., description="The meat of the content, broken into digestible sections"
    )
    call_to_action: str = Field(
        ...,
        description=(
            "Call to action focused on relationship building, not just broadcasting. "
            "I usually want people to read the blog post, leave a like, comment, and share."
        ),
    )
    ending_question: str = Field(
        ...,
        description="Specific question designed to generate thoughtful comments and discussion",
    )
    hashtags: List[str] = Field(
        ...,
        max_items=5,
        description=(
            "List of relevant hashtags, including some outside main niche if valuable. "
            "They should always begin with '#'."
        ),
    )
    engagement_intent: str = Field(
        default="relationship_building",
        description="Primary intent: relationship_building, community_discussion, authority_demonstration",
    )

    @model_validator(mode="after")
    def validate_content(self):
        for hashtag in self.hashtags:
            if not hashtag.startswith("#"):
                raise ValueError(f"Hashtag '{hashtag}' must start with '#'.")
        if len(self.hook.line1) > 60:
            raise ValueError("Hook line 1 should be punchy and under 60 characters")
        for element in self.authority_elements:
            if (
                element.story_type in ["lesson_learned", "failure", "success"]
                and not element.specific_example
            ):
                raise ValueError(
                    f"Authority element of type '{element.story_type}' should include specific example"
                )
        return self

    def format_post(self) -> str:
        post_content = f"{self.hook.line1}\n{self.hook.line2}\n{self.hook.line3}\n\n"
        for element in self.authority_elements:
            post_content += f"{element.content}\n\n"
            if element.specific_example:
                post_content += f"{element.specific_example}\n\n"
        for section in self.main_content:
            post_content += f"{section.content}\n\n"
        post_content += f"{self.call_to_action}\n\n"
        post_content += f"{self.ending_question}\n\n"
        post_content += " ".join([hashtag.lower() for hashtag in self.hashtags])
        return post_content


class BlueSkyPost(BaseModel):
    strong_hook: str = Field(
        ...,
        description=(
            "Context-Lean Setup: Start with minimal context, jump straight to the core message. "
            "Use a bold, surprising, or emotionally charged statement that interrupts scrolling. "
            "Create immediate curiosity gap - tease value without revealing everything. "
            "Avoid bland intros and long-winded setups."
        ),
    )
    clear_stance: str = Field(
        ...,
        description=(
            "A direct, clear stance or unique insight. Don't be afraid to be bold or controversial. "
            "Avoid too much nuance - clarity and conviction get noticed."
        ),
    )
    value_delivery: str = Field(
        ...,
        description=(
            "Deliver value, emotion, or entertainment. Use specific details, numbers, or examples "
            "for credibility and impact. Should teach something, make people laugh, or evoke feeling."
        ),
    )
    call_to_action: Optional[str] = Field(
        None,
        description=(
            "Optional engaging call to action or thought-provoking line. "
            "Examples: 'Agree or disagree?' 'What's your take?' or a memorable punchline. "
            "Do not include URL!"
        ),
    )
    hashtags: List[str] = Field(
        ...,
        max_items=2,
        description="A list of hashtags (max 2) to be included in the BlueSky post",
    )
    url: str = Field(
        ..., description="The URL to click on for more information. Just the raw URL."
    )
    post_intent: str = Field(
        default="value_delivery",
        description="Primary intent: value_delivery, emotion, entertainment, controversy",
    )
    authenticity_check: bool = Field(
        default=True,
        description="Confirms this represents genuine beliefs and authentic voice",
    )

    @model_validator(mode="after")
    def validate_content(self):
        errors = []
        if len(self.hashtags) > 2:
            errors.append("BlueSky post can have a maximum of 2 hashtags.")
        for hashtag in self.hashtags:
            if not hashtag.startswith("#"):
                errors.append(f"Hashtag '{hashtag}' must start with '#.'")
        total_content = self.format_post(with_url=False)
        if len(total_content) > 283:
            errors.append(
                "Total content must be 283 characters or less (excluding URL to account for buff.ly link shortening)."
            )
        if len(total_content) < 100:
            errors.append("Total content must be at least 100 characters for impact.")
        if self.call_to_action and self.url in self.call_to_action:
            errors.append("URL should not be present in the call to action.")
        if not self.url.startswith("http://") and not self.url.startswith("https://"):
            errors.append("URL must start with 'http://' or 'https://'.")
        weak_openers = [
            "ever wondered",
            "have you ever",
            "i think that",
            "maybe we should",
        ]
        if any(opener in self.strong_hook.lower() for opener in weak_openers):
            errors.append(
                "Hook should be bold and direct, avoid weak openers like 'Ever wondered...'"
            )
        if len(self.clear_stance.split()) > 30:
            errors.append(
                "Clear stance should be concise - under 30 words for maximum impact"
            )
        if not self.authenticity_check:
            errors.append("Post must represent authentic beliefs and voice")
        if errors:
            raise ValueError(", ".join(errors))
        return self

    def format_post(self, with_url: bool = True) -> str:
        post_content = f"{self.strong_hook} {self.clear_stance} {self.value_delivery}"
        if self.call_to_action:
            post_content += f" {self.call_to_action}"
        if with_url:
            post_content += f" {self.url}"
        post_content += f" {' '.join(self.hashtags)}"
        return post_content


# ---------------------------------------------------------------------------
# Local blog post reading + URL building
# ---------------------------------------------------------------------------


def find_repo_root() -> Path:
    current = Path(__file__).parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd()


def parse_contents_lr(file_path: Path) -> dict:
    content = file_path.read_text()
    fields: dict = {}
    current_field = None
    current_value: list[str] = []

    for line in content.split("\n"):
        if line == "---":
            if current_field and current_value:
                fields[current_field] = "\n".join(current_value).strip()
            current_field = None
            current_value = []
        elif current_field is None and ":" in line:
            field_name, field_value = line.split(":", 1)
            field_name = field_name.strip()
            field_value = field_value.strip()
            if field_value:
                fields[field_name] = field_value
            else:
                current_field = field_name
        elif current_field:
            current_value.append(line)

    if current_field and current_value:
        fields[current_field] = "\n".join(current_value).strip()

    return fields


def get_local_blog_post(slug: str) -> dict:
    repo_root = find_repo_root()
    blog_path = repo_root / "content" / "blog" / slug / "contents.lr"
    if not blog_path.exists():
        raise FileNotFoundError(f"Blog post not found: {blog_path}")
    fields = parse_contents_lr(blog_path)
    return {
        "title": fields.get("title", ""),
        "body": fields.get("body", ""),
        "pub_date": fields.get("pub_date", ""),
        "slug": slug,
    }


def build_public_url(pub_date: str, slug: str) -> str:
    if not pub_date:
        return f"https://ericmjl.github.io/blog/{slug}/"
    y, m, d = pub_date.split("-")
    return f"https://ericmjl.github.io/blog/{y}/{m.lstrip('0')}/{d.lstrip('0')}/{slug}/"


# ---------------------------------------------------------------------------
# GLM-backed structured generation (calls litellm directly)
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response, tolerating code fences."""
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
    response_format / tool use, which the Z.ai coding-plan endpoint does not
    handle for these custom models). Code fences are stripped from the reply
    and the result is validated against ``model_cls``, retrying on failure.
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
    for _ in range(num_attempts):
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
        f"Failed to produce a valid {model_cls.__name__} after {num_attempts} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Please provide a blog slug."}))
        sys.exit(1)

    slug = sys.argv[1]
    try:
        post = get_local_blog_post(slug)
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)

    url = build_public_url(post["pub_date"], slug)
    body = post["body"]

    linkedin_post = generate_structured(
        compose_linkedin_post(body, url).content, LinkedInPost
    )
    linkedin_text = linkedin_post.format_post()

    bluesky_post = generate_structured(
        compose_bluesky_post(body, url).content, BlueSkyPost
    )
    bluesky_text = bluesky_post.format_post()

    print(
        json.dumps(
            {
                "slug": slug,
                "title": post["title"],
                "url": url,
                "linkedin": linkedin_text,
                "bluesky": bluesky_text,
            }
        )
    )


if __name__ == "__main__":
    main()
