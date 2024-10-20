"""Pydantic models for each of the social media posts."""

from typing import List

from pydantic import BaseModel, Field, model_validator


class LinkedInPostSection(BaseModel):
    content: str = Field(..., description=(
        "The content of a section in the LinkedIn post"
    ))


class LinkedInPost(BaseModel):
    hook: str = Field(..., description=(
        "A short, thought-provoking statement to grab attention. "
        "If it is a question, start with 'What' or 'Why' directly, "
        "don't do things like 'Ever wondered what...'."
    ))
    teaser: str = Field(..., description=(
        "A statement closer to the topic, "
        "likely to be cut off by the default UI post fold."
    ))
    main_content: List[LinkedInPostSection] = Field(..., description=(
        "The meat of the content, broken into digestible sections."
    ))
    call_to_action: str = Field(..., description=(
        "A call to action for readers. I usually want people to read the blog post, "
        "leave a like, comment, and share."
    ))
    ending_question: str = Field(..., description=(
        "An ending question for the post reader to engage with via comments."
    ))
    hashtags: List[str] = Field(..., description=(
        "A list of hashtags to be included in the LinkedIn post. "
        "They should always begin with '#'."
    ))

    @model_validator(mode="after")
    def validate_content(self):
        """Validate the structure and content of the LinkedIn post."""
        # Validate hashtags
        for hashtag in self.hashtags:
            if not hashtag.startswith("#"):
                raise ValueError(f"Hashtag '{hashtag}' must start with '#'.")

        return self

    def format_post(self) -> str:
        """Format the LinkedIn post content."""
        post_content = f"{self.hook}\n\n{self.teaser}\n\n"
        for section in self.main_content:
            post_content += f"{section.content}\n\n"
        post_content += self.call_to_action

        post_content += "\n\n" + self.ending_question

        # Add hashtags to the post content, ensuring they are all lowercase
        post_content += (
            "\n\n" + " ".join([hashtag.lower() for hashtag in self.hashtags])
        )
        return post_content


class TwitterPost(BaseModel):
    content: str = Field(..., description="The content of the Twitter post")

    @model_validator(mode="after")
    def validate_content(self):
        """Validate correctness of the content."""

        # Cannot be >280 characters long
        if len(self.content) > 280:
            raise ValueError("Content must be 280 characters or less.")

        return self


class SubstackPost(BaseModel):
    content: str = Field(..., description="The content of the Substack post")


class Summary(BaseModel):
    content: str = Field(..., description="The content of the summary")


class Tags(BaseModel):
    content: list[str] = Field(..., description="The list of tags")

    @model_validator(mode="after")
    def validate_tags(self):
        """Validate correctness of the tags."""

        # Tags should be 2 words or less
        problematic_tags = [tag for tag in self.content if len(tag.split()) > 2]
        if problematic_tags:
            raise ValueError(
                "Tags must be 2 words or less, but found: "
                f"{', '.join(problematic_tags)}"
            )

        # Tags should have no hyphens (this is a way to cheat two word validation)
        problematic_tags = [tag for tag in self.content if "-" in tag]
        if problematic_tags:
            raise ValueError(
                f"Tags cannot contain hyphens, but found: {', '.join(problematic_tags)}"
            )

        # There should be 10 tags
        if len(self.content) < 10:
            raise ValueError("There should be at least 10 tags.")

        # 7 of the 10 should be one word long
        one_word_tags = [tag for tag in self.content if len(tag.split()) == 1]
        if len(one_word_tags) < 7:
            raise ValueError("There should be at least 7 one-word tags.")

        return self
