"""Pydantic models for each of the social media posts."""

from pydantic import BaseModel, Field, model_validator


class LinkedInPost(BaseModel):
    content: str = Field(..., description="The content of the LinkedIn post")


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
