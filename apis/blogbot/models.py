"""Pydantic models for each of the social media posts."""

from typing import List

from pydantic import BaseModel, Field, model_validator


class LinkedInPostSection(BaseModel):
    content: str = Field(
        ..., description=("The content of a section in the LinkedIn post")
    )


class LinkedInPost(BaseModel):
    hook: str = Field(
        ...,
        description=(
            "A short, thought-provoking statement to grab attention. "
            "If it is a question, start with 'What' or 'Why' directly, "
            "don't do things like 'Ever wondered what...'."
        ),
    )
    teaser: str = Field(
        ...,
        description=(
            "A statement closer to the topic, "
            "likely to be cut off by the default UI post fold."
        ),
    )
    main_content: List[LinkedInPostSection] = Field(
        ..., description=("The meat of the content, broken into digestible sections.")
    )
    call_to_action: str = Field(
        ...,
        description=(
            "A call to action for readers. "
            "I usually want people to read the blog post, "
            "leave a like, comment, and share."
        ),
    )
    ending_question: str = Field(
        ...,
        description=(
            "An ending question for the post reader to engage with via comments."
        ),
    )
    hashtags: List[str] = Field(
        ...,
        description=(
            "A list of hashtags to be included in the LinkedIn post. "
            "They should always begin with '#'."
        ),
    )

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
        post_content += "\n\n" + " ".join(
            [hashtag.lower() for hashtag in self.hashtags]
        )
        return post_content


class TwitterPost(BaseModel):
    hook: str = Field(
        ...,
        description=(
            "A short, attention-grabbing statement or question "
            "that covers the ideas in the blog post."
        ),
    )
    body: str = Field(
        ...,
        description=(
            "The body of the Twitter post. "
            "It should be a concise summary of the blog post."
        ),
    )
    call_to_action: str = Field(
        ...,
        description=(
            "A call to action for readers to read, share and re-post. "
            "Do not include URL!"
        ),
    )
    hashtags: List[str] = Field(
        ...,
        max_items=2,
        description="A list of hashtags (max 2) to be included in the Twitter post",
    )
    url: str = Field(
        ..., description="The URL to click on for more information. Just the raw URL."
    )

    @model_validator(mode="after")
    def validate_content(self):
        """Validate the structure and content of the Twitter post."""
        errors = []

        # Validate hashtags
        if len(self.hashtags) > 2:
            errors.append("Twitter post can have a maximum of 2 hashtags.")
        for hashtag in self.hashtags:
            if not hashtag.startswith("#"):
                errors.append(f"Hashtag '{hashtag}' must start with '#.'")

        # Validate total length
        total_content = self.format_post(with_url=False)
        if len(total_content) > 280:
            errors.append("Total content must be 280 characters or less.")
        if len(total_content) < 200:
            errors.append("Total content must be at least 200 characters.")

        if self.url in self.call_to_action:
            errors.append("URL should not be present in the call to action.")

        if not self.url.startswith("http://") and not self.url.startswith("https://"):
            errors.append("URL must start with 'http://' or 'https://'.")

        if errors:
            raise ValueError(", ".join(errors))

        return self

    def format_post(self, with_url: bool = True) -> str:
        """Format the Twitter post content."""
        post_content = f"{self.hook} {self.body} {self.call_to_action}"
        if with_url:
            post_content += f" {self.url}"
        post_content += f" {' '.join(self.hashtags)}"

        print(f"Post content: {post_content}")
        print(f"Post content length: {len(post_content)}")

        return post_content


class SubstackSection(BaseModel):
    content: str = Field(
        ..., description="The content of a section in the Substack post"
    )


class SubstackPost(BaseModel):
    title: str = Field(..., description="The title of the Substack post")
    subtitle: str = Field(
        ...,
        description=(
            "A brief subtitle or teaser for the post. "
            "An example that I really like is: "
            "'Alternatively titled: <something unpretentious and fun goes here>.'"
        ),
    )
    introduction: str = Field(
        ..., description="An introductory paragraph to hook the reader"
    )
    main_content: List[SubstackSection] = Field(
        ..., description="The main body of the post"
    )
    conclusion: str = Field(
        ...,
        description=("A concluding paragraph summarizing key points of the blog post."),
    )
    call_to_action: str = Field(
        ...,
        description="A call to action for readers, such as subscribing or sharing",
    )

    signoff: str = Field(
        ...,
        description=(
            "A signoff to end the post. Use one of "
            "Cheers\nEric or Happy Coding\nEric, "
            "the latter being suited to posts about coding."
        ),
    )

    @model_validator(mode="after")
    def validate_content(self):
        """Validate the structure and content of the Substack post."""
        errors = []

        # Validate content length
        total_content = self.format_post()
        if len(total_content) < 500:
            errors.append("Total content should be at least 500 characters.")

        if errors:
            raise ValueError(", ".join(errors))

        return self

    def format_post(self) -> str:
        """Format the Substack post content."""
        post_content = f"# {self.title}\n\n"
        post_content += f"*{self.subtitle}*\n\n"
        post_content += f"{self.introduction}\n\n"

        for section in self.main_content:
            post_content += f"{section.content}\n\n"

        post_content += f"{self.conclusion}\n\n"
        post_content += f"*{self.call_to_action}*\n\n"
        post_content += f"{self.signoff}"
        return post_content


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
