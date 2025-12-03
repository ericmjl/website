"""Pydantic models for each of the social media posts."""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class LinkedInPostSection(BaseModel):
    content: str = Field(
        ..., description=("The content of a section in the LinkedIn post")
    )
    section_type: Optional[str] = Field(
        None, description="Type: story, insight, lesson, example"
    )


class LinkedInHook(BaseModel):
    """Implements Kallaway's 3-Step Hook Formula"""

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
            "Scroll-Stop Interjection: Use a bold, surprising, or emotionally charged statement "  # noqa: E501
            "that interrupts scrolling. Could be a shocking fact, strong opinion, or unexpected insight. "  # noqa: E501
            "Jolt the viewer out of autopilot and make them pay attention."
        ),
    )
    line3: str = Field(
        ...,
        description=(
            "Curiosity Gap: Create a gap between what the viewer knows and wants to know. "  # noqa: E501
            "Tease an answer, reveal, or transformation without giving it away. "
            "Compel the viewer to click 'see more' to satisfy their curiosity."
        ),
    )


class LinkedInAuthorityElement(BaseModel):
    """Support for authority and trust building"""

    story_type: str = Field(
        ..., description="Type: personal_story, lesson_learned, failure, success"
    )
    content: str = Field(..., description="The authority-building content")
    specific_example: Optional[str] = Field(
        None, description="Concrete example demonstrating expertise"
    )


class LinkedInPost(BaseModel):
    # Enhanced hook structure for 3-line hack
    hook: LinkedInHook = Field(
        ..., description="Three-line hook following the 3-Line Hack strategy"
    )

    # Authority building elements
    authority_elements: List[LinkedInAuthorityElement] = Field(
        default_factory=list,
        description="Personal stories, lessons, or insights that build authority and trust",  # noqa: E501
    )

    # Main content with enhanced structure
    main_content: List[LinkedInPostSection] = Field(
        ..., description="The meat of the content, broken into digestible sections"
    )

    # Enhanced engagement strategy
    call_to_action: str = Field(
        ...,
        description=(
            "Call to action focused on relationship building, not just broadcasting. "
            "I usually want people to read the blog post, leave a like, comment, and share."  # noqa: E501
        ),
    )

    ending_question: str = Field(
        ...,
        description=(
            "Specific question designed to generate thoughtful comments and discussion"
        ),
    )

    # Enhanced hashtag strategy
    hashtags: List[str] = Field(
        ...,
        max_items=5,
        description=(
            "List of relevant hashtags, including some outside main niche if valuable. "
            "They should always begin with '#'."
        ),
    )

    # Engagement metadata
    engagement_intent: str = Field(
        default="relationship_building",
        description="Primary intent: relationship_building, community_discussion, authority_demonstration",  # noqa: E501
    )

    @model_validator(mode="after")
    def validate_content(self):
        """Enhanced validation for LinkedIn best practices"""
        # Existing hashtag validation
        for hashtag in self.hashtags:
            if not hashtag.startswith("#"):
                raise ValueError(f"Hashtag '{hashtag}' must start with '#'.")

        # Validate 3-line hook structure
        if len(self.hook.line1) > 60:
            raise ValueError("Hook line 1 should be punchy and under 60 characters")

        # Ensure authority elements are specific
        for element in self.authority_elements:
            if (
                element.story_type in ["lesson_learned", "failure", "success"]
                and not element.specific_example
            ):
                raise ValueError(
                    f"Authority element of type '{element.story_type}' should include specific example"  # noqa: E501
                )

        return self

    def format_post(self) -> str:
        """Enhanced formatting following LinkedIn best practices"""
        # Start with 3-line hook
        post_content = f"{self.hook.line1}\n{self.hook.line2}\n{self.hook.line3}\n\n"

        # Add authority elements if present
        for element in self.authority_elements:
            post_content += f"{element.content}\n\n"
            if element.specific_example:
                post_content += f"{element.specific_example}\n\n"

        # Add main content sections
        for section in self.main_content:
            post_content += f"{section.content}\n\n"

        # Add call to action
        post_content += f"{self.call_to_action}\n\n"

        # Add ending question
        post_content += f"{self.ending_question}\n\n"

        # Add hashtags
        post_content += " ".join([hashtag.lower() for hashtag in self.hashtags])

        return post_content


class BlueSkyPost(BaseModel):
    strong_hook: str = Field(
        ...,
        description=(
            "Context-Lean Setup: Start with minimal context, jump straight to the core message. "  # noqa: E501
            "Use a bold, surprising, or emotionally charged statement that interrupts scrolling. "  # noqa: E501
            "Create immediate curiosity gap - tease value without revealing everything. "  # noqa: E501
            "Avoid bland intros and long-winded setups."
        ),
    )
    clear_stance: str = Field(
        ...,
        description=(
            "A direct, clear stance or unique insight. Don't be afraid to be bold or controversial. "  # noqa: E501
            "Avoid too much nuance - clarity and conviction get noticed."
        ),
    )
    value_delivery: str = Field(
        ...,
        description=(
            "Deliver value, emotion, or entertainment. Use specific details, numbers, or examples "  # noqa: E501
            "for credibility and impact. Should teach something, make people laugh, or evoke feeling."  # noqa: E501
        ),
    )
    call_to_action: Optional[str] = Field(
        None,
        description=(
            "Optional engaging call to action or thought-provoking line. "
            "Examples: 'Agree or disagree?' 'What's your take?' or a memorable punchline. "  # noqa: E501
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

    # Engagement metadata
    post_intent: str = Field(
        default="value_delivery",
        description="Primary intent: value_delivery, emotion, entertainment, controversy",  # noqa: E501
    )
    authenticity_check: bool = Field(
        default=True,
        description="Confirms this represents genuine beliefs and authentic voice",
    )

    @model_validator(mode="after")
    def validate_content(self):
        """Validate the structure and content of the BlueSky post."""
        errors = []

        # Validate hashtags
        if len(self.hashtags) > 2:
            errors.append("BlueSky post can have a maximum of 2 hashtags.")
        for hashtag in self.hashtags:
            if not hashtag.startswith("#"):
                errors.append(f"Hashtag '{hashtag}' must start with '#.'")

        # Validate total length
        # (283 chars excluding URL to account for buff.ly shortening)
        total_content = self.format_post(with_url=False)
        if len(total_content) > 283:
            errors.append(
                "Total content must be 283 characters or less (excluding URL to account for buff.ly link shortening)."  # noqa: E501
            )
        if len(total_content) < 100:
            errors.append("Total content must be at least 100 characters for impact.")

        if self.url in self.call_to_action if self.call_to_action else "":
            errors.append("URL should not be present in the call to action.")

        if not self.url.startswith("http://") and not self.url.startswith("https://"):
            errors.append("URL must start with 'http://' or 'https://'.")

        # Validate hook strength
        weak_openers = [
            "ever wondered",
            "have you ever",
            "i think that",
            "maybe we should",
        ]
        if any(opener in self.strong_hook.lower() for opener in weak_openers):
            errors.append(
                "Hook should be bold and direct, avoid weak openers like 'Ever wondered...'"  # noqa: E501
            )

        # Validate stance clarity
        if len(self.clear_stance.split()) > 30:
            errors.append(
                "Clear stance should be concise - under 30 words for maximum impact"
            )

        # Validate authenticity
        if not self.authenticity_check:
            errors.append("Post must represent authentic beliefs and voice")

        if errors:
            raise ValueError(", ".join(errors))

        return self

    def format_post(self, with_url: bool = True) -> str:
        """Format the BlueSky post following compelling post template."""
        # Follow template: [Strong hook]. [Clear stance]. [Value delivery]. [Optional CTA]  # noqa: E501
        post_content = f"{self.strong_hook} {self.clear_stance} {self.value_delivery}"

        if self.call_to_action:
            post_content += f" {self.call_to_action}"

        if with_url:
            post_content += f" {self.url}"

        post_content += f" {' '.join(self.hashtags)}"

        print(f"Post content: {post_content}")
        print(f"Post content length: {len(post_content)}")

        return post_content


# For backwards compatibility, keep TwitterPost available
TwitterPost = BlueSkyPost


class SubstackSection(BaseModel):
    content: str = Field(
        ..., description="The content of a section in the Substack post"
    )


class TitleVariant(BaseModel):
    """A title variant for Substack title testing."""

    title: str = Field(
        ...,
        description=(
            "The title text. Should be clear, compelling, and optimized "
            "for open rates. Must accurately represent the content (no clickbait)."
        ),
    )
    variant_type: str = Field(
        ...,
        description=(
            "The type of variant and principle being tested. "
            "Examples: 'question_based', 'statement_based', 'emotional_appeal', "
            "'factual_direct', 'curiosity_gap', 'how_to', 'numbered_list', "
            "'short_punchy', 'longer_contextual'. "
            "Should describe what makes this variant different."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "Brief explanation of why this variant might work well and what hypothesis "
            "it's testing. Should reference the principle (e.g., "
            "'Tests whether question-based titles create more curiosity' or "
            "'Tests if emotional appeal increases urgency')."
        ),
    )


class SubstackPost(BaseModel):
    title: str = Field(
        ...,
        description=(
            "Primary recommended title that matches the blog post title. "
            "This is the main title to use, but you should also generate "
            "title variants for testing. Should be clear, compelling, and intriguing."
        ),
    )
    title_variants: List[TitleVariant] = Field(
        ...,
        min_items=2,
        max_items=4,
        description=(
            "Alternative title variants optimized for Substack title testing. "
            "Generate 2-4 variants that test different principles: "
            "- Question-based vs statement-based "
            "- Emotional appeal vs factual/direct "
            "- Short punchy vs longer contextual "
            "- Curiosity gap vs clear value proposition "
            "- Different emotional tones (urgency, excitement, exclusivity) "
            "Each variant should be meaningfully different, not just minor "
            "word changes. All variants must accurately represent the content "
            "(no clickbait)."
        ),
    )
    subtitle: str = Field(
        ...,
        description=(
            "Brief subtitle that adds context or intrigue. "
            "Examples: 'Alternatively titled: <something unpretentious and fun>', "
            "or a hook that teases the value readers will get."
        ),
    )
    hook_introduction: str = Field(
        ...,
        description=(
            "Strong opening that hooks the reader immediately. "
            "Start with an interesting question, story, or bold statement. "
            "Be authentic and jump straight to something engaging. "
            "Match the tone and style of the original blog post."
        ),
    )
    purpose_statement: str = Field(
        ...,
        description=(
            "Clear statement of what this post is about and why it matters. "
            "Help readers understand the central idea and what they'll gain. "
            "Keep it focused and authentic to your voice."
        ),
    )
    main_content: SubstackSection = Field(
        ...,
        description=(
            "Well-structured content that provides genuine value. "
            "Share useful insights, personal stories, or resources. "
            "Use logical flow with clear paragraphs. Include your thinking process and journey. "  # noqa: E501
            "Give enough detail to be valuable while creating curiosity about the full blog post."  # noqa: E501
        ),
    )
    key_takeaway: str = Field(
        ...,
        description=(
            "Clear, memorable takeaway that summarizes the core value. "
            "What's the main point readers should remember? "
            "Make it actionable and authentic to your perspective."
        ),
    )
    engagement_question: str = Field(
        ...,
        description=(
            "Thoughtful question that invites replies and discussion. "
            "Encourage interaction by asking about readers' experiences or perspectives. "  # noqa: E501
            "Make it specific and genuinely curious."
        ),
    )
    call_to_action: str = Field(
        ...,
        description=(
            "Clear call to action that provides value. "
            "Examples: read the full blog post, share with others, subscribe. "
            "Be generous in your ask - focus on how it helps the reader."
        ),
    )

    signoff: str = Field(
        ...,
        description=(
            "Authentic sign-off that matches your voice. "
            "Use 'Cheers\nEric' or 'Happy Coding\nEric' (for coding posts). "
            "Keep it consistent with your established voice."
        ),
    )

    # Substack best practices metadata
    tone_authenticity: str = Field(
        default="authentic_and_genuine",
        description="Tone should be authentic, imperfect, and genuine - matching blog post style",  # noqa: E501
    )

    value_focus: str = Field(
        default="generous_sharing",
        description="Focus on giving value generously through insights, resources, or stories",  # noqa: E501
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

    def format_post(self, include_title_variants: bool = True) -> str:
        """Format the Substack post content following best practices."""
        post_content = f"# {self.title}\n\n"

        # Add title variants section for Substack title testing
        if include_title_variants and self.title_variants:
            post_content += "## Title Variants for Testing\n\n"
            post_content += (
                "*Use these variants in Substack's title testing feature "
                "(up to 4 variants). Each tests a different principle to "
                "optimize open rates.*\n\n"
            )
            for i, variant in enumerate(self.title_variants, 1):
                post_content += f"**Variant {i}: {variant.title}**\n\n"
                post_content += f"*Type: {variant.variant_type}*\n\n"
                post_content += f"*Rationale: {variant.rationale}*\n\n"
            post_content += "---\n\n"

        post_content += f"*{self.subtitle}*\n\n"
        post_content += f"{self.hook_introduction}\n\n"
        post_content += f"{self.purpose_statement}\n\n"

        post_content += f"{self.main_content.content}\n\n"

        post_content += f"{self.key_takeaway}\n\n"
        post_content += f"{self.engagement_question}\n\n"
        post_content += f"*{self.call_to_action}*\n\n"
        post_content += f"{self.signoff}"
        return post_content


class Summary(BaseModel):
    content: str = Field(..., description="The content of the summary")


class DallEImagePrompt(BaseModel):
    content: str = Field(..., description="The content of the DALL-E image prompt")


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
