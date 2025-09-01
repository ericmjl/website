"""Prompts for my blog bot."""

from llamabot.prompt_manager import prompt


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
def compose_patreon_post(text, url):
    """This is a blog post that I just wrote.

        {{ text }}

    It came from the following url: {{ url }}.

    Based on the blog post, please compose for me a patreon post
    that encourages my Patreon supporters to read it.
    Ensure that it is written in first-person, humble, and inviting tone.
    Ensure that there is a placeholder for me to paste in the URL,
    which reads as "Please find the preview [here](url goes here)
    before it goes live on my blog."
    Sign off with "Cheers,\nEric".
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
    return prompt


# For backwards compatibility, keep compose_twitter_post available
compose_twitter_post = compose_bluesky_post


@prompt(role="user")
def compose_substack_post(text, url):
    """This is a blog post that I just wrote:

        {{ text }}

    It came from the following url: {{ url }}.

    Please compose for me a Substack post that follows compelling Substack best practices.

    **CRITICAL**: Match the tone and style of the original blog post. If the blog post is technical and direct, be technical and direct. If it's conversational and personal, be conversational and personal. Mirror the author's voice and writing style throughout.

    Guidelines:
    - Start with a clear purpose and hook - use an interesting question, story, or bold statement
    - Be authentic and genuine - share your thinking process and journey, don't try to be perfect
    - Structure for clarity with logical flow and clear takeaways
    - Encourage engagement by asking thoughtful questions that invite replies
    - Provide generous value through insights, resources, or personal stories
    - Use first-person voice and maintain the same tone as the original blog post
    - Address readers as "Hello fellow datanistas!" but adapt if the blog post suggests a different audience
    - Include the blog post URL using Markdown syntax: [link text]({{ url }})
    - End with a clear call to action that helps readers (read full post, share, subscribe)
    - Use authentic sign-off: "Cheers, Eric" or "Happy Coding, Eric" (for coding posts)

    Focus on creating a post that's clear, honest, thoughtfully structured, and written with real people in mind. Make it part of a conversation, not just a broadcast.
    """  # noqa: E501


@prompt(role="user")
def compose_tags(text):
    """Generate for me 10 tags for this blog post.
    Maximum two words.
    All lowercase.
    No `#` symbol is needed.
    Spaces are okay, so for example, if you have "webdevelopment",
    you can change it to "web development".
    Here is the blog post:

        {{ text }}
    """


@prompt(role="user")
def compose_summary(text, url):
    """
    Here is my blog post:

        {{ text }}

    It came from the following url: {{ url }}.

    I need a summary of the post in 100 words or less.
    Please write in first person.
    Please start the summary with, "In this blog post".
    Please end the summary with a question that entices the reader to read on.
    """


@prompt(role="system")
def bannerbot_dalle_prompter_sysprompt():
    """
    **As 'Prompt Designer',**
    your role is to create highly detailed and imaginative prompts for DALL-E,
    designed to generate banner images for blog posts in a watercolor style,
    with a 16:4 aspect ratio.

    You will be given a chunk of text or a summary that comes from the blog post.
    Your task is to translate the key concepts, ideas,
    and themes from the text into an image prompt.

    **Guidelines for creating the prompt:**
    - Use vivid and descriptive language to specify the image's mood, colors,
      composition, and style.
    - Vary your approach significantly between prompts - avoid repetitive patterns,
      elements, or compositions that could make images look similar.
    - Explore diverse watercolor techniques: washes, wet-on-wet, dry brush,
      salt effects, splattering, or layered glazes.
    - Consider different artistic styles within watercolor: impressionistic,
      expressionistic, minimalist, detailed botanical, atmospheric, or abstract.
    - Vary the color palettes: warm vs cool tones, monochromatic vs complementary,
      muted vs vibrant, or seasonal color schemes.
    - Mix different compositional approaches: centered focal points, rule of thirds,
      diagonal compositions, or asymmetrical balance.
    - Incorporate varied symbolic elements: natural objects, architectural forms,
      organic shapes, geometric patterns, or conceptual representations.
    - Focus on maximizing the use of imagery and symbols to represent ideas,
      avoiding any inclusion of text or character symbols in the image.
    - If the text is vague or lacks detail, make thoughtful and creative assumptions
      to create a compelling visual representation.

    The prompt should be suitable for a variety of blog topics,
    evoking an emotional or intellectual connection to the content.
    Ensure the description specifies the watercolor art style,
    the wide 16:4 banner aspect ratio,
    and your chosen artistic approach.

    **Example Output Prompts (showing variety):**

    Example 1 (Minimalist): "A minimalist watercolor composition in 16:4 aspect ratio,
    featuring a single elegant tree branch with delicate cherry blossoms against a soft,
    pale background. The painting uses a limited palette of soft pinks and creams,
    with subtle watercolor washes creating gentle atmospheric depth."

    Example 2 (Expressionistic): "A dynamic watercolor painting in 16:4 aspect ratio,
    with bold, gestural brushstrokes in deep blues and purples creating an energetic
    abstract composition. The paint flows freely across the surface, suggesting movement
    and creativity through organic, flowing forms and vibrant color interactions."

    Example 3 (Detailed): "A detailed watercolor botanical study in 16:4 aspect ratio,
    featuring intricate leaves and flowers rendered with precise brushwork and layered
    glazes. The composition uses a rich, earthy palette with careful attention to
    light and shadow, creating depth through multiple transparent washes."

    Do **NOT** include any text or character symbols in the image description.
    """


@prompt(role="user")
def compose_feedback_revision(
    original_content, feedback_request, post_type, blog_text, blog_url
):
    """This is a previously generated {{ post_type }} post:

    {{ original_content }}

    The user has provided this feedback for improvement:

    {{ feedback_request }}

    Please revise the {{ post_type }} post based on the feedback while maintaining the structure and requirements.

    For context, here is the original blog post:

    {{ blog_text }}

    Blog URL: {{ blog_url }}

    Please generate a revised version that addresses the feedback while following the same JSON structure and guidelines as the original.
    """  # noqa: E501
