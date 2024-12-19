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
def compose_twitter_post(text, url):
    """This is a blog post that I just wrote:

        {{ text }}

    It came from the following url: {{ url }}.

    Please compose for me a Twitter post
    that entices my followers on Twitter to read it.
    I usually like to open off with a question that the post answers.
    Ensure that there is a call to action to interact with the post after reading it,
    such as retweeting, commenting, or sharing it with others.
    Include hashtags inline with the Twitter post.
    Hashtags should be all lowercase.
    Ensure that you insert the URL of the blog post in an appropriate place,
    using Markdown syntax to link to the post.
    Also ensure that it is written in first-person, humble, and inviting tone.
    """
    return prompt


@prompt(role="user")
def compose_substack_post(text, url):
    """This is a blog post that I just wrote:

        {{ text }}

    It came from the following url: {{ url }}.

    Please compose for me a Substack post
    that encourages my readers on Substack to read the blog post I just gave you.
    Please address my readers by opening with,
    "Hello fellow datanistas!"
    And then open off with a question that the post answers.
    Ensure that there is a call to action for the reader,
    such as forwarding the post to others they think may benefit from it.
    Ensure that you insert the URL of the blog post in an appropriate place,
    using Markdown syntax to link to the post.
    Also ensure that it is written in first-person, humble, and inviting tone.
    """


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
    - Incorporate abstract and symbolic elements that reflect the blog's topic,
      ensuring the image is visually engaging and creative.
    - Focus on maximizing the use of imagery and symbols to represent ideas,
      avoiding any inclusion of text or character symbols in the image.
    - If the text is vague or lacks detail, make thoughtful and creative assumptions
      to create a compelling visual representation.

    The prompt should be suitable for a variety of blog topics,
    evoking an emotional or intellectual connection to the content.
    Ensure the description specifies the watercolor art style,
    the wide 16:4 banner aspect ratio,
    and abstract or symbolic design.

    **Example Output Prompt:**
    "A serene watercolor landscape in a 16:4 aspect ratio,
    featuring a vibrant sunrise over rolling hills with abstract swirls
    symbolizing creativity and growth.
    The sky transitions from soft orange to pastel blue,
    with floating geometric shapes representing innovation
    and interconnectedness.
    The scene conveys optimism and inspiration."

    Do **NOT** include any text or character symbols in the image description.
    """
