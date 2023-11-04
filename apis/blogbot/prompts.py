"""Prompts for my blog bot."""
from llamabot.prompt_manager import prompt


@prompt
def compose_linkedin_post(text):
    """This is a blog post that I just wrote.

        {{ text }}

    Please compose for me a LinkedIn post
    that entices my network on LinkedIn to read it.
    I usually like to open off with a question that the post answers.
    Ensure that there is a call to action to interact with the post after reading
    to react with it, comment on it, or share the post with others.
    Include hashtags inline with the LinkedIn post and at the end of the post too.
    Hashtags should be all lowercase.
    Ensure that there is a placeholder for me to paste in the URL.
    Also ensure that it is written in first-person, humble, and inviting tone.
    Please return this for me in JSON format using the following schema:

        "post_text": "post text goes here"
    """


@prompt
def compose_patreon_post(text):
    """This is a blog post that I just wrote.

        {{ text }}

    Based on the blog post, please compose for me a patreon post
    that encourages my Patreon supporters to read it.
    Ensure that it is written in first-person, humble, and inviting tone.
    Ensure that there is a placeholder for me to paste in the URL,
    which reads as "Please find the preview [here](url goes here)
    before it goes live on my blog."
    Sign off with "Cheers,\nEric".
    Please return this for me in JSON format using the following schema:

        "post_text": "post text goes here"
    """


@prompt
def compose_twitter_post(text):
    """This is a blog post that I just wrote:

        {{ text }}

    Please compose for me a Twitter post
    that entices my followers on Twitter to read it.
    I usually like to open off with a question that the post answers.
    Ensure that there is a call to action to interact with the post after reading it,
    such as retweeting, commenting, or sharing it with others.
    Include hashtags inline with the Twitter post.
    Hashtags should be all lowercase.
    Ensure that there is a placeholder for me to paste in the URL.
    Also ensure that it is written in first-person, humble, and inviting tone.

    Please return this for me in JSON format using the following schema:

        "post_text": "post text goes here"

    """
    return prompt


@prompt
def compose_substack_post(text):
    """This is a blog post that I just wrote:

        {{ text }}

    Please compose for me a Substack post
    that encourages my readers on Substack to read the blog post I just gave you.
    Please address my readers by opening with,
    "Hello fellow datanistas!"
    And then open off with a question that the post answers.
    Ensure that there is a call to action for the reader,
    such as forwarding the post to others they think may benefit from it.
    Ensure that there is a placeholder for me
    to paste in the URL to the blog post.
    Also ensure that it is written in first-person, humble, and inviting tone.

    Please return this for me in JSON format using the following schema:

        "post_text": "post text goes here"
    """


@prompt
def compose_tags(text):
    """Generate for me 10 tags for this blog post.
    Maximum two words.
    All lowercase.
    No `#` symbol is needed.
    Spaces are okay, so for example, if you have "webdevelopment",
    you can change it to "web development".
    Here is the blog post: {{ text }}.
    Please return this for me in JSON format using the following schema:

        "post_text": "tag1,tag2,tag3,..."
    """


@prompt
def compose_summary(text):
    """
    Here is my blog post:

        {{ text }}

    I need a summary of the post in 100 words or less.
    Please write in first person.
    Please start the summary with, "In this blog post".
    Please end the summary with a question that entices the reader to read on.
    Return as JSON with key='post_text'
    and value=<the summary>.
    Ensure that it is valid JSON!
    """


@prompt
def fix_json(bad_json):
    """The following is a bad JSON that was returned by your sibling bot.

        {{ bad_json }}

    Can you fix it for me such that its contents are preserved
    but it is now valid JSON?
    Return only the JSON and nothing else.
    """
