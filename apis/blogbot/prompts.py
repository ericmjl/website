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
    Please return this for me in JSON format using the following schema:

        "post_text": "post text goes here"
    """


@prompt
def compose_patreon_post(text):
    """This is a blog post that I just wrote.

        {{ text }}

    Please compose for me a patreon post
    that encourages my readers to read it and comment on it.
    Ensure that there is a call to action to interact with the post after reading
    to react with it, comment on it, or share the post with others.
    Include hashtags inline with the patreon post and at the end of the post too.
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

    Please return this for me in JSON format using the following schema:

        "post_text": "post text goes here"

    """
    return prompt


@prompt
def compose_substack_post(text):
    """This is a blog post that I just wrote:

        {{ text }}

    Please compose for me a post on Substack
    that entices my readers on Substack to read it.
    I usually like to open off with a question that the post answers.
    Ensure that there is a call to action for the reader,
    such as forwarding the post to others they think may benefit from it.
    Ensure that there is a placeholder for me
    to paste in the URL to the blog post.

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
    My usual tone is to write in first person,
    so please do so as well.
    For example, I usually start with "In this blog post".
    Return as JSON with key='post_text'
    and value=<the summary>.
    Ensure that it is valid JSON!
    """
