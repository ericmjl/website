def compose_linkedin_post(text):
    prompt = f"""
This is a blog post that I just wrote.

{text}

Please compose for me a LinkedIn post
that entices my network on LinkedIn to read it.
Ensure that there is a call to action to interact with the post after reading
to react with it, comment on it, or share the post with others,
and to support my work on Patreon.
My Patreon link is https://patreon.com/ericmjl/
Include hashtags inline with the LinkedIn post and at the end of the post too.
Please return this for me in JSON format using the following schema:


    "post_text": "post text goes here"

    """
    return prompt


def compose_twitter_post(text):
    prompt = f"""
This is a blog post that I just wrote:

{text}

Please compose for me a Twitter post
that entices my followers on Twitter to read it.
Ensure that there is a call to action to interact with the post after reading it,
such as retweeting, commenting, or sharing it with others,
and to support my work on Patreon.
My Patreon link is https://patreon.com/ericmjl/
Include hashtags inline with the Twitter post.

Please return this for me in JSON format using the following schema:


    "post_text": "post text goes here"

    """
    return prompt


def compose_tags(text):
    prompt = f"""
Generate me up to 10 tags for this blog post.
Maximum two words. All lowercase. No `#` symbol is needed.
Spaces are okay.
Here is the blog post: {text}.
Return as JSON with key='post_text'
and value=<the tags concatenated as one string with linebreaks>.
Ensure that it is valid JSON!
    """
    return prompt


def compose_summary(text):
    prompt = f"""
Here is my blog post:

{text}

I need a summary of the post.
It should entice users to read the post
but should not be sensational.
My usual tone is to write in first person,
so please do so as well.
Return as JSON with key='post_text'
and value=<the summary>.
Ensure that it is valid JSON!
    """
    return prompt
