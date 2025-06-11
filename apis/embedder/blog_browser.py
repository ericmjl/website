# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot[all]==0.12.7",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import llamabot as lmb
    import marimo as mo
    from lr_parser import parse_lr_content

    return Path, lmb, mo, parse_lr_content


@app.cell
def _(mo):
    query = mo.ui.text(
        placeholder="Find anything you're looking for in Eric's blog",
        full_width=True,
        label="Search for blog posts.",
    )
    query
    return (query,)


@app.cell
def _(Path, lmb, query):
    docstore = lmb.components.docstore.LanceDBDocStore(
        table_name="ericmjl-blog-posts",
        storage_path=Path("./lancedb/"),
    )

    docs = []
    if query.value:
        docs = docstore.retrieve(query.value)
    return (docs,)


@app.cell
def _(docs, parse_lr_content):
    docs_parsed = [parse_lr_content(doc) for doc in docs]
    if docs_parsed:
        docs_parsed[0]
    return (docs_parsed,)


@app.cell
def _():
    return


@app.cell
def _(docs_parsed, mo):
    options = []
    dropdown = None
    ms = None
    if docs_parsed:
        options = [d["title"] for d in docs_parsed]
        dropdown = mo.ui.dropdown(
            options,
            value=options[0],
            label="Select post to preview",
            full_width=False,
        )

    if options:
        ms = mo.ui.multiselect(
            options=options, value=[options[0]], label="Select posts to chat"
        )
    return dropdown, ms


@app.cell
def _(dropdown):
    dropdown
    return


@app.cell
def _(docs_parsed, dropdown, mo):
    blog_text = None
    if dropdown.value:
        blog_text = [d["body"] for d in docs_parsed if d["title"] == dropdown.value][0]

    mo.md(blog_text)
    return


@app.cell
def _(docs_parsed, ms):
    import json

    selected_texts = [json.dumps(d) for d in docs_parsed if d["title"] in ms.value]
    return (selected_texts,)


@app.cell
def _(lmb, selected_texts):
    from typing import List

    @lmb.prompt("system")
    def selected_blog_chatbot(blog_posts: List[str]):
        """You are a bot that answers questions based on the following blog posts:

        ---
        {% for post in blog_posts %}
        {{ post }}
        ---
        {% endfor %}

        Do not hallucinate, use only these blog posts.
        """

    blog_chat_bot = lmb.SimpleBot(selected_blog_chatbot(selected_texts))

    # blog_chat_bot("Tell me aobut the blog post.")
    return (blog_chat_bot,)


@app.cell
def _(ms):
    ms
    return


@app.cell
def _(blog_chat_bot, mo):
    def blog_chat_bot_model(messages, config):
        return blog_chat_bot(messages[-1].content).content

    mo.ui.chat(blog_chat_bot_model)
    return


if __name__ == "__main__":
    app.run()
