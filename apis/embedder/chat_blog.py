# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "duckdb==1.3.0",
#     "llamabot[all]==0.12.10",
#     "marimo",
#     "polars[pyarrow]==1.30.0",
#     "pyprojroot==0.3.0",
#     "pytest==8.4.0",
#     "sqlglot==26.25.3",
#     "vegafusion==2.0.2",
#     "vl-convert-python==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import llamabot as lmb

    return Path, lmb


@app.cell
def _(Path, lmb):
    chat_memory = lmb.components.docstore.LanceDBDocStore(
        storage_path=Path("./lancedb/"), table_name="ericbot-chat-history"
    )
    chat_memory.reset()
    return (chat_memory,)


@app.cell
def _(Path, chat_memory, lmb):
    @lmb.prompt("system")
    def ericbot_sysprompt():
        """You are an expert on the blog author Eric Ma.
        You will respond in the tone and voice of Eric,
        and in first-person voice as well.
        Use only the sources provided to you.
        If you do not know how to answer, respond as such.
        In your responses, provide citations from my blog.
        The blog's URL pattern is https://ericmjl.github.io/blog/<slug>/
        Do not repeat yourself.
        """

        # I later added the last instructions

    docstore = lmb.components.docstore.LanceDBDocStore(
        storage_path=Path("./lancedb/"), table_name="ericmjl-blog-posts"
    )

    qb = lmb.QueryBot(
        system_prompt=ericbot_sysprompt(),
        docstore=docstore,
        chat_memory=chat_memory,
        # api_base="https://ericmjl--ollama-service-ollamaservice-server.modal.run",
        # model_name="ollama_chat/llama3-gradient"
    )
    return (qb,)


@app.cell
def _(qb):
    qb("What does Eric have to say about the SciPy conference?")
    return


@app.cell
def _(qb):
    qb("Can you tell me more about your thoughts on data science hiring?")
    return


@app.cell
def _(qb):
    qb("what advice do you have for job seekers wanting to move into data science?")
    return


@app.cell
def _(qb):
    qb("What are your thoughts on leadership? Specifically servant leadership?")
    return


@app.cell
def _(mo):
    mo.md(r"""The following question is one in which I am observing hallucination.""")
    return


@app.cell
def _(qb):
    qb(
        "Which was the blog post(s) in which you talked about doing well at work?",
        n_results=20,
    )
    return


@app.cell
def _(qb):
    qb(
        "Search for me the blog posts that talk about my graduate school experiences and what I learned during that time.",  # noqa: E501
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    user_info = mo.md(
        """
        - What's your name?: {name}
        - When were you born?: {birthday}
        """
    ).batch(name=mo.ui.text(), birthday=mo.ui.date(value="2022-01-01"))
    return (user_info,)


@app.cell
def _(user_info):
    user_info
    return


@app.cell
def _(user_info):
    user_info.value["name"], user_info.value["birthday"]
    return


if __name__ == "__main__":
    app.run()
