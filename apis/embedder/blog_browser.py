# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot[all]==0.12.7",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium", layout_file="layouts/blog_browser.grid.json")


@app.cell
def _():
    from pathlib import Path

    import llamabot as lmb
    import marimo as mo
    from lr_parser import parse_lr_content

    return Path, lmb, mo, parse_lr_content


@app.cell
def _(mo):
    query = mo.ui.text()
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
    docs_parsed[0]
    return (docs_parsed,)


@app.cell
def _():
    return


@app.cell
def _(docs_parsed, mo):
    options = [d["title"] for d in docs_parsed]
    options = mo.ui.dropdown(options)
    options
    return (options,)


@app.cell
def _(docs_parsed, mo, options):
    blog_text = None
    if options.value:
        blog_text = [d["body"] for d in docs_parsed if d["title"] == options.value][0]

    mo.md(blog_text)
    return


if __name__ == "__main__":
    app.run()
